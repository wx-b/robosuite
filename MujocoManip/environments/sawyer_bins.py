import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *
import MujocoManip.miscellaneous.utils as U


class SawyerBinsEnv(SawyerEnv):

    def __init__(self, 
                 gripper_type='TwoFingerGripper',
                 use_eef_ctrl=False,
                 table_size=(0.2, 0.4, 0.4),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 reward_shaping=False,
                 gripper_visualization=False,
                 n_each_object=1,
                 single_object_mode=False,
                 **kwargs):
        """
            @gripper_type, string that specifies the gripper type
            @use_eef_ctrl, position controller or default joint controllder
            @table_size, full dimension of the table
            @table_friction, friction parameters of the table
            @use_camera_obs, using camera observations
            @use_object_obs, using object physics states
            @camera_name, name of camera to be rendered
            @camera_height, height of camera observation
            @camera_width, width of camera observation
            @camera_depth, rendering depth
            @reward_shaping, using a shaping reward
            @gripper_visualization: visualizing gripper site
        """
        # initialize objects of interest
        self.n_each_object = n_each_object
        self.single_object_mode = single_object_mode

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(gripper_type=gripper_type,
                         use_eef_ctrl=use_eef_ctrl,
                         use_camera_obs=use_camera_obs,
                         camera_name=camera_name,
                         gripper_visualization=gripper_visualization,
                         **kwargs)

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = BinsArena()
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.4 + self.table_size[0] / 2,-0.3,0])

        self.ob_inits = [DefaultMilkObject, DefaultBreadObject, DefaultCerealObject, DefaultCanObject]
        self.vis_inits = [DefaultMilkVisualObject, DefaultBreadVisualObject, DefaultCerealVisualObject, DefaultCanVisualObject]
        self.item_names = ["Milk", "Bread", "Cereal", "Can"]
        self.item_names_org = list(self.item_names)

        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]),self.vis_inits[j]()))
        self.visual_objects = lst

        if self.single_object_mode:
            self.n_each_object = 1
            self.selected_bin = np.random.randint(len(self.ob_inits))
            self.ob_inits = [self.ob_inits[self.selected_bin]]
            self.item_names = [self.item_names[self.selected_bin]]
            lst = [(str(self.item_names[0])+'{}'.format(0), self.ob_inits[0]())]
        else:
            lst = []
            for i in range(self.n_each_object):
                for j in range(len(self.ob_inits)):
                    ob = self.ob_inits[j]()
                    lst.append((str(self.item_names[j])+'{}'.format(i), ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = BinsTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects, self.visual_objects)
        self.model.place_objects()
        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin2_body.get('pos'))
        self.bin_size = self.model.shelf_size

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.l_finger_geom_id = self.sim.model.geom_name2id('l_fingertip_g0')
        self.r_finger_geom_id = self.sim.model.geom_name2id('r_fingertip_g0')

        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                # obj_str = str(self.ob_inits[j]) + '{}'.format(i)
                obj_str = str(self.item_names[j]) + '{}'.format(i)
                self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
                self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

        # for checking distance to / contact with objects we want to pick up
        self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros((self.n_each_object, len(self.ob_inits)))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.ob_inits), 3))
        for j in range(len(self.ob_inits)):
            bin_id = j
            if self.single_object_mode:
                bin_id = self.selected_bin
            bin_x_low = self.bin_pos[0]
            bin_y_low = self.bin_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2 :
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[j, :] = [bin_x_low, bin_y_low, self.bin_pos[2]]

    def reward(self, action = None):
        # stages: reaching, grasping, lifting, dropping, lifting

        if self.reward_shaping:
            r_goal = 0.
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            for i in range(self.n_each_object):
                for j in range(len(self.ob_inits)):
                    # obj_str = str(self.ob_inits[j]) + '{}'.format(i)
                    obj_str = str(self.item_names[j]) + '{}'.format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    dist = np.linalg.norm(gripper_site_pos - obj_pos)
                    r_reach = 1 - np.tanh(10.0 * dist)
                    bin_id = j
                    if self.single_object_mode:
                        bin_id = self.selected_bin
                    r_obj_goal = int((not self.not_in_bin(obj_pos, bin_id)) and r_reach < 0.6)
                    self.objects_in_bins[i, j] = r_obj_goal
                    r_goal += r_obj_goal
            staged_rewards = self.staged_rewards()
            return r_goal + max(staged_rewards)

        else:
            # +1 reward for every object in the bin
            reward = 0.
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            for i in range(self.n_each_object):
                for j in range(len(self.ob_inits)):
                    # obj_str = str(self.ob_inits[j]) + '{}'.format(i)
                    obj_str = str(self.item_names[j]) + '{}'.format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    dist = np.linalg.norm(gripper_site_pos - obj_pos)
                    r_reach = 1 - np.tanh(10.0 * dist)
                    bin_id = j
                    if self.single_object_mode:
                        bin_id = self.selected_bin
                    obj_not_in_bin = self.not_in_bin(obj_pos, bin_id)
                    reward += int((r_reach < 0.6) and (not obj_not_in_bin))
                    self.objects_in_bins[i, j] = int(not obj_not_in_bin)
                            
            return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.

        Stages consist of reaching, grasping, lifting, and hovering.
        """

        # reach_mult = 0.1
        # grasp_mult = 0.25
        # lift_mult = 0.4
        # hover_mult = 0.7

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        objs_to_reach = []
        geoms_to_grasp = []
        target_bin_placements = []
        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                if self.objects_in_bins[i, j]:
                    continue
                obj_str = str(self.item_names[j]) + '{}'.format(i)
                objs_to_reach.append(self.obj_body_id[obj_str])
                geoms_to_grasp.append(self.obj_geom_id[obj_str])
                target_bin_placements.append(self.target_bin_placements[j])
        target_bin_placements = np.array(target_bin_placements)

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # get reaching reward via minimum distance to a target object
            target_object_pos = self.sim.data.body_xpos[objs_to_reach]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dists = np.linalg.norm(target_object_pos - gripper_site_pos.reshape(1, -1), axis=1)
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom1)
                if c.geom2 == self.l_finger_geom_id:
                    touch_left_finger = True
                if c.geom2 == self.r_finger_geom_id:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom2)
                if c.geom1 == self.l_finger_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.r_finger_geom_id:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.bin_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        ### hover reward for getting object above bin ###
        r_hover = 0.

        ### uncomment for left vs. right segmentation ###

        # if len(objs_to_reach):
        #     # segment objects into left of the bins and above the bins
        #     object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
        #     objects_above_bins = (object_xy_locs[:, 1] - (self.bin_pos[1] - self.bin_size[1] / 2.)) > 0
        #     objects_not_above_bins = (object_xy_locs[:, 1] - (self.bin_pos[1] - self.bin_size[1] / 2.)) <= 0
        #     dists = np.linalg.norm(target_bin_placements[:, :2] - object_xy_locs, axis=1)
        #     # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
        #     r_hover_all = np.zeros(len(objs_to_reach))
        #     r_hover_all[objects_above_bins] = lift_mult + (1 - np.tanh(10.0 * dists[objects_above_bins])) * (hover_mult - lift_mult)
        #     r_hover_all[objects_not_above_bins] = r_lift + (1 - np.tanh(10.0 * dists[objects_not_above_bins])) * (hover_mult - lift_mult)
        #     r_hover = np.max(r_hover_all)

        ### separate by being above target bin or not ###

        if len(objs_to_reach):
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
            y_check = np.abs(object_xy_locs[:, 1] - target_bin_placements[:, 1]) < self.bin_size[1] / 4.
            x_check = np.abs(object_xy_locs[:, 0] - target_bin_placements[:, 0]) < self.bin_size[0] / 4.
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(target_bin_placements[:, :2] - object_xy_locs, axis=1)
            # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(objs_to_reach))
            r_hover_all[objects_above_bins] = lift_mult + (1 - np.tanh(10.0 * dists[objects_above_bins])) * (hover_mult - lift_mult)
            r_hover_all[objects_not_above_bins] = r_lift + (1 - np.tanh(10.0 * dists[objects_not_above_bins])) * (hover_mult - lift_mult)
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin_pos[0]
        bin_y_low = self.bin_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0]/2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1]/2

        bin_x_high = bin_x_low + self.bin_size[0]/2
        bin_y_high = bin_y_low + self.bin_size[1]/2

        res = True
        if obj_pos[2] > self.bin_pos[2] and obj_pos[0] < bin_x_high and \
                obj_pos[0] > bin_x_low and obj_pos[1] < bin_y_high and \
                obj_pos[1] > bin_y_low  and obj_pos[2] < self.bin_pos[2] + 0.1 :
            res = False
        return res

    def _get_observation(self):
        """
            Adds hand_position, hand_velocity or 
            (current_position, current_velocity, target_velocity) of all targets
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

        # low-level object information
        if self.use_object_obs:

            ### TODO: everything is in world frame right now... ###

            # gripper orientation
            # di['gripper_pos'] = self.sim.data.get_body_xpos('right_hand')
            di['gripper_pos'] = self.sim.data.site_xpos[self.eef_site_id]
            di['gripper_quat'] = self.sim.data.get_body_xquat('right_hand')

            gripper_pose = U.pose2mat((di['gripper_pos'], di['gripper_quat']))
            world_pose_in_gripper = U.pose_inv(gripper_pose)

            ### TODO: should we transform these poses to robot base frame? ###
            for i in range(self.n_each_object):
                for j in range(len(self.item_names_org)): 
                    
                    if self.single_object_mode and self.selected_bin != j:
                        obj_str = str(self.item_names_org[j]) + '{}'.format(i)
                        di["{}_pos".format(obj_str)] = np.zeros(3)
                        di["{}_quat".format(obj_str)] = np.zeros(4)
                        di["{}_to_gripper_pos".format(obj_str)] = np.zeros(3) 
                        di["{}_to_gripper_quat".format(obj_str)] = np.zeros(4)
                        continue

                    obj_str = str(self.item_names_org[j]) + '{}'.format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    obj_quat = self.sim.data.body_xquat[self.obj_body_id[obj_str]]
                    di["{}_pos".format(obj_str)] = obj_pos
                    di["{}_quat".format(obj_str)] = obj_quat

                    object_pose = U.pose2mat((obj_pos, obj_quat))
                    rel_pose = U.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
                    rel_pos, rel_quat = U.mat2pose(rel_pose)
                    # gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
                    # di["gripper_to_{}".format(obj_str)] = gripper_site_pos - obj_pos 
                    di["{}_to_gripper_pos".format(obj_str)] = rel_pos 
                    di["{}_to_gripper_quat".format(obj_str)] = rel_quat

        # proprioception
        di['proprio'] = np.concatenate([
            np.sin(di['joint_pos']),
            np.cos(di['joint_pos']),
            di['joint_vel'],
            di['gripper_pos'],
        ])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.finger_names or \
               self.sim.model.geom_id2name(contact.geom2) in self.finger_names:
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        return False

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x : np.sum(np.square(x - self.sim.data.get_site_xpos('grip_site')))
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[self.object_site_ids] # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
