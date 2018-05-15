import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *


class BinsEnv(SawyerEnv):

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

        self.n_each_object = 2
        # ob_inits = [DefaultCerealObject, DefaultBreadObject, DefaultLemonObject, DefaultMilkObject]
        self.ob_inits = [DefaultMilkObject, DefaultBreadObject, DefaultCerealObject, DefaultCanObject]
        self.vis_inits = [DefaultMilkVisualObject, DefaultBreadVisualObject, DefaultCerealVisualObject, DefaultCanVisualObject]
        # ob_inits = [DefaultBreadObject, DefaultAtomizerObject, DefaultLemonObject]
        lst = []
        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                ob = self.ob_inits[j]()
                lst.append((str(self.ob_inits[j])+'{}'.format(i), ob))
        self.mujoco_objects = OrderedDict(lst)

        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]),self.vis_inits[j]()))
        self.visual_objects = lst

        self.n_objects = len(self.mujoco_objects)

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

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.4 + self.table_size[0] / 2,-0.3,0])

        # task includes arena, robot, and objects of interest
        self.model = BinsTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects, self.visual_objects)
        self.model.place_objects()
        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin2_body.get('pos'))
        self.bin_size = self.model.shelf_size

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id={}

        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                obj_str = str(self.ob_inits[j]) + '{}'.format(i)
                self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

    def reward(self, action = None):
        reward = 1
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                obj_str = str(self.ob_inits[j]) + '{}'.format(i)
                obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                dist = np.linalg.norm(gripper_site_pos - obj_pos)
                r_reach = 1 - np.tanh(10.0 * dist)
                if r_reach > 0.6 or self.not_in_bin(obj_pos,j):
                    reward = 0

        return reward

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin_pos[0]
        bin_y_low = self.bin_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0]/2
        if bin_id < 2 :
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
        # if self.use_object_obs:
        #     # position and rotation of the first cube
        #     cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        #     cubeA_quat = self.sim.data.body_xquat[self.cubeA_body_id]
        #     di['cubeA_pos'] = cubeA_pos
        #     di['cubeA_quat'] = cubeA_quat

        #     # position and rotation of the second cube
        #     cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        #     cubeB_quat = self.sim.data.body_xquat[self.cubeB_body_id]
        #     di['cubeB_pos'] = cubeB_pos
        #     di['cubeB_quat'] = cubeB_quat

        #     gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        #     di['gripper_to_cubeA'] = gripper_site_pos - cubeA_pos
        #     di['gripper_to_cubeB'] = gripper_site_pos - cubeB_pos
        #     di['cubeA_to_cubeB'] = cubeA_pos - cubeB_pos

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
