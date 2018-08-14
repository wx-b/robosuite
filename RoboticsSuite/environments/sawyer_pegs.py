import numpy as np
from collections import OrderedDict
from RoboticsSuite.miscellaneous import RandomizationError
from RoboticsSuite.environments.sawyer import SawyerEnv
from RoboticsSuite.environments.demo_sampler import DemoSampler
from RoboticsSuite.models import *
import RoboticsSuite.miscellaneous.utils as U
import random


class SawyerPegsEnv(SawyerEnv):
    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        use_eef_ctrl=False,
        table_size=(0.2, 0.4, 0.4),
        table_friction=None,
        use_camera_obs=True,
        use_object_obs=True,
        camera_name="frontview",
        reward_shaping=False,
        gripper_visualization=False,
        placement_initializer=None,
        n_each_object=1,
        single_object_mode=0,  # 0 full, 1 single obj with full obs 2 single with single obs
        selected_peg=None,
        demo_config=None,
        **kwargs
    ):
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
        # number of objects per category
        self.n_each_object = n_each_object
        self.single_object_mode = single_object_mode
        self.selected_peg = selected_peg
        self.obj_to_use = None
        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        self.demo_config = demo_config
        if self.demo_config is not None:
            self.demo_sampler = DemoSampler(
                self.demo_config.demo_file,
                self.demo_config,
                preload=self.demo_config.preload,
                number=self.demo_config.num_samples,
            )
        self.eps_reward = 0
        # placement initilizer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomPegsSampler(
                x_range=[-0.15, 0.],
                y_range=[-0.2, 0.2],
                z_range=[0.02, 0.10],
                ensure_object_boundary_in_range=False,
                z_rotation=True,
            )

        super().__init__(
            gripper_type=gripper_type,
            use_eef_ctrl=use_eef_ctrl,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            gripper_visualization=gripper_visualization,
            **kwargs
        )

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = PegsArena()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.4 + self.table_size[0] / 2, -0.15, 0])

        # define mujoco objects
        self.ob_inits = [DefaultSquareNutObject, DefaultRoundNutObject]
        self.item_names = ["SquareNut", "RoundNut"]
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[1] + "{}").format(0)
        self.ngeoms = [5, 9]

        lst = []
        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                ob = self.ob_inits[j]()
                lst.append((str(self.item_names[j]) + "{}".format(i), ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = PegsTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.placement_initializer,
        )
        self.model.place_objects()
        self.bin_pos = string_to_array(self.model.bin1_body.get("pos"))
        self.peg1_pos = string_to_array(self.model.peg1_body.get("pos"))  # square
        self.peg2_pos = string_to_array(self.model.peg2_body.get("pos"))  # round
        self.bin_size = self.model.shelf_size

    def clear_objects(self, obj):
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                # print(self.sim.model.get_joint_qpos_addr(obj_name))
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name)[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                obj_str = str(self.item_names[j]) + "{}".format(i)
                self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
                geom_ids = []
                for k in range(self.ngeoms[j]):
                    geom_ids.append(
                        self.sim.model.geom_name2id(obj_str + "-{}".format(k))
                    )
                self.obj_geom_id[obj_str] = geom_ids
        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()
        self.l_finger_geom_id = self.sim.model.geom_name2id("l_fingertip_g0")
        self.r_finger_geom_id = self.sim.model.geom_name2id("r_fingertip_g0")
        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros((self.n_each_object, len(self.ob_inits)))

    def _reset_from_random(self):
        # inherited class should reset positions of objects
        self.model.place_objects()
        if self.single_object_mode == 1:
            self.obj_to_use = (random.choice(self.item_names) + "{}").format(0)
            self.clear_objects(self.obj_to_use)
        elif self.single_object_mode == 2:
            self.obj_to_use = (self.item_names[self.selected_peg] + "{}").format(0)
            self.clear_objects(self.obj_to_use)

    def _reset_internal(self):
        super()._reset_internal()
        if self.demo_config is not None:
            self.demo_sampler.log_score(self.eps_reward)
            state = self.demo_sampler.sample()
            if state is None:
                self._reset_from_random()
            else:
                self.sim.set_state_from_flattened(state)
                self.sim.forward()
        else:
            self._reset_from_random()

    def reward(self, action=None):
        if self.reward_shaping:
            r_goal = 0.
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            for i in range(self.n_each_object):
                for j in range(len(self.ob_inits)):
                    obj_str = str(self.item_names[j]) + "{}".format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    dist = np.linalg.norm(gripper_site_pos - obj_pos)
                    r_reach = 1 - np.tanh(10.0 * dist)
                    r_obj_goal = int(self.on_peg(obj_pos, j) and r_reach < 0.6)
                    self.objects_on_pegs[i, j] = r_obj_goal
                    r_goal += r_obj_goal
            staged_rewards = self.staged_rewards()
            return r_goal + max(staged_rewards)

        else:
            # +1 for every object on a peg
            reward = 0
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            for i in range(self.n_each_object):
                for j in range(len(self.ob_inits)):
                    r_on_peg, r_lift = 0, 0
                    obj_str = str(self.item_names[j]) + "{}".format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    dist = np.linalg.norm(gripper_site_pos - obj_pos)
                    r_reach = 1 - np.tanh(10.0 * dist)
                    if r_reach < 0.6 and self.on_peg(obj_pos, j):
                        reward += 1.

        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        names_to_reach = []
        objs_to_reach = []
        geoms_to_grasp = []
        geoms_by_array = []

        for i in range(self.n_each_object):
            for j in range(len(self.ob_inits)):
                if self.objects_on_pegs[i, j]:
                    continue
                obj_str = str(self.item_names[j]) + "{}".format(i)
                names_to_reach.append(obj_str)
                objs_to_reach.append(self.obj_body_id[obj_str])
                geoms_to_grasp.extend(self.obj_geom_id[obj_str])
                geoms_by_array.append(self.obj_geom_id[obj_str])

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # reaching reward via minimum distance to the handles of the objects (the last geom of each nut)
            geom_ids = [elem[-1] for elem in geoms_by_array]
            target_geom_pos = self.sim.data.geom_xpos[geom_ids]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dists = np.linalg.norm(
                target_geom_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                if c.geom2 == self.l_finger_geom_id:
                    touch_left_finger = True
                if c.geom2 == self.r_finger_geom_id:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                if c.geom1 == self.l_finger_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.r_finger_geom_id:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.bin_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        ### hover reward for getting object above peg ###
        r_hover = 0.
        if len(objs_to_reach):
            r_hovers = np.zeros(len(objs_to_reach))
            for i in range(len(objs_to_reach)):
                if names_to_reach[i].startswith(self.item_names[0]):
                    peg_pos = self.peg1_pos[:2]
                elif names_to_reach[i].startswith(self.item_names[1]):
                    peg_pos = self.peg2_pos[:2]
                else:
                    raise Exception(
                        "Got invalid object to reach: {}".format(names_to_reach[i])
                    )
                ob_xy = self.sim.data.body_xpos[objs_to_reach[i]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (
                    hover_mult - lift_mult
                )
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = self.peg1_pos
        else:
            peg_pos = self.peg2_pos

        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.model.shelf_offset[2] + 0.05
        ):
            res = True
        return res

    def _get_observation(self):
        """
            Adds hand_position, hand_velocity or 
            (current_position, current_velocity, target_velocity) of all targets
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:

            ### TODO: everything is in world frame right now...

            # gripper orientation
            # di['gripper_pos'] = self.sim.data.get_body_xpos('right_hand')
            di["eef_pos"] = self.sim.data.site_xpos[self.eef_site_id]
            di["eef_quat"] = U.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            gripper_pose = U.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = U.pose_inv(gripper_pose)

            ### TODO: should we transform these poses to robot base frame?
            for i in range(self.n_each_object):
                for j in range(len(self.item_names_org)):

                    if self.single_object_mode == 2 and self.selected_peg != j:
                        # skip observations
                        continue

                    obj_str = str(self.item_names_org[j]) + "{}".format(i)
                    obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                    obj_quat = U.convert_quat(
                        self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                    )
                    di["{}_pos".format(obj_str)] = obj_pos
                    di["{}_quat".format(obj_str)] = obj_quat

                    object_pose = U.pose2mat((obj_pos, obj_quat))
                    rel_pose = U.pose_in_A_to_pose_in_B(
                        object_pose, world_pose_in_gripper
                    )
                    rel_pos, rel_quat = U.mat2pose(rel_pose)
                    di["{}_to_eef_pos".format(obj_str)] = rel_pos
                    di["{}_to_eef_quat".format(obj_str)] = rel_quat

            if self.single_object_mode == 1:
                # zero out other objs
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.obj_to_use:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_eef_pos".format(obj_str)] *= 0.0
                        di["{}_to_eef_quat".format(obj_str)] *= 0.0
        # proprioception
        di["proprio"] = np.concatenate(
            [
                np.sin(di["joint_pos"]),
                np.cos(di["joint_pos"]),
                di["joint_vel"],
                di["gripper_pos"],
                di["eef_pos"],
                di["eef_quat"],
            ]
        )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
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
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
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


class SawyerPegsSingleEnv(SawyerPegsEnv):
    """
    Easier version of task - place one round nut into its peg.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=1, **kwargs)


class SawyerPegsSquareEnv(SawyerPegsEnv):
    """
    Easier version of task - place one square nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "selected_peg" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, selected_peg=0, **kwargs)


class SawyerPegsRoundEnv(SawyerPegsEnv):
    """
    Easier version of task - place one round nut into its peg.
    """

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "selected_peg" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, selected_peg=1, **kwargs)
