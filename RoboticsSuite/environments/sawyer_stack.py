import numpy as np
from collections import OrderedDict
from RoboticsSuite.utils import RandomizationError
from RoboticsSuite.environments.sawyer import SawyerEnv
from RoboticsSuite.environments.demo_sampler import DemoSampler
from RoboticsSuite.models import *
import RoboticsSuite.utils as U
import pickle
import random


class SawyerStackEnv(SawyerEnv):
    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        use_eef_ctrl=False,
        table_size=(0.8, 0.8, 0.8),
        table_friction=None,
        use_camera_obs=True,
        use_object_obs=True,
        camera_name="frontview",
        reward_shaping=False,
        gripper_visualization=False,
        placement_initializer=None,
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

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.demo_config = demo_config
        if self.demo_config is not None:
            self.demo_sampler = DemoSampler(
                "demonstrations/sawyer-stack.pkl", self.demo_config
            )

        self.eps_reward = 0
        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
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

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            full_size=self.table_size, friction=self.table_friction
        )

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2, 0, 0])

        # initialize objects of interest
        cubeA = RandomBoxObject(
            size_min=[0.02, 0.02, 0.02], size_max=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1]
        )
        cubeB = RandomBoxObject(
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cubeA", cubeA), ("cubeB", cubeB)])
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self.cubeA_body_id = self.sim.model.body_name2id("cubeA")
        self.cubeB_body_id = self.sim.model.body_name2id("cubeB")
        self.l_finger_geom_id = self.sim.model.geom_name2id("l_fingertip_g0")
        self.r_finger_geom_id = self.sim.model.geom_name2id("r_fingertip_g0")
        self.cubeA_geom_id = self.sim.model.geom_name2id("cubeA")
        self.cubeB_geom_id = self.sim.model.geom_name2id("cubeB")

    def _reset_from_random(self):
        # inherited class should reset positions of objects
        self.model.place_objects()
        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def _reset_internal(self):
        super()._reset_internal()
        if self.demo_config is not None:
            self.demo_sampler.log_score(self.eps_reward)
            state = self.demo_sampler.sample()
            if state is None:
                self._reset_from_random()
            else:
                self.sim.set_state(state)
                self.sim.forward()
        else:
            self._reset_from_random()

    def reward(self, action):
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 1.0 if r_stack > 0 else 0.0

        self.eps_reward = max(reward, self.eps_reward)
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states
        """
        # reaching is successful when the gripper site is close to
        # the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # collision checking
        touch_left_finger = False
        touch_right_finger = False
        touch_cubeA_cubeB = False

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.l_finger_geom_id and c.geom2 == self.cubeA_geom_id:
                touch_left_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 == self.l_finger_geom_id:
                touch_left_finger = True
            if c.geom1 == self.r_finger_geom_id and c.geom2 == self.cubeA_geom_id:
                touch_right_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 == self.r_finger_geom_id:
                touch_right_finger = True
            if c.geom1 == self.cubeA_geom_id and c.geom2 == self.cubeB_geom_id:
                touch_cubeA_cubeB = True
            if c.geom1 == self.cubeB_geom_id and c.geom2 == self.cubeA_geom_id:
                touch_cubeA_cubeB = True

        # additional grasping reward
        if touch_left_finger and touch_right_finger:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top
        # by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_size[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(
                np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2])
            )
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and
        # the gripper is not holding the object
        r_stack = 0
        not_touching = not touch_left_finger and not touch_right_finger
        if not_touching and r_lift > 0 and touch_cubeA_cubeB:
            r_stack = 2.0

        return (r_reach, r_lift, r_stack)

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
            # position and rotation of the first cube
            cubeA_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id])
            cubeA_quat = U.convert_quat(
                np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw"
            )
            di["cubeA_pos"] = cubeA_pos
            di["cubeA_quat"] = cubeA_quat

            # position and rotation of the second cube
            cubeB_pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id])
            cubeB_quat = U.convert_quat(
                np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw"
            )
            di["cubeB_pos"] = cubeB_pos
            di["cubeB_quat"] = cubeB_quat

            # relative positions between gripper and cubes
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cubeA"] = gripper_site_pos - cubeA_pos
            di["gripper_to_cubeB"] = gripper_site_pos - cubeB_pos
            di["cubeA_to_cubeB"] = cubeA_pos - cubeB_pos

            # gripper orientation
            di["gripper_site_pos"] = gripper_site_pos
            di[
                "gripper_quat"
            ] = (
                self._right_hand_quat
            )  # This is unfortunately in robot base instead of world frame

            di["low-level"] = np.concatenate(
                [
                    cubeA_pos,
                    cubeA_quat,
                    cubeB_pos,
                    cubeB_quat,
                    di["gripper_to_cubeA"],
                    di["gripper_to_cubeB"],
                    di["cubeA_to_cubeB"],
                    di["gripper_site_pos"],
                    di["gripper_quat"],
                ]
            )
        # proprioception
        di["proprio"] = np.concatenate(
            [
                np.sin(di["joint_pos"]),
                np.cos(di["joint_pos"]),
                di["joint_vel"],
                di["gripper_pos"],
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
        r_reach, r_lift, r_stack = self.staged_rewards()
        return r_stack > 0

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
