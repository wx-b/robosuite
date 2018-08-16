import numpy as np
from collections import OrderedDict
from RoboticsSuite.utils import RandomizationError
from RoboticsSuite.environments.baxter import BaxterEnv
from RoboticsSuite.environments.demo_sampler import DemoSampler
from RoboticsSuite.models import *
from RoboticsSuite.models.model_util import xml_path_completion
import RoboticsSuite.utils as U
import pickle
import random


class BaxterLift(BaxterEnv):
    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        use_eef_ctrl=False,
        table_size=(0.8, 0.8, 0.8),
        table_friction=None,
        use_camera_obs=True,
        use_object_obs=True,
        camera_name="frontview",
        reward_shaping=True,
        gripper_visualization=False,
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
        """
        # initialize objects of interest
        self.pot = GeneratedPotObject()
        self.mujoco_objects = OrderedDict([("pot", self.pot)])

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        self.demo_config = demo_config
        if self.demo_config is not None:
            self.demo_sampler = DemoSampler(
                "demonstrations/baxter-lift.pkl", self.demo_config
            )

        self.eps_reward = 0
        # reward configuration
        self.reward_shaping = reward_shaping

        self.object_initializer = UniformRandomSampler(
            x_range=(-0.15, -0.04),
            y_range=(-0.015, 0.015),
            z_rotation=(-0.15 * np.pi, 0.15 * np.pi),
            ensure_object_boundary_in_range=False,
        )

        super().__init__(
            gripper_left="LeftTwoFingerGripper",
            gripper_right="TwoFingerGripper",
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
        self.mujoco_arena = TableArena(
            full_size=self.table_size, friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.45 + self.table_size[0] / 2, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.object_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id("pot")
        self.handle_1_site_id = self.sim.model.site_name2id("pot_handle_1")
        self.handle_2_site_id = self.sim.model.site_name2id("pot_handle_2")
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id("pot_center")

    def _reset_from_random(self):
        # inherited class should reset positions of objects
        self.model.place_objects()

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

    def reward(self, action):
        """
        1. the agent only gets the lifting reward when flipping no more than 30 degrees.
        2. the lifting reward is smoothed and ranged from 0 to 2, capped at 2.0. 
           the initial lifting reward is 0 when the pot is on the table;
           the agent gets the maximum 2.0 reward when the pot’s height is above a threshold.
        3. the reaching reward is 0.5 when the left gripper touches the left handle,
           or when the right gripper touches the right handle before the gripper geom 
           touches the handle geom, and once it touches we use 0.5
        """
        reward = 0

        # (TODO) remove hardcoded pot dimension
        cube_height = self.sim.data.site_xpos[self.pot_center_id][2] - 0.07
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # check if the pot is tilted more than 30 degrees
        mat = U.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.15:
            reward = 1.0 * direction_coef

        # use a shaping reward
        if self.reward_shaping:
            reward = 0

            # lifting reward
            elevation = cube_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.2)
            reward += 10. * direction_coef * r_lift

            l_gripper_to_handle = self._l_gripper_to_handle
            r_gripper_to_handle = self._r_gripper_to_handle

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer
            l_contacts = list(
                self.find_contacts(
                    self.gripper_left.contact_geoms(), self.pot.handle_1_geoms()
                )
            )
            r_contacts = list(
                self.find_contacts(
                    self.gripper_right.contact_geoms(), self.pot.handle_2_geoms()
                )
            )
            l_gh_dist = np.linalg.norm(l_gripper_to_handle)
            r_gh_dist = np.linalg.norm(r_gripper_to_handle)

            if len(l_contacts) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(l_gh_dist))

            if len(r_contacts) > 0:
                reward += 0.5
            else:
                reward += 0.5 * (1 - np.tanh(r_gh_dist))

        self.eps_reward = max(reward, self.eps_reward)
        return reward

    @property
    def _l_eef_xpos(self):
        return self.sim.data.site_xpos[self.left_eef_site_id]

    @property
    def _r_eef_xpos(self):
        return self.sim.data.site_xpos[self.right_eef_site_id]

    @property
    def _handle_1_xpos(self):
        return self.sim.data.site_xpos[self.handle_1_site_id]

    @property
    def _handle_2_xpos(self):
        return self.sim.data.site_xpos[self.handle_2_site_id]

    @property
    def _pot_quat(self):
        return U.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

    @property
    def _world_quat(self):
        return U.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _l_gripper_to_handle(self):
        return self._handle_1_xpos - self._l_eef_xpos

    @property
    def _r_gripper_to_handle(self):
        return self._handle_2_xpos - self._r_eef_xpos

    def _get_observation(self):
        di = super()._get_observation()
        # camera observations
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
            # position and rotation of object
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            cube_quat = U.convert_quat(
                self.sim.data.body_xquat[self.cube_body_id], to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            di["l_eef_xpos"] = self._l_eef_xpos
            di["r_eef_xpos"] = self._r_eef_xpos
            di["handle_1_xpos"] = self._handle_1_xpos
            di["handle_2_xpos"] = self._handle_2_xpos
            di["l_gripper_to_handle"] = self._l_gripper_to_handle
            di["r_gripper_to_handle"] = self._r_gripper_to_handle

            di["low-level"] = np.concatenate(
                [
                    di["cube_pos"],
                    di["cube_quat"],
                    di["l_eef_xpos"],
                    di["r_eef_xpos"],
                    di["handle_1_xpos"],
                    di["handle_2_xpos"],
                    di["l_gripper_to_handle"],
                    di["r_gripper_to_handle"],
                ]
            )

        # proprioception
        di["proprio"] = np.concatenate(
            [np.sin(di["joint_pos"]), np.cos(di["joint_pos"]), di["joint_vel"]]
        )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = (
            self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        )
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in contact_geoms
                or self.sim.model.geom_id2name(contact.geom2) in contact_geoms
            ):
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_size[2]
        return cube_height > table_height + 0.10
