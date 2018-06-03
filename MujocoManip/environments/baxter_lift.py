import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.baxter import BaxterEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion


class BaxterLiftEnv(BaxterEnv):

    def __init__(self, 
                 gripper_type='TwoFingerGripper',
                 use_eef_ctrl=False,
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 reward_shaping=True,
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
        """
        # initialize objects of interest
        # cube = RandomBoxObject(size_min=[0.02, 0.02, 0.02],
        #                        size_max=[0.025, 0.025, 0.025])
        self.pot = GeneratedPotObject()
        #pot = cube
        self.mujoco_objects = OrderedDict([('pot', self.pot)])

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # reward configuration
        self.reward_shaping = reward_shaping

        self.object_initializer = UniformRandomSampler(x_range=(-0.2, 0),
                                                       y_range=(-0.2, 0.2),
                                                       z_rotation=(-0.2 * np.pi, 0.2 * np.pi))

        super().__init__(gripper_left='LeftTwoFingerGripper',
                         gripper_right='TwoFingerGripper',
                         use_eef_ctrl=use_eef_ctrl,
                         use_camera_obs=use_camera_obs,
                         camera_name=camera_name,
                         gripper_visualization=gripper_visualization,
                         **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(full_size=self.table_size,
                                       friction=self.table_friction)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.45 + self.table_size[0] / 2,0,0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena,
                                  self.mujoco_robot,
                                  self.mujoco_objects,
                                  self.object_initializer)
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id('pot')
        self.handle_1_site_id = self.sim.model.site_name2id('pot_handle_1')
        self.handle_2_site_id = self.sim.model.site_name2id('pot_handle_2')
        self.table_top_id = self.sim.model.site_name2id('table_top')
        self.pot_center_id = self.sim.model.site_name2id('pot_center')

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

    def reward(self, action):
        reward = 0
        cube_height = self.sim.data.site_xpos[self.pot_center_id][2] - 0.07
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.15:
            reward = 1.5

        # use a shaping reward
        if self.reward_shaping:
            # Height reward
            reward += 10 * min(cube_height - table_height, 0.2)

            l_gripper_to_handle = self._l_gripper_to_handle
            r_gripper_to_handle = self._r_gripper_to_handle

            # gh stands for gripper-handle

            # When grippers are far away, tell them to be closer
            l_gh_dist = np.linalg.norm(l_gripper_to_handle)
            r_gh_dist = np.linalg.norm(r_gripper_to_handle)
            # if l_gh_dist_xy > 0.05:
            reward -= np.tanh(l_gh_dist)
            # if r_gh_dist_xy > 0.05:
            reward -= np.tanh(r_gh_dist)
 
            # l_gh_dist_xy = np.linalg.norm(l_gripper_to_handle[:2])
            # r_gh_dist_xy = np.linalg.norm(r_gripper_to_handle[:2])

            # l_gh_z_diff = l_gripper_to_handle[2]
            # r_gh_z_diff = r_gripper_to_handle[2]

            # handle_reward = 0

            # When gripper is close and above, tell them to move away and down
            # When gripper is close and below, tell them to move close and up
            # coef_z = 2
            # coef_xy = 1
            # if l_gh_z_diff > 0: # gripper below handle
            #     handle_reward -= coef_z * l_gh_z_diff
            #     handle_reward -= coef_xy * l_gh_dist_xy
            # else:
            #     handle_reward += coef_z * l_gh_z_diff
            #     handle_reward += coef_xy * min(0, -0.05 + l_gh_dist_xy) # force the gripper to move away a bit
            # if r_gh_z_diff > 0:
            #     handle_reward -= coef_z * r_gh_z_diff
            #     handle_reward -= coef_xy * r_gh_dist_xy
            # else:
            #     handle_reward += coef_z * l_gh_z_diff
            #     handle_reward += coef_xy * min(0, -0.05 + l_gh_dist_xy) # force the gripper to move away a bit

            # reward += handle_reward

            # contact_reward = 0.25
            # for contact in self.find_contacts(self.gripper_left.contact_geoms(),
            #                                   self.pot.handle_1_geoms()):
            #     reward += contact_reward
            # for contact in self.find_contacts(self.gripper_right.contact_geoms(),
            #                                   self.pot.handle_2_geoms()):
            #     reward += contact_reward

            # Allow flipping no more than 30 degrees
            handle_1_pos = self._handle_1_xpos
            handle_2_pos = self._handle_2_xpos
            z_diff = abs(handle_1_pos[2] - handle_2_pos[2])
            angle = np.pi / 4 # 45 degrees
            handle_distance = self.pot.handle_distance
            z_diff_tolerance = np.sin(angle) * handle_distance
            reward -= 2 * max(z_diff - z_diff_tolerance, 0)

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
    def _l_gripper_to_handle(self):
        return self._handle_1_xpos - self._l_eef_xpos

    @property
    def _r_gripper_to_handle(self):
        return self._handle_2_xpos - self._r_eef_xpos

    def _get_observation(self):
        di = super()._get_observation()
        # camera observations
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
            # position and rotation of object
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            cube_quat = self.sim.data.body_xquat[self.cube_body_id]
            di['cube_pos'] = cube_pos
            di['cube_quat'] = cube_quat

            di['l_eef_xpos'] = self._l_eef_xpos
            di['r_eef_xpos'] = self._r_eef_xpos
            di['handle_1_xpos'] = self._handle_1_xpos
            di['handle_2_xpos'] = self._handle_2_xpos
            di['l_gripper_to_handle'] = self._l_gripper_to_handle
            di['r_gripper_to_handle'] = self._r_gripper_to_handle

            di['low-level'] = np.concatenate([
                di['cube_pos'],
                di['cube_quat'],
                di['l_eef_xpos'],
                di['r_eef_xpos'],
                di['handle_1_xpos'],
                di['handle_2_xpos'],
                di['l_gripper_to_handle'],
                di['r_gripper_to_handle'],
            ])

        # proprioception
        di['proprio'] = np.concatenate([
            np.sin(di['joint_pos']),
            np.cos(di['joint_pos']),
            di['joint_vel'],
        ])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in contact_geoms or \
               self.sim.model.geom_id2name(contact.geom2) in contact_geoms:
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
        return (cube_height > table_height + 0.10)
