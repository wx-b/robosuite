import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion


class SawyerLiftEnv(SawyerEnv):

    def __init__(self, 
                 gripper_type='TwoFingerGripper',
                 use_eef_ctrl=False,
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 reward_shaping=False,
                 gripper_visualization=False,
                 placement_initializer=None,
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

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03], y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=True)

        super().__init__(gripper_type=gripper_type,
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
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])

        # initialize objects of interest
        cube = RandomBoxObject(size_min=[0.020, 0.020, 0.020], #[0.015, 0.015, 0.015],
                               size_max=[0.022, 0.022, 0.022], #[0.018, 0.018, 0.018])
                               rgba=[1, 0, 0, 1])
        self.mujoco_objects = OrderedDict([('cube', cube)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena, 
                                self.mujoco_robot, 
                                self.mujoco_objects,
                                initializer=self.placement_initializer)
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id('cube')
        self.l_finger_geom_id = self.sim.model.geom_name2id('l_fingertip_g0')
        self.r_finger_geom_id = self.sim.model.geom_name2id('r_fingertip_g0')
        self.cube_geom_id = self.sim.model.geom_name2id('cube')

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()
        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208,  0.4155, 1.8409,
                             -0.4955, 0.6482,  1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action):
        reward = 0
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_size[2]

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.04:
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 == self.l_finger_geom_id and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 == self.l_finger_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.r_finger_geom_id and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 == self.r_finger_geom_id:
                    touch_right_finger = True
            if touch_right_finger and touch_right_finger:
                reward += 0.25

        return reward

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
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_quat = np.array(self.sim.data.body_xquat[self.cube_body_id])
            di['cube_pos'] = cube_pos
            di['cube_quat'] = cube_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di['gripper_to_cube'] = gripper_site_pos - cube_pos

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
            if self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms() or \
               self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms():
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

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id('cube')
            dist = np.sum(np.square(self.sim.data.site_xpos[cube_site_id] - self.sim.data.get_site_xpos('grip_site')))

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba


