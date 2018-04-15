import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion

class SawyerLiftEnv(SawyerEnv):

    def __init__(self, 
                 gripper='TwoFingerGripper',
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 camera_height=256,
                 camera_width=256,
                 camera_depth=True,
                 **kwargs):
        """
            @table_size, the full dimension of the table 
        """
        # initialize objects of interest
        self.mujoco_objects = [
            RandomBoxObject(size_min=[0.025, 0.025, 0.03], size_max=[0.05, 0.05, 0.05])
        ]

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # settings for camera observation
        self.use_camera_obs = use_camera_obs
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(gripper=gripper, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.model.place_objects()

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

    def reward(self, action):
        reward = 0
        #TODO(yukez): implementing a stacking reward
        return reward

    def _get_observation(self):
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

        # di['low-level'] = np.concatenate([self._object_pos,
        #                                   self._object_vel,
        #                                   self._target_pos,
        #                                   self._right_hand_pos,
        #                                   self._right_hand_vel
        #                                 ])
        return di

    def _check_terminated(self):
        #TODO(yukez): define termination conditions
        return False
