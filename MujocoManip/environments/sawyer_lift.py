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
                 **kwargs):
        """
            @table_size, the FULL size of the table 
        """
        # Handle parameters
        self.mujoco_objects = [
            RandomBoxObject(size_min=[0.025, 0.025, 0.03], size_max=[0.05, 0.05, 0.05])
        ]

        self.table_size = table_size
        self.table_friction = table_friction

        super().__init__(gripper=gripper, **kwargs)
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
        
        self.target_bottom_offset = self.mujoco_objects[0].get_bottom_offset()
        self.task = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)

        self.model = self.task
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset position of target and then reset position of object
        self.model.place_objects()

    def _pre_action(self, action):
        super()._pre_action(action)
        # self.pre_action_object_target_dist = np.linalg.norm(self._target_pos - self._object_pos)

    def reward(self, action):
        reward = 0
        #TODO(yukez): implementing a stacking reward
        return reward

    def _get_observation(self):
        di = super()._get_observation()
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

    # Properties for objects

    @property
    def observation_space(self):
        low=np.ones(37) * -100.
        high=np.ones(37) * 100.
        return low, high

    # @property
    # def _object_pos(self):
    #     return self.sim.data.get_body_xpos('object') - self._pos_offset
    #
    # @_object_pos.setter
    # def _object_pos(self, pos):
    #     low, high = self._ref_object_pos_indexes
    #     self.sim.data.qpos[low:high] = pos + self._pos_offset
    #
    # @property
    # def _target_pos(self):
    #     return self.sim.model.body_pos[self.sim.model.body_name2id('target')] - self._pos_offset
    #
    # @_target_pos.setter
    # def _target_pos(self, pos):
    #     self.sim.model.body_pos[self.sim.model.body_name2id('target')] = pos + self._pos_offset
    #
    # @property
    # def _object_vel(self):
    #     return self.sim.data.get_body_xvelp('object')
    #
