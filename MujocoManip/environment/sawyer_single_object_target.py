import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model import MujocoFileObject, SingleTargetTask, TableArena, SingleObjectTargetTask
from MujocoManip.model.model_util import xml_path_completion

class SawyerSingleObjectTargetEnv(SawyerEnv):
    def __init__(self, 
                mujoco_object=None,
                table_size=(0.8, 0.8, 0.8),
                table_friction=None,
                reward_lose=-1,
                reward_win=1,
                reward_action_norm_factor=0,
                reward_objective_factor=5,
                win_rel_tolerance=1e-2,
                **kwargs):
        """
            @mujoco_object(None), the object to be pushed, need that is is an MujocoObject instace
            If None, load 'object/object_ball.xml'
            @table_size, the FULL size of the table 
            @friction: friction coefficient of table, None for mujoco default
            @reward_win: reward given to the agent when it completes the task
            @reward_lose: reward given to the agent when it fails the task
            @reward_action_norm_factor: reward scaling factor that penalizes large actions
            @reward_objective_factor: reward scaling factor for being close to completing the objective
            @win_rel_tolerance: relative tolerance between object and target location 
                used when deciding if the agent has completed the task
        """
        # Handle parameters
        self.mujoco_object = mujoco_object
        if self.mujoco_object is None:
            self.mujoco_object = MujocoFileObject(xml_path_completion('object/object_ball.xml'))
        self.table_size = table_size
        self.table_friction = table_friction

        self.reward_lose = reward_lose
        self.reward_win = reward_win
        self.win_rel_tolerance = win_rel_tolerance
        self.reward_action_norm_factor = reward_action_norm_factor
        self.reward_objective_factor = reward_objective_factor

        super().__init__(**kwargs)
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

    def _get_reference(self):
        super()._get_reference()
        self._ref_object_pos_indexes = self.model.get_joint_qpos_addr('object_free_joint')
        self._ref_object_vel_indexes = self.model.get_joint_qvel_addr('object_free_joint')

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
        
        self.target_bottom_offset = self.mujoco_object.get_bottom_offset()
        self.task = SingleObjectTargetTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_object)

        self.model = self.task

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset position of target and then reset position of object

    def _pre_action(self, action):
        super()._pre_action(action)
        self.pre_action_object_target_dist = np.linalg.norm(self._target_pos - self._object_pos)

    def reward(self, action):
        reward = 0
        self.post_action_object_target_dist = np.linalg.norm(self._target_pos - self._object_pos)
        
        if self._check_win():
            reward += self.reward_win
        elif self._check_lose():
            reward += self.reward_lose
        reward += self.reward_objective_factor * (self.pre_action_object_target_dist - self.post_action_object_target_dist)
        reward += self.reward_action_norm_factor * np.linalg.norm(action, 2)
        return reward

    def _get_observation(self):
        di = super()._get_observation()
        di['low-level'] = np.concatenate([self._object_pos,
                                          self._object_vel,
                                          self._target_pos,
                                          self._right_hand_pos,
                                          self._right_hand_vel
                                        ])
        return di

    def _check_lose(self):
        x_out = np.abs(self._object_pos[0]) > self.table_size[0] / 2
        y_out = np.abs(self._object_pos[1]) > self.table_size[1] / 2
        return x_out or y_out or self.t > self.horizon

    def _check_win(self):
        return np.allclose(self._target_pos, self._object_pos, rtol=self.win_rel_tolerance)

    @property
    def observation_space(self):
        low=np.ones(37) * -100.
        high=np.ones(37) * 100.
        return low, high


    ####
    # Properties for objects
    ####

    @property
    def _object_pos(self):
        return self.sim.data.get_body_xpos('object') - self._pos_offset

    @_object_pos.setter
    def _object_pos(self, pos):
        low, high = self._ref_object_pos_indexes
        self.sim.data.qpos[low:high] = pos + self._pos_offset

    @property
    def _target_pos(self):
        return self.sim.model.body_pos[self.sim.model.body_name2id('target')] - self._pos_offset

    @_target_pos.setter
    def _target_pos(self, pos):
        self.sim.model.body_pos[self.sim.model.body_name2id('target')] = pos + self._pos_offset

    @property
    def _object_vel(self):
        return self.sim.data.get_body_xvelp('object')

