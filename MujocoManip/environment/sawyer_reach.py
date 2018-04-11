import numpy as np
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model.model_util import xml_path_completion
from MujocoManip.model import MujocoFileObject, SingleTargetTask, TableArena

class SawyerReachEnv(SawyerEnv):
    def __init__(self,
                mujoco_object=None,
                gripper='PushingGripper',
                reward_objective_factor=1,
                max_target_height=0.1,
                min_target_height=0.5,
                table_size=(0.8, 0.8, 0.8),
                table_friction=None,
                win_rel_tolerance=1e-2,
                **kwargs):
        """
        """
        
        if mujoco_object is None:
            mujoco_object = MujocoFileObject(xml_path_completion('object/object_ball.xml'))
        self.mujoco_object = mujoco_object
        self.table_size = table_size
        self.table_friction = table_friction
        

        super().__init__(gripper=gripper, **kwargs)
        self.reward_objective_factor=reward_objective_factor
        self.win_rel_tolerance=win_rel_tolerance
        self.max_target_height=max_target_height
        self.min_target_height=min_target_height
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))


    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
        
        self.target_bottom_offset = self.mujoco_object.get_bottom_offset()
        self.task = SingleTargetTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_object)

        self.model = self.task


    def _reset_internal(self):
        super()._reset_internal()
        print('Sawyer reach is temporarily deprecated due to change in model loading')
        # TODO: fix reacher env
        # 
        # table_x_half = self.table_size[0] / 2
        # table_y_half = self.table_size[1] / 2
        # target_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        # target_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        # target_z = np.random.uniform(high=self.max_target_height, low=self.min_target_height)
        # self._target_pos = np.array([target_x,target_y,target_z]) - self.target_bottom_offset
        # self._target_pos = np.array([0, 0, 0.2])

    def _pre_action(self, action):
        super()._pre_action(action)
        self.pre_action_hand_target_dist = np.linalg.norm(self._target_pos - self._right_hand_pos)

    def reward(self, action):
        reward = super().reward(action)
        self.post_action_hand_target_dist = np.linalg.norm(self._target_pos - self._right_hand_pos)
        reward += self.reward_objective_factor * (self.pre_action_hand_target_dist - self.post_action_hand_target_dist)
        # reward = np.exp(-1 * np.linalg.norm(self._target_pos - self._right_hand_pos))
        # print('reward', reward)
        return reward

    def _get_observation(self):
        di = super()._get_observation()
        di['low-level'] = np.concatenate([self._right_hand_pos,
                              self._target_pos])
        return di


    def _check_win(self):
        return np.allclose(self._target_pos, self._right_hand_pos, rtol=self.win_rel_tolerance)

    def _check_lose(self):
        return False

    @property
    def _target_pos(self):
        return self.sim.model.body_pos[self.sim.model.body_name2id('target')] - self._pos_offset

    @_target_pos.setter
    def _target_pos(self, pos):
        self.sim.model.body_pos[self.sim.model.body_name2id('target')] = pos + self._pos_offset

    def observation_space(self):
        low=np.ones(self.robot.dof() + 3) * -100.
        high=np.ones(self.robot.dof() + 3) * 100.
        return low, high

class SawyerReachEnvEEVel(SawyerReachEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # overrides
    def _pre_action(self, action):
        self.pre_action_object_target_dist = np.linalg.norm(self._target_pos[:2] - self._object_pos[:2])
        jacp = self.sim.data.get_body_jacp('right_hand').reshape([3, -1])
        jacp_joint = jacp[:, self._ref_joint_vel_indexes]
        vel = action
        sol, _, _, _ = np.linalg.lstsq(jacp_joint, vel)

        self.sim.data.ctrl[:] = np.concatenate([sol, self.gripper.rest_pos()])

        # correct for gravity
        self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

    def dof(self):
        return 3
