import numpy as np
from MujocoManip.environment.sawyer_single_object_target import SawyerSingleObjectTargetEnv
from MujocoManip.model import MujocoXMLObject
from MujocoManip.model.model_util import xml_path_completion

class SawyerReachEnv(SawyerSingleObjectTargetEnv):
    def __init__(self,
                mujoco_object=None,
                gripper='PushingGripper',
                reward_objective_factor=-1,
                max_target_height=0.1,
                min_target_height=0.5,
                **kwargs):
        """
            @min_target_xy_distance: Minimal x/y distance between object and target
            @reward_touch_object_factor: coefficient for custom find grained reward (touching object)
            @reward_touch_object_factor: coefficient for custom find grained reward (aligning direction)
        """
        if mujoco_object is None:
            mujoco_object = MujocoXMLObject(xml_path_completion('object/object_ball.xml'))

        super().__init__(gripper=gripper, mujoco_object=mujoco_object, reward_objective_factor=0, **kwargs)
        self.reward_objective_factor=reward_objective_factor
        self.max_target_height = max_target_height
        self.min_target_height = min_target_height


    def _reset_internal(self):
        super()._reset_internal()

        table_x_half = self.table_size[0] / 2
        table_y_half = self.table_size[1] / 2
        target_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        target_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        target_z = np.random.uniform(high=self.max_target_height, low=self.min_target_height)
        self._target_pos = np.array([target_x,target_y,target_z]) - self.target_bottom_offset

        object_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        object_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        self._object_pos=np.concatenate([[object_x,object_y,0,] - self.mujoco_object.get_bottom_offset(), [0,0,0,0]])
    
    def _reward(self, action):
        reward = super()._reward(action)
        reward += 0.01 * self.reward_objective_factor * np.linalg.norm(self._right_hand_pos - self._target_pos)
        # print('reward', reward)
        return reward

    def _check_win(self):
        return np.allclose(self._target_pos, self._right_hand_pos, rtol=self.win_rel_tolerance)
