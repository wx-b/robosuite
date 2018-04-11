import numpy as np
from MujocoManip.environment.sawyer_single_object_target import SawyerSingleObjectTargetEnv
from MujocoManip.model import MujocoFileObject
from MujocoManip.model.model_util import xml_path_completion


class SawyerPushEnv(SawyerSingleObjectTargetEnv):
    def __init__(self,
                mujoco_object=None,
                gripper='PushingGripper',
                min_target_xy_distance=(0.1,0.1),
                reward_touch_object_factor=0.01,
                reward_align_direction_factor=0.01,
                **kwargs):
        """
            @min_target_xy_distance: Minimal x/y distance between object and target
            @reward_touch_object_factor: coefficient for custom find grained reward (touching object)
            @reward_touch_object_factor: coefficient for custom find grained reward (aligning direction)
            @use_torque_ctrl: if True, actions are joint torques, not joint velocities
        """
        if mujoco_object is None:
            mujoco_object = MujocoFileObject(xml_path_completion('object/object_ball.xml'))

        super().__init__(gripper=gripper, mujoco_object=mujoco_object, **kwargs)
        self.min_target_xy_distance = min_target_xy_distance

        self.reward_touch_object_factor=reward_touch_object_factor
        self.reward_align_direction_factor=reward_align_direction_factor


    def _reset_internal(self):
        super()._reset_internal()
        self.model.place_object(min_target_xy_distance=self.min_target_xy_distance)

    
    def reward(self, action):
        reward = super().reward(action)
        # Credit to jyg
        # Secret sauce to get pushing working
        reward += self.reward_touch_object_factor * np.exp(-20. * np.linalg.norm(self._right_hand_pos - self._object_pos))
        reward += self.reward_align_direction_factor * np.dot(self._right_hand_pos - self._object_pos, 
                 self._object_pos - self._target_pos) / (np.linalg.norm(
                     self._object_pos - self._target_pos) * \
                         np.linalg.norm(self._right_hand_pos - self._object_pos))
        return reward
