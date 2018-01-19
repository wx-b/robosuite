import numpy as np
from MujocoManip.environment.sawyer_single_object_target import SawyerSingleObjectTargetEnv
from MujocoManip.model import MujocoXMLObject
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
        """
        if mujoco_object is None:
            mujoco_object = MujocoXMLObject(xml_path_completion('object/object_ball.xml'))

        super().__init__(gripper=gripper, mujoco_object=mujoco_object, **kwargs)
        self.min_target_xy_distance = min_target_xy_distance

        self.reward_touch_object_factor=reward_touch_object_factor
        self.reward_align_direction_factor=reward_align_direction_factor



    def _reset_internal(self):
        super()._reset_internal()

        table_x_half = self.table_size[0] / 2
        table_y_half = self.table_size[1] / 2
        target_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        target_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        self._target_pos = np.array([target_x,target_y,0]) - self.target_bottom_offset

        success = False
        for i in range(1000):
            object_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
            object_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
            if abs(object_x - target_x) > self.min_target_xy_distance[0] and \
                abs(object_y - target_y) > self.min_target_xy_distance[1]:
                success = True
                self._object_pos=[object_x,object_y,0,] - self.mujoco_object.get_bottom_offset()
                break
        if not success:
            raise RandomizationError('Cannot place all objects on the desk')
    
    def _reward(self, action):
        reward = super()._reward(action)
        # Credit to jyg
        # Secret sauce to get pushing working
        reward += self.reward_touch_object_factor * np.exp(-20. * np.linalg.norm(self._right_hand_pos - self._object_pos))
        reward += self.reward_align_direction_factor * np.dot(self._right_hand_pos - self._object_pos, 
                 self._object_pos - self._target_pos) / (np.linalg.norm(
                     self._object_pos - self._target_pos) * \
                         np.linalg.norm(self._right_hand_pos - self._object_pos))
        return reward


