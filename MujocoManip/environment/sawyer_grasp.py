import numpy as np
from MujocoManip.environment.sawyer_single_object_target import SawyerSingleObjectTargetEnv
from MujocoManip.model import BoxObject

class SawyerGraspEnv(SawyerSingleObjectTargetEnv):
    def __init__(self,
                mujoco_object=None,
                gripper='TwoFingerGripper',
                min_target_xy_distance=(0.1,0.1),
                max_target_height=0.1,
                min_target_height=0.5,
                **kwargs):
        """
            @min_target_xy_distance: Minimal x/y distance between object and target
            @max_target_height: Maximal height for the target
            @min_target_height: Minimal height for the target
                when they are not given, they are inferred from object size
        """
        if mujoco_object is None:
            mujoco_object = BoxObject(size=[0.02, 0.02, 0.02], rgba=[1,0,0,1])

        super().__init__(gripper=gripper, mujoco_object=mujoco_object, **kwargs)
        self.min_target_xy_distance = min_target_xy_distance
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

        success = False
        for i in range(1000):
            object_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
            object_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
            if abs(object_x - target_x) > self.min_target_xy_distance[0] and \
                abs(object_y - target_y) > self.min_target_xy_distance[1]:
                success = True
                self._object_pos=[object_x,object_y,0,0,0,0,0]
                break
        if not success:
            raise RandomizationError('Cannot place all objects on the desk')
