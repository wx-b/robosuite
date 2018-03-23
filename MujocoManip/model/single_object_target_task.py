import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.world import MujocoWorldBase
from MujocoManip.model.model_util import *
from MujocoManip.miscellaneous.utils import *

class SingleObjectTargetTask(MujocoWorldBase):
    def __init__(self, mujoco_arena, mujoco_robot, mujoco_object):
        super().__init__()
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_object(mujoco_object)

    def merge_arena(self, mujoco_arena):
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.full_size
        self.merge(mujoco_arena)

    def merge_robot(self, mujoco_robot):
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_object(self, mujoco_object):
        self.merge_asset(mujoco_object)
        # Load object
        task_object = mujoco_object.get_full()
        self.task_object = task_object
        task_object.set('name', 'object')
        self.object_bottom_offset = mujoco_object.get_bottom_offset()
        self.object_center_offset = self.table_top_offset - self.object_bottom_offset
        self.object_radius = mujoco_object.get_horizontal_radius()
        task_object.append(joint(name='object_free_joint', type='free'))
        self.worldbody.append(task_object)

        # Load target
        task_target = mujoco_object.get_visual()
        self.task_target = task_target
        set_alpha(task_target, 0.2)
        task_target.set('name', 'target')
        self.worldbody.append(task_target)

    def place_object(self, min_target_xy_distance=None):
        """
            Places object and target:
        Args:
            min_target_xy_distance: Minimal distance between object and target array of [x,y]
            None: [0,0]
            float/int r: [r,r]
            iterable arr: [arr[0], arr[1]] 
        """
        if min_target_xy_distance is None:
            min_target_xy_distance = [0,0]
        if isinstance(min_target_xy_distance, float) or isinstance(min_target_xy_distance, int):
            min_target_xy_distance = [min_target_xy_distance, min_target_xy_distance]

        table_x_half = self.table_size[0] / 2 - self.object_radius
        table_y_half = self.table_size[1] / 2 - self.object_radius
        target_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        target_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        self.task_target.set('pos', array_to_string(self.object_center_offset + np.array([target_x,target_y,0])))

        success = False
        for i in range(1000):
            object_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
            object_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
            if abs(object_x - target_x) > min_target_xy_distance[0] and \
                abs(object_y - target_y) > min_target_xy_distance[1]:
                success = True
                self.task_object.set('pos', array_to_string(self.object_center_offset + np.array([object_x,object_y,0])))
                break
        if not success:
            raise RandomizationError('Cannot place all objects on the desk')
    