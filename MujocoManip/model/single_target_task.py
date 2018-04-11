import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.world import MujocoWorldBase
from MujocoManip.model.model_util import *

class SingleTargetTask(MujocoWorldBase):
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

        self.object_bottom_offset = mujoco_object.get_bottom_offset()
        object_center_offset = self.table_top_offset - self.object_bottom_offset

        # Load target
        task_target = mujoco_object.get_visual()
        set_alpha(task_target, 0.2)
        task_target.set('name', 'target')

        task_target.set('pos', array_to_string(object_center_offset))
        self.worldbody.append(task_target)

    def place_target(self, min_target_height, max_target_height):
        table_x_half = self.table_size[0] / 2
        table_y_half = self.table_size[1] / 2
        target_x = np.random.uniform(high=table_x_half, low= -1 * table_x_half)
        target_y = np.random.uniform(high=table_y_half, low= -1 * table_y_half)
        target_z = np.random.uniform(high=max_target_height, low=min_target_height)
        self._target_pos = np.array([target_x,target_y,target_z]) - self.object_bottom_offset
        self._target_pos = np.array([0, 0, 0.2])