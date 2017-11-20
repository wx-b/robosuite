import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.world import MujocoWorldBase
from MujocoManip.model.model_util import *

class PusherTask(MujocoWorldBase):
    def __init__(self, mujoco_arena, mujoco_robot, mujoco_object):
        super().__init__()
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_object(mujoco_object)

    def merge_arena(self, mujoco_arena):
        self.table_top_offset = mujoco_arena.table_top_abs
        self.merge(mujoco_arena)

    def merge_robot(self, mujoco_robot):
        self.merge(mujoco_robot)

    def merge_object(self, mujoco_object):
        self.merge_asset(mujoco_object)
        # Load object
        pusher_object = mujoco_object.get_full()
        pusher_object.set('name', 'pusher_object')
        object_bottom_offset = mujoco_object.get_bottom_offset()
        object_center_offset = self.table_top_offset - object_bottom_offset
        pusher_object.set('pos', array_to_string(object_center_offset))
        pusher_object.append(joint(name='pusher_object_free_joint', type='free'))
        self.worldbody.append(pusher_object)

        # Load target
        pusher_target = mujoco_object.get_visual()
        set_alpha(pusher_target, 0.2)
        pusher_target.set('name', 'pusher_target')
        pusher_target.set('pos', array_to_string(object_center_offset))
        self.worldbody.append(pusher_target)
