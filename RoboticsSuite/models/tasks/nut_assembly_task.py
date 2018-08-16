import xml.etree.ElementTree as ET
from collections import OrderedDict
import numpy as np

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError
from RoboticsSuite.models.world import MujocoWorldBase
from RoboticsSuite.models.model_util import *
from RoboticsSuite.models.tasks.placement_sampler import UniformRandomPegsSampler
from RoboticsSuite.utils import *


class NutAssemblyTask(MujocoWorldBase):
    """Create MJCF model of a nut assembly task.

    A nut assembly task consists of one robot pick up nuts from a table and
    and assembly them into pegs positioned on the tabletop. This class combines
    the robot, the arena with pegs, and the nut objetcts into a single MJCF model.

    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, initializer=None):
        super().__init__()

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)

        if initializer is None:
            initializer = UniformRandomPegsSampler()
        self.initializer = initializer
        self.initializer.setup(self.mujoco_objects, self.bin_offset, self.bin_size)

    def merge_robot(self, mujoco_robot):
        """Add robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Add arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.bin_offset = mujoco_arena.table_top_abs
        self.bin_size = mujoco_arena.full_size
        self.bin1_body = mujoco_arena.bin1_body
        self.peg1_body = mujoco_arena.peg1_body
        self.peg2_body = mujoco_arena.peg2_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Add physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = {}  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(joint(name=obj_name, type="free", damping="0.0005"))
            self.objects[obj_name] = obj
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def place_objects(self):
        """Place objects randomly until no collisions or max iterations hit."""
        pos_arr, quat_arr = self.initializer.sample()
        index = 0
        for obj_name in self.objects:
            self.objects[obj_name].set("pos", array_to_string(pos_arr[index]))
            self.objects[obj_name].set("quat", array_to_string(quat_arr[index]))
            index += 1
