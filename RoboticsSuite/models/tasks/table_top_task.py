import numpy as np
import xml.etree.ElementTree as ET
import collections

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError, RandomizationError
from RoboticsSuite.models.world import MujocoWorldBase
from RoboticsSuite.models.model_util import *
from RoboticsSuite.utils.utils import *


class TableTopTask(MujocoWorldBase):

    """
        Table top manipulation task can be specified 
        by three elements of the environment.
        @mujoco_arena, MJCF robot workspace (e.g., table top)
        @mujoco_robot, MJCF robot model
        @mujoco_objects, a list of MJCF objects of interest
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, initializer=None):
        super().__init__()
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        if initializer is None:
            initializer = UniformRandomSampler()
        mjcfs = [x for _, x in self.mujoco_objects.items()]
        self.initializer = initializer
        self.initializer.setup(mjcfs, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(joint(name=obj_name, type="free"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def place_objects(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        Args:
            position_sampler: generate random positions to put objects
        """
        pos_arr, quat_arr = self.initializer.sample()
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(pos_arr[i]))
            self.objects[i].set("quat", array_to_string(quat_arr[i]))
