import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError
from RoboticsSuite.models.world import MujocoWorldBase
from RoboticsSuite.models.model_util import *
from RoboticsSuite.utils.utils import *
from RoboticsSuite.models.tasks.placement_sampler import ObjectPositionSampler


class PegsTask(MujocoWorldBase):

    """
        APC manipulation task can be specified 
        by three elements of the environment.
        @mujoco_arena, MJCF robot workspace (e.g., shelves)
        @mujoco_robot, MJCF robot model
        @mujoco_objects, a list of MJCF objects of interest
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
        self.initializer.setup(self.mujoco_objects, self.shelf_offset, self.shelf_size)

    def merge_robot(self, mujoco_robot):
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        self.arena = mujoco_arena
        self.shelf_offset = mujoco_arena.table_top_abs
        self.shelf_size = mujoco_arena.full_size
        self.bin1_body = mujoco_arena.bin1_body
        self.peg1_body = mujoco_arena.peg1_body
        self.peg2_body = mujoco_arena.peg2_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        self.n_objects = len(mujoco_objects)
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

    def sample_quat(self):
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        else:
            return [1, 0, 0, 0]

    def place_objects(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        Args:
            position_sampler: generate random positions to put objects
        """
        pos_arr, quat_arr = self.initializer.sample()
        i = 0
        for obj_name in self.objects:
            self.objects[obj_name].set("pos", array_to_string(pos_arr[i]))
            self.objects[obj_name].set("quat", array_to_string(quat_arr[i]))
            i += 1
