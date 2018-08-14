import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError
from RoboticsSuite.models.world import MujocoWorldBase
from RoboticsSuite.models.model_util import *
from RoboticsSuite.utils.utils import *


class BinsTask(MujocoWorldBase):

    """
        APC manipulation task can be specified 
        by three elements of the environment.
        @mujoco_arena, MJCF robot workspace (e.g., shelves)
        @mujoco_robot, MJCF robot model
        @mujoco_objects, a list of MJCF objects of interest
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, visual_objects):
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.object_metadata = []
        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(OrderedDict(visual_objects))
        self.visual_objects = visual_objects

    def merge_robot(self, mujoco_robot):
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        self.arena = mujoco_arena
        self.shelf_offset = mujoco_arena.bin_abs
        self.shelf_size = mujoco_arena.full_size
        self.bin2_body = mujoco_arena.bin2_body
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        self.n_objects = len(mujoco_objects)
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(joint(name=obj_name, type="free", damping="0.0005"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def merge_visual(self, mujoco_objects):
        self.visual_obj_mjcf = []
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_visual(name=obj_name, site=False)
            self.visual_obj_mjcf.append(obj)
            self.worldbody.append(obj)

    def sample_quat(self):
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        else:
            return [1, 0, 0, 0]

    def place_objects(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        """
        # Objects
        # print(self.shelf_offset)
        placed_objects = []
        index = 0
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for _ in range(5000):  # 5000 retries
                shelf_x_half = self.shelf_size[0] / 2 - horizontal_radius - 0.05
                shelf_y_half = self.shelf_size[1] / 2 - horizontal_radius - 0.05
                object_x = np.random.uniform(high=shelf_x_half, low=-shelf_x_half)
                object_y = np.random.uniform(high=shelf_y_half, low=-shelf_y_half)
                # objects cannot overlap
                pos = (
                    self.shelf_offset
                    - bottom_offset
                    + np.array([object_x, object_y, 0])
                )
                location_valid = True
                for pos2, r in placed_objects:
                    if (
                        np.linalg.norm(pos[:2] - pos2[:2], np.inf)
                        <= r + horizontal_radius
                    ):
                        location_valid = False
                        break
                if location_valid:  # bad luck, reroll
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set("pos", array_to_string(pos))

                    # random z-rotation
                    quat = self.sample_quat()
                    self.objects[index].set("quat", array_to_string(quat))

                    success = True
                    break
                    # location is valid, put the object down
                    # quarternions, later we can add random rotation
            if not success:
                raise RandomizationError("Cannot place all objects on the shelves")
            # print(placed_objects)
            index += 1

    def place_visual(self):
        """
        Place objects randomly until no more collisions or max iterations hit.
        """
        # Objects
        # print(self.shelf_offset)
        placed_objects = []
        index = 0
        bin_pos = string_to_array(self.bin2_body.get("pos"))
        bin_size = self.shelf_size

        for _, obj_mjcf in self.visual_objects:

            bin_x_low = bin_pos[0]
            bin_y_low = bin_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= bin_size[0] / 2
            if index < 2:
                bin_y_low -= bin_size[1] / 2

            bin_x_high = bin_x_low + bin_size[0] / 2
            bin_y_high = bin_y_low + bin_size[1] / 2
            bottom_offset = obj_mjcf.get_bottom_offset()

            pos = (
                np.array(
                    [bin_x_low + bin_x_high, bin_y_low + bin_y_high, 2 * bin_pos[2]]
                )
                / 2
                - bottom_offset
            )
            self.visual_obj_mjcf[index].set("pos", array_to_string(pos))
            index += 1
