import numpy as np
import xml.etree.ElementTree as ET

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils.mjcf_utils import *
from RoboticsSuite.utils import XMLError


class Arena(MujocoXML):
    """Base arena class."""

    def set_origin(self, offset):
        """Apply a constant offset to all objects."""
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def add_pos_indicator(self):
        """Add a new position indicator."""
        body = new_body(name="pos_indicator")
        body.append(
            new_geom(
                "sphere",
                [0.03],
                rgba=[1, 0, 0, 0.5],
                group=1,
                contype="0",
                conaffinity="0",
            )
        )
        body.append(joint(type="free", name="pos_indicator"))
        self.worldbody.append(body)
