import xml.etree.ElementTree as ET
import numpy as np

from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError
from RoboticsSuite.models.model_util import xml_path_completion


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
