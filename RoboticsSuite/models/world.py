import numpy as np
import xml.etree.ElementTree as ET
from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.miscellaneous import XMLError
from RoboticsSuite.models.model_util import xml_path_completion

### Base class to inherit all mujoco worlds from
class MujocoWorldBase(MujocoXML):
    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
