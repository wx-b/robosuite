import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXMLFile
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import xml_path_completion

### Base class to inherit all mujoco worlds from
class MujocoWorldBase(MujocoXMLFile):
    def __init__(self):
        super().__init__(xml_path_completion('base.xml'))

