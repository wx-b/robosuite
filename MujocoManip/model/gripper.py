import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import *

class MujocoGripper(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname):
        super().__init__(fname)

class TwoFingerGripper(MujocoGripper):
	def __init__(self):
		super().__init__(xml_path_completion('gripper/two_finger_gripper.xml'))