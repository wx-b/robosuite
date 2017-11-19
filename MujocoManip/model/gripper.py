import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError


class MujocoGripper(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname):
        super().__init__(fname)
