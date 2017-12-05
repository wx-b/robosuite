import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import *



class MujocoGripper(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname):
        super().__init__(fname)

    def format_action_1d(self, action):
        """
            Given 1-d action in closed (-1, 1) to open, 
            returns the control for underlying actuators
            returns 1-d np array 
        """
        raise NotImplementedError

    def rest_pos(self):
        """
            Returns rest(open) qpos of the gripper
        """
        raise NotImplementedError

    def dof(self):
        """
            Returns the number of DOF of the gripper
        """
        raise NotImplementedError


class TwoFingerGripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/two_finger_gripper.xml'))

    def format_action_1d(self, action):
        return np.array([-1 * action, 1 * action])

    def rest_pos(self):
        return np.array([0.020833, -0.020833] )

    def dof(self):
        return 2

class PR2Gripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/pr2_gripper.xml'))

    def format_action_1d(self, action):
        return np.ones(4) * action

    def rest_pos(self):
        return np.zeros(4)

    def dof(self):
        return 4

class RobotiqGripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/robotiq_gripper.xml'))

    def format_action_1d(self, action):
        return -1 * np.ones(6) * action

    def rest_pos(self):
        return [ 3.3161, 0., 0., 0., 0., 0.]

    def dof(self):
        return 6

class PushingGripper(TwoFingerGripper):
    """Same as Two FingerGripper, but always closed"""
    def format_action_1d(self, action):
        return np.array([1, -1])

class RobotiqThreeFingerGripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/robotiq_gripper_s.xml'))

    def format_action_1d(self, action):
        return np.array([0] + [action] * 3 + [0] + [action] * 6)

    def rest_pos(self):
        return np.zeros(11)

    def dof(self):
        return 11

def gripper_factory(name):
    """Genreator for grippers"""
    if name == "TwoFingerGripper":
        return TwoFingerGripper()
    if name == "PR2Gripper":
        return PR2Gripper()
    if name == "RobotiqGripper":
        return RobotiqGripper()
    if name == "PushingGripper":
        return PushingGripper()
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper()
    raise XMLError('Unkown gripper name {}'.format(name))



