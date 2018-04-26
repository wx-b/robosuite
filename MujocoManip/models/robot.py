import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
from MujocoManip.models.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.models.gripper import gripper_factory
from MujocoManip.models.model_util import *


class MujocoRobot(MujocoXML):
    """
        Base class for all robots
    """
    def __init__(self, fname):
        """
            Initialize from file @fname
        """
        super().__init__(fname)
        self.grippers = OrderedDict()

    def add_gripper(self, arm_name, gripper):
        """
            Adds a gripper (@gripper as MujocoGripper instance) to hand
            throws error if robot already has a gripper or gripper type is incorrect
        """
        if arm_name in self.grippers:
            raise ValueError('Attempts to add multiple grippers to one body')

        arm_subtree = self.worldbody.find(".//body[@name='{}']".format(arm_name))

        for actuator in gripper.actuator:
            if not (actuator.get('name') is not None and actuator.get('name').startswith('gripper')):
                raise XMLError('Actuator name {} does not have prefix "gripper"'.format(actuator.get('name')))

        for body in gripper.worldbody:
            arm_subtree.append(body)

        self.merge(gripper, merge_body=False)
        self.grippers[arm_name] = gripper

    @property
    def dof(self):
        """
            Returns the number of DOF of the robot, not including gripper
        """
        raise NotImplementedError

    @property
    def joints(self):
        """
            Returns a list of joint names of the robot
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
            Returns default qpos
        """
        raise NotImplementedError


class SawyerRobot(MujocoRobot):

    def __init__(self, use_eef_ctrl=False):
        if use_eef_ctrl:
            super().__init__(xml_path_completion('robot/sawyer/robot_mocap.xml'))
        else:
            super().__init__(xml_path_completion('robot/sawyer/robot.xml'))

        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set('pos', array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ['right_j{}'.format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])


class BaxterRobot(MujocoRobot):

    def __init__(self, use_eef_ctrl=False):
        if use_eef_ctrl:
            super().__init__(xml_path_completion('robot/baxter/robot_mocap.xml'))
        else:
            super().__init__(xml_path_completion('robot/baxter/robot.xml'))

        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0, 0, -0.913])
        self.left_hand = self.worldbody.find(".//body[@name='left_hand']")

    def set_base_xpos(self, pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set('pos', array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 14

    @property
    def joints(self):
        out = []
        for s in ['right_', 'left_']:
            out.extend(s+a for a in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'])
        return out

    @property
    def init_qpos(self):
        return np.array([-2.80245441e-04, -5.50127483e-01, -2.56679166e-04, 1.28390663e+00,
             -3.02081392e-05, 2.61554090e-01, 1.43798268e-06, 3.10821564e-09,
              -5.50000000e-01, 1.38161579e-09, 1.28400000e+00, 4.89875129e-11,
                2.61600000e-01, -2.71076012e-11])
