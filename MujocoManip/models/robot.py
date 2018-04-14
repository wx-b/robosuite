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

    def add_gripper(self, arm_name, gripper_type):
        """
            Adds a gripper (@gripper as MujocoGripper instance) to hand
            throws error if robot already has a gripper or gripper type is incorrect
        """
        if arm_name in self.grippers:
            raise ValueError('Attempts to add multiple grippers to one body')

        arm_subtree = self.worldbody.find(".//body[@name='{}']".format(arm_name))
        gripper = gripper_factory(gripper_type)

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

    def __init__(self, use_torque_ctrl=False, use_eef_ctrl=False):
        if use_torque_ctrl:
            super().__init__(xml_path_completion('robot/sawyer/robot_torque.xml'))
        elif use_eef_ctrl:
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
