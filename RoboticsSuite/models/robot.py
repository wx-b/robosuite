import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
from RoboticsSuite.models.base import MujocoXML
from RoboticsSuite.utils import XMLError
from RoboticsSuite.models.gripper import gripper_factory
from RoboticsSuite.models.model_util import *


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
            raise ValueError("Attempts to add multiple grippers to one body")

        arm_subtree = self.worldbody.find(".//body[@name='{}']".format(arm_name))

        for actuator in gripper.actuator:
            if not (
                actuator.get("name") is not None
                and actuator.get("name").startswith("gripper")
            ):
                raise XMLError(
                    'Actuator name {} does not have prefix "gripper"'.format(
                        actuator.get("name")
                    )
                )

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
            super().__init__(xml_path_completion("robot/sawyer/robot_mocap.xml"))
        else:
            super().__init__(xml_path_completion("robot/sawyer/robot.xml"))

        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])


class BaxterRobot(MujocoRobot):
    def __init__(self):
        super().__init__(xml_path_completion("robot/baxter/robot.xml"))

        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0, 0, -0.913])
        self.left_hand = self.worldbody.find(".//body[@name='left_hand']")

    def set_base_xpos(self, pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 14

    @property
    def joints(self):
        out = []
        for s in ["right_", "left_"]:
            out.extend(s + a for a in ["s0", "s1", "e0", "e1", "w0", "w1", "w2"])
        return out

    @property
    def init_qpos(self):
        # Arms ready to work on the table
        return np.array(
            [
                0.5345519804154484,
                -0.09304970892037841,
                0.03846904167098771,
                0.16630128482283482,
                0.6432089753419815,
                1.9595760820824744,
                -1.297010528040148,
                -0.5177209505174132,
                -0.025671285598989225,
                -0.0764039726049278,
                0.17466058823908354,
                -0.7475668205536514,
                1.6408873046359498,
                -0.15750487460079515,
            ]
        )

        # Arms half extended
        return np.array(
            [
                0.75192989,
                -0.03836027,
                -0.02080136,
                0.16089599,
                0.34811643,
                2.09545544,
                -0.53135648,
                -0.58547775,
                -0.11657964,
                -0.03672875,
                0.16377322,
                -0.53622305,
                1.54251348,
                0.20374393,
            ]
        )

        # Arms fully extended
        return np.zeros(14)
