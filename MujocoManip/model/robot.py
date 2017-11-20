import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.gripper import MujocoGripper
from MujocoManip.model.model_util import *

class MujocoRobot(MujocoXML):
    """
        Base class for all robots
    """
    # TODO: custom actuator logic
    def __init__(self, fname):
        """
            Initialize from file @fname
        """
        super().__init__(fname)
        self.right_hand = self.worldbody.find(".//body[@name='right_hand']")
        self.has_gripper = False
        if self.right_hand is None:
            print('Error: body with name "right_hand" not found.')
            raise ValueError

    def add_gripper(self, gripper):
        """
            Adds a gripper (@gripper as MujocoGripper instance) to hand
            throws error if robot already has a gripper or gripper type is incorrect
        """
        if self.has_gripper:
            raise XMLError('Attempts to add multiple grippers')
        if not isinstance(gripper, MujocoGripper):
            raise XMLError('{} is not a MujocoGripper instance.'.format(type(gripper)))
        for actuator in gripper.actuator:
            self.actuator.append(actuator)
        for asset in gripper.asset:
            self.asset.append(asset)
        for body in gripper.worldbody:
            self.right_hand.append(body)
        self.has_gripper = True

class SawyerRobot(MujocoRobot):
    def __init__(self):
        super().__init__(xml_path_completion('robot/sawyer/robot.xml'))
        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0,0,-0.913])
    
    def place_on(self, on_pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set('pos', array_to_string(on_pos - self.bottom_offset))
