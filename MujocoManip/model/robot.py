import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.gripper import MujocoGripper

class MujocoRobot(MujocoXML):
    """
        Base class for all robots
        Since we will only be having sawyer for a while, all sawyer methods are put in here
    """
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

    # TODO: custom actuator logic