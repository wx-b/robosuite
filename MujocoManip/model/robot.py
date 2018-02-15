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
            if not (actuator.get('name') is not None and actuator.get('name').startswith('gripper')):
                raise XMLError('Actuator name {} does not have prefix "gripper"'.format(actuator.get('name')))
        
        for body in gripper.worldbody:
            self.right_hand.append(body)

        self.merge(gripper, merge_body=False)
        
        self.has_gripper = True

    def dof(self):
        """
            Returns the number of DOF of the robot, not including gripper
        """
        raise NotImplementedError



class SawyerRobot(MujocoRobot):
    def __init__(self, use_torque_ctrl=False):
        if use_torque_ctrl:
            super().__init__(xml_path_completion('robot/sawyer/robot_torque.xml'))
        else:
            super().__init__(xml_path_completion('robot/sawyer/robot.xml'))
        # TODO: fix me to the correct value
        self.bottom_offset = np.array([0,0,-0.913])
    
    def place_on(self, on_pos):
        """place the robot on position @pos"""
        node = self.worldbody.find("./body[@name='base']")
        node.set('pos', array_to_string(on_pos - self.bottom_offset))

    def dof(self):
        """
            Returns the number of DOF of the robot, not including gripper
        """
        return 7

    @property
    def joints(self):
        return ['right_j{}'.format(x) for x in range(7)]

    @property
    def rest_pos(self):
        return [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]