import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.models.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.models.model_util import *



class MujocoGripper(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname):
        super().__init__(fname)

    def format_action(self, action):
        """
            Given (-1,1) abstract control as np-array
            returns the (-1,1) for underlying actuators as 1-d np array 
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
            Returns rest(open) qpos of the gripper
        """
        raise NotImplementedError

    @property
    def dof(self):
        """
            Returns the number of DOF of the gripper
        """
        raise NotImplementedError

    def contact_geoms(self):
        """
            Returns a list of names corresponding to the geoms
            used to determine contact with the gripper.
        """
        return []

    def visualization_sites(self):
        """
            Returns a list of sites corresponding to the geoms
            used to aid visualization by human. (and showed be hidden from robots)
        """
        return []

    def visualization_geoms(self):
        """
            Returns a list of sites corresponding to the geoms
            used to aid visualization by human. (and showed be hidden from robots)
        """
        return []

    def hide_visualization(self):
        for site_name in self.visualization_sites():
            site = self.worldbody.find(".//site[@name='{}']".format(site_name))
            site.set('rgba', '0 0 0 0')
        for geom_name in self.visualization_geoms():
            site = self.worldbody.find(".//geom[@name='{}']".format(geom_name))
            geom.set('rgba', '0 0 0 0')


class TwoFingerGripperBase(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/two_finger_gripper.xml'))

    def format_action(self, action):
        return action
        # return np.array([-1 * action, 1 * action])

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def joints(self):
        return ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint']

    @property
    def dof(self):
        return 2

    def visualization_sites(self):
        return ['grip_site', 'grip_site_cylinder']

    def contact_geoms(self):
        return ["r_finger_g0", "r_finger_g1", "l_finger_g0", "l_finger_g1", "r_fingertip_g0", "l_fingertip_g0"]

class TwoFingerGripper(TwoFingerGripperBase):
    """
    Modifies two finger base to only take one action
    """
    def format_action(self, action):
        assert len(action) == 1
        return np.array([-1 * action[0], 1 * action[0]])

    @property
    def dof(self):
        return 1

class PR2Gripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/pr2_gripper.xml'))

    def format_action(self, action):
        return action
 #       return np.ones(4) * action

    @property
    def init_qpos(self):
        return np.zeros(4)

    @property
    def joints(self):
        return ['r_gripper_r_finger_joint', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_l_finger_tip_joint']

    @property
    def dof(self):
        return 4

    def contact_geoms(self):
        raise NotImplementedError

class RobotiqGripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/robotiq_gripper.xml'))

    def format_action(self, action):
        return action
#         return -1 * np.ones(6) * action

    @property
    def init_qpos(self):
        return [ 3.3161, 0., 0., 0., 0., 0.]

    @property
    def joints(self):
        return [
            'robotiq_85_left_knuckle_joint', 
            'robotiq_85_left_inner_knuckle_joint', 
            'robotiq_85_left_finger_tip_joint',
            'robotiq_85_right_knuckle_joint',
            'robotiq_85_right_inner_knuckle_joint',
            'robotiq_85_right_finger_tip_joint',
            ]

    @property
    def dof(self):
        return 6

    def contact_geoms(self):
        raise NotImplementedError

class PushingGripper(TwoFingerGripper):
    """Same as Two FingerGripper, but always closed"""
    def format_action(self, action):
        return np.array([1, -1])

    @property
    def dof(self):
        return 0

    def contact_geoms(self):
        raise NotImplementedError

class RobotiqThreeFingerGripper(MujocoGripper):
    def __init__(self):
        super().__init__(xml_path_completion('gripper/robotiq_gripper_s.xml'))


    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.zeros(11)

    @property
    def joints(self):
        return [
            "palm_finger_1_joint",
            "finger_1_joint_1",
            "finger_1_joint_2",
            "finger_1_joint_3",
            "palm_finger_2_joint",
            "finger_2_joint_1",
            "finger_2_joint_2",
            "finger_2_joint_3",
            "finger_middle_joint_1",
            "finger_middle_joint_2",
            "finger_middle_joint_3"
        ]

    @property
    def dof(self):
        return 11

    def contact_geoms(self):
        raise NotImplementedError

def gripper_factory(name):
    """Genreator for grippers"""
    if name == "TwoFingerGripper":
        return TwoFingerGripper()
    if name == "TwoFingerGripperBase":
        return TwoFingerGripperBase()
    if name == "PR2Gripper":
        return PR2Gripper()
    if name == "RobotiqGripper":
        return RobotiqGripper()
    if name == "PushingGripper":
        return PushingGripper()
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper()
    raise XMLError('Unkown gripper name {}'.format(name))



