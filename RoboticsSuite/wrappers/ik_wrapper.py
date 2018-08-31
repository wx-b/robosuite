"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import os
import numpy as np
import RoboticsSuite
import RoboticsSuite.utils as U
from RoboticsSuite.wrappers import Wrapper


class IKWrapper(Wrapper):
    env = None

    def __init__(self, env):
        """
        Initializes the inverse kinematics wrapper.
        This wrapper allows for controlling the robot through end effector 
        movements instead of joint velocities.

        Args:
            env (MujocoEnv instance): The environment to wrap.
        """
        super().__init__(env)
        if self.env.mujoco_robot.name == "sawyer":
            from RoboticsSuite.controllers.sawyer_ik_controller import SawyerIKController
            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(RoboticsSuite.assets_path, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        elif self.env.mujoco_robot.name == "baxter":
            from RoboticsSuite.controllers.baxter_ik_controller import BaxterIKController
            self.controller = BaxterIKController(
                bullet_data_path=os.path.join(RoboticsSuite.assets_path, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception("Only Sawyer and Baxter robot environments are supported for IK control currently.")

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def step(self, action):
        """
        Move the end effector(s) according to the input control.

        Args:
            action (numpy array): The array should have the corresponding elements.
                0-2: The desired change in end effector position in x, y, and z.
                3-6: The desired change in orientation, expressed as a (x, y, z, w) quaternion.
                    Note that this quaternion encodes a relative rotation with respect to the
                    current gripper orientation. If the current rotation is r, this corresponds
                    to a quaternion d such that r * d will be the new rotation. 
                *: Controls for gripper actuation.

                Note: Baxter environments should supply two such actions concatenated
                together, one for each end effector.
            
        """

        ### TODO(jcreus): support the Baxter here as well. ###
        dpos = action[:3]
        dquat = action[3:7]

        # IK controller takes an absolute orientation in robot base frame
        rotation = U.quat2mat(U.quat_multiply(self.env._right_hand_quat, action[3:7]))
        velocities = self.controller.get_control(dpos=dpos, rotation=rotation)
        action = np.concatenate([velocities, action[7:]])
        return self.env.step(action)



