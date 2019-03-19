"""
This file implements a wrapper for controlling the robot through end effector
movements instead of joint velocities. This is useful in learning pipelines
that want to output actions in end effector space instead of joint space.
"""

import os
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper
import pybullet as p

class IKWrapper(Wrapper):
    env = None

    def __init__(self, env, action_repeat=1):
        """
        Initializes the inverse kinematics wrapper.
        This wrapper allows for controlling the robot through end effector
        movements instead of joint velocities.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            action_repeat (int): Determines the number of times low-level joint
                control actions will be commanded per high-level end effector
                action. Higher values will allow for more precise control of
                the end effector to the commanded targets.
        """
        super().__init__(env)
        if self.env.mujoco_robot.name == "sawyer":
            from robosuite.controllers import SawyerIKController

            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        elif self.env.mujoco_robot.name == "baxter":
            from robosuite.controllers import BaxterIKController

            self.controller = BaxterIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        self.action_repeat = action_repeat

    def set_robot_joint_positions(self, positions):
        """
        Overrides the function to set the joint positions directly, since we need to notify
        the IK controller of the change.
        """
        self.env.set_robot_joint_positions(positions)
        self.controller.sync_state()

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def reset(self):
        ret = super().reset()
        self.controller.sync_state()
        return ret

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

                Note: When wrapping around a Baxter environment, the indices 0-6 inidicate the
                right hand. Indices 7-13 indicate the left hand, and the rest (*) are the gripper
                inputs (first right, then left).
        """

        input_1 = self._make_input(action[:7], self.env._right_hand_quat)
        if self.env.mujoco_robot.name == "sawyer":
            velocities = self.controller.get_control(**input_1)
            low_action = np.concatenate([velocities, action[7:]])
        elif self.env.mujoco_robot.name == "baxter":
            input_2 = self._make_input(action[7:14], self.env._left_hand_quat)
            velocities = self.controller.get_control(input_1, input_2)
            low_action = np.concatenate([velocities, action[14:]])
        else:
            raise Exception(
                "Only Sawyer and Baxter robot environments are supported for IK "
                "control currently."
            )

        if self.done:
            raise ValueError("executing action in terminated episode")

        self.env.timestep += 1

        low_action[-1] = np.clip(low_action[-1], -1, 1)
        gripper = self.env.gripper.format_action([low_action[-1]])
        ctrl_range = self.env.sim.model.actuator_ctrlrange

        joint_actions = low_action[:-1]
        joint_actions = np.clip(joint_actions, ctrl_range[:-2,0], ctrl_range[:-2,1])
        self.env.sim.data.ctrl[:] = np.concatenate([joint_actions, gripper])

        self.env.sim.data.qfrc_applied[
            self.env._ref_joint_vel_indexes
        ] = self.env.sim.data.qfrc_bias[self.env._ref_joint_vel_indexes]

        try:
            end_time = self.env.cur_time + self.env.control_timestep
            while self.cur_time < end_time:
                self.env.sim.step()
                self.env.cur_time += self.env.model_timestep
        except Exception as e:
            print(e)
            import os
            os._exit(1)

        self.env.cur_time += self.env.model_timestep

        reward, done, info = self.env._post_action(action)
        return self.env._get_observation(), reward, done, info

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """

        act = T.mat2euler(T.quat2mat(action[3:7]))
        act = np.clip(act, -np.pi/40, np.pi/40)
        act = p.getQuaternionFromEuler(act)

        rotation = T.quat2mat(T.quat_multiply(old_quat, act))
        return {
            "dpos": action[:3], #np.clip(action[:3], -0.05, 0.05),
            # IK controller takes an absolute orientation in robot base frame
            "rotation": rotation,
        }
