"""End-effector control for bimanual Baxter robot.

This script shows how to use inverse kinematics solver from Bullet
to command the end-effectors of two arms of the Baxter robot.
"""

import os
import numpy as np

import RoboticsSuite
from RoboticsSuite.models import *
from RoboticsSuite.controllers.baxter_ik_controller import BaxterIKController

if __name__ == "__main__":

    # initialize a Baxter environment
    env = RoboticsSuite.make(
        "BaxterLift",
        ignore_done=True,
        has_renderer=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
    ])

    bullet_data_path = os.path.join(RoboticsSuite.assets_path, "bullet_data")

    def robot_jpos_getter():
        return np.array(env._joint_positions)

    ik_controller = BaxterIKController(
        bullet_data_path=bullet_data_path, robot_jpos_getter=robot_jpos_getter
    )

    # gripper_controls = [[1., -1.], [-1., 1.]]

    for t in range(100000):
        omega = 2 * np.pi / 1000.
        orn = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        A = 1e-3
        action_right = {
            "dpos": np.array([A * np.cos(omega * t), 0, A * np.sin(omega * t)]),
            "rotation": orn,
        }
        action_left = {
            "dpos": np.array([A * np.sin(omega * t), A * np.cos(omega * t), 0]),
            "rotation": orn,
        }

        velocities = ik_controller.get_control(action_right, action_left)
        assert len(velocities) == 14
        grasp = 0.
        action = np.concatenate([velocities, [grasp, grasp]])

        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            break
