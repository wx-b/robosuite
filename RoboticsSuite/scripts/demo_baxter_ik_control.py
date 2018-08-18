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
        "BaxterPegInHole",
        display=True,
        ignore_done=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
        0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00
    ])

    side = "right"

    # function to return robot joint angles
    def robot_jpos_getter():
        if side == "left":
            return np.array(env._joint_positions[7:])
        else:
            return np.array(env._joint_positions[:7])

    urdf_file_path = "models/assets/bullet_data/baxter_common/%s.urdf" % side
    bullet_data_path = os.path.join(RoboticsSuite.__path__[0], urdf_file_path)

    ik_controller = BaxterIKController(
        bullet_data_path=bullet_data_path,
        robot_jpos_getter=robot_jpos_getter,
    )

    gripper_controls = [[1., -1.], [-1., 1.]]

    for i in range(100000):
        # TODO(joan): sample random dpos and rotation here
        state = {"dpos": np.array([0, 0, 1e-4]), "rotation": np.eye(3), "grasp": 0}
        dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]

        velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
        if side == "left":
            action = np.concatenate([np.zeros(7), velocities, np.zeros(4)])
        else:
            action = np.concatenate([velocities, np.zeros(7), np.zeros(4)])

        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            break