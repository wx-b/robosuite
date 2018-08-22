"""Teleoperate robot with SpaceMouse.

We use the SpaceMouse 3D mouse to control the end-effector of the robot.
The mouse provides 6-DoF control commands. The commands are mapped to joint
velocities through an inverse kinematics solver from Bullet physics.

The two side buttons of SpaceMouse are used for controlling the grippers.

SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
We used the SpaceMouse Wireless in our experiments. The paper below used the same device
to collect human demonstrations for imitation learning.

Reinforcement and Imitation Learning for Diverse Visuomotor Skills
Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
RSS 2018

Example:
    $ python demo_spacemouse_ik_control.py --environment SawyerPickPlaceCan

Note:
    This current script only supports Mac OS X (Linux support can be added) and
    Download and install the driver before running the script:
        https://www.3dconnexion.com/service/drivers.html

"""

import argparse
import numpy as np

import RoboticsSuite
from RoboticsSuite.models import *
from RoboticsSuite.wrappers import DataCollector
from RoboticsSuite.controllers.spacemouse import SpaceMouse
from RoboticsSuite.controllers.ik_controller import IKController


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerPickPlaceCan")
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()

    env = RoboticsSuite.make(
        args.environment,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100,
    )

    # function to return robot joint angles
    def robot_jpos_getter():
        return np.array(env._joint_positions)

    # initialize space_mouse controller
    space_mouse = SpaceMouse()

    # initialize IK controller
    ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data",
        robot_jpos_getter=robot_jpos_getter,
    )

    gripper_controls = [[1.], [-1.]]

    obs = env.reset()
    env.viewer.set_camera(2)
    env.viewer.viewer._hide_overlay = True
    env.render()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    for i in range(args.timesteps):
        state = space_mouse.get_controller_state()
        dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
        velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
        action = np.concatenate([velocities, [grasp, -grasp]])

        obs, reward, done, info = env.step(action)
        env.render()
        print("reward: {0:.2f}".format(reward))

        if done:
            break
