from MujocoManip import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

from MujocoManip.miscellaneous.baxter_ik import BaxterIKController as IKController
import MujocoManip.miscellaneous.utils as U

import threading

from collections import defaultdict as dd
import glfw

ispressed = dd(bool)
flipflop = dd(bool)


def okp(window, key, scancode, action, mods):
    ispressed[key] = True
    flipflop[key] = True


def oku(window, key, scancode, action, mods):
    ispressed[key] = False


control = [0 for _ in range(6)]
button = 0

if __name__ == "__main__":

    # a test case: do completely random actions at each time step
    env = make(
        "BaxterHoleEnv",
        display=True,
        ignore_done=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )
    env.viewer.add_keypress_callback("any", okp)
    env.viewer.add_keyup_callback("any", oku)

    # function to return robot joint angles

    obs = env.reset()

    # rotate the gripper so we can see it easily
    # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
    env.set_robot_joint_positions(
        [
            -2.80245441e-04,
            -5.50127483e-01,
            -2.56679166e-04,
            1.28390663e+00,
            -3.02081392e-05,
            2.61554090e-01,
            1.43798268e-06,
            3.10821564e-09,
            -5.50000000e-01,
            1.38161579e-09,
            1.28400000e+00,
            4.89875129e-11,
            2.61600000e-01,
            -2.71076012e-11,
        ]
    )

    # spacenav = SpaceNavigator()

    dof = 9
    print("action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    side = "right"

    def left_robot_jpos_getter():
        return np.array(env._joint_positions[7:])

    def right_robot_jpos_getter():
        return np.array(env._joint_positions[:7])

    # spacenav = SpaceNavigator()
    rest = [
        -2.80245441e-04,
        -5.50127483e-01,
        -2.56679166e-04,
        1.28390663e+00,
        -3.02081392e-05,
        2.61554090e-01,
        1.43798268e-06,
        3.10821564e-09,
        -5.50000000e-01,
        1.38161579e-09,
        1.28400000e+00,
        4.89875129e-11,
        2.61600000e-01,
        -2.71076012e-11,
    ]
    left_ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data/baxter_common/left.urdf",
        robot_jpos_getter=left_robot_jpos_getter,
        rest_poses=rest[7:],
    )
    right_ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data/baxter_common/right.urdf",
        robot_jpos_getter=right_robot_jpos_getter,
        rest_poses=rest[:7],
    )

    con = {"right": right_ik_controller, "left": left_ik_controller}

    gripper_controls = [[1., -1.], [-1., 1.]]

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        state = {"dpos": np.array([0, 0, 0]), "rotation": np.eye(3), "grasp": 0}
        dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
        # print(ik_controller.ik_robot_target_pos)
        # ik_controller.joint_positions_for_user_displacement(dpos, rotation)
        # print(ik_controller.ik_robot_target_pos)
        # env.set_robot_joint_positions(list(env._joint_positions[:7]) + list(ik_controller.joint_positions_for_user_displacement(dpos, rotation)))
        # while True:env.render()
        rotation = {
            "left": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            "right": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        }
        rotation["right"] = rotation["right"].dot(
            U.rotation_matrix(angle=-np.pi / 2., direction=[1, 0, 0], point=None)[
                :3, :3
            ]
        )
        rotation["left"] = rotation["left"].dot(
            U.rotation_matrix(angle=-np.pi / 2., direction=[0, 0, 1], point=None)[
                :3, :3
            ]
        )
        # print(rotation)
        # exit()

        for i in range(100000):
            kk = 0.005
            control = [0 for _ in range(6)]
            if ispressed[glfw.KEY_F1]:
                control[0] = kk
            if ispressed[(glfw.KEY_F2)]:
                control[0] = -kk
            if ispressed[(glfw.KEY_F3)]:
                control[1] = -kk
            if ispressed[(glfw.KEY_F4)]:
                control[1] = kk
            if ispressed[(glfw.KEY_F5)]:
                control[2] = -kk
            if ispressed[(glfw.KEY_F6)]:
                control[2] = kk
            if flipflop["k"]:
                grasp = 1
                flipflop["k"] = False
            dpos = np.array(control[:3])

            if grasp != 0:
                print("switching")
                grasp = 0
                side = "right" if side == "left" else "left"
                control = [0 for _ in range(6)]
                dpos = np.zeros(3)
                con[side].setup_inverse_kinematics()
                con[side].sync_state()
            # print(side,dpos,rotation)
            velocities = con[side].get_control(dpos=dpos, rotation=rotation[side])
            action = np.zeros(18)
            st = np.zeros(14)

            st[:7] = (
                con["right"].ik_solution
                if hasattr(con["right"], "ik_solution")
                else right_robot_jpos_getter()
            )
            st[7:] = (
                con["left"].ik_solution
                if hasattr(con["left"], "ik_solution")
                else left_robot_jpos_getter()
            )
            env.set_robot_joint_positions(st)
            if side == "left":
                action = np.concatenate([np.zeros(7), velocities, np.zeros(4)])
            else:
                action = np.concatenate([velocities, np.zeros(7), np.zeros(4)])
            # action = np.concatenate([velocities, [grasp, -grasp]])
            # action = np.concatenate([np.zeros(7)+1, np.zeros(2), np.zeros(9)])
            if grasp:
                button = 0

            # action = np.random.randn(18)
            obs, reward, done, info = env.step(action)
            # print("reward", reward)
            env.render()

            # if i % 100 == 0:
            #     print(i)

            if done:
                print("done: {}".format(reward))
                break
