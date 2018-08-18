from RoboticsSuite import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from RoboticsSuite.models import *
from RoboticsSuite.wrappers import DataCollector

from RoboticsSuite.utils.baxter_ik import BaxterIKController as IKController

if __name__ == "__main__":

    # a test case: do completely random actions at each time step
    env = make(
        "BaxterHoleEnv",
        display=True,
        ignore_done=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )

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

    dof = 9
    print("action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    side = "right"

    def robot_jpos_getter():
        if side == "left":
            return np.array(env._joint_positions[7:])
        else:
            return np.array(env._joint_positions[:7])

    # space_mouse = SpaceMouse()
    ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data/baxter_common/%s.urdf" % side,
        robot_jpos_getter=robot_jpos_getter,
    )

    gripper_controls = [[1., -1.], [-1., 1.]]

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        state = {"dpos": np.array([0, 0, 1]), "rotation": np.eye(3), "grasp": 0}
        dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
        print(ik_controller.ik_robot_target_pos)
        # ik_controller.joint_positions_for_user_displacement(dpos, rotation)
        print(ik_controller.ik_robot_target_pos)
        # env.set_robot_joint_positions(list(env._joint_positions[:7]) + list(ik_controller.joint_positions_for_user_displacement(dpos, rotation)))
        # while True:env.render()

        for i in range(100000):
            # state = space_mouse.get_controller_state()
            state = {"dpos": np.array([0, 0, 1e-4]), "rotation": np.eye(3), "grasp": 0}
            dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
            velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
            if side == "left":
                action = np.concatenate([np.zeros(7), velocities, np.zeros(4)])
            else:
                action = np.concatenate([velocities, np.zeros(7), np.zeros(4)])
            # action = np.concatenate([velocities, [grasp, -grasp]])
            # action = np.concatenate([np.zeros(7)+1, np.zeros(2), np.zeros(9)])

            obs, reward, done, info = env.step(action)
            env.render()

            # if i % 100 == 0:
            #     print(i)

            if done:
                print("done: {}".format(reward))
                break
