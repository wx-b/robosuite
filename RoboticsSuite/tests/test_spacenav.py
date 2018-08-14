from RoboticsSuite import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from RoboticsSuite.models import *
from RoboticsSuite.wrappers import DataCollector

from RoboticsSuite.miscellaneous.spacenavigator import SpaceNavigator
from RoboticsSuite.miscellaneous.ik_controller import IKController

if __name__ == "__main__":

    env = make(
        "SawyerBinsCanEnv",
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        single_object_mode=True,
    )

    # function to return robot joint angles
    def robot_jpos_getter():
        return np.array(env._joint_positions)

    obs = env.reset()
    env.viewer.set_camera(2)
    env.viewer.viewer._hide_overlay = True

    dof = env.dof
    print("Action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    spacenav = SpaceNavigator()
    ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data",
        robot_jpos_getter=robot_jpos_getter,
    )

    gripper_controls = [[1.], [-1.]]

    while True:
        obs = env.reset()
        env.viewer.set_camera(2)
        env.viewer.viewer._hide_overlay = True

        # rotate the gripper so we can see it easily
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        for i in range(100000):
            state = spacenav.get_controller_state()
            dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
            velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
            action = np.concatenate([velocities, [grasp, -grasp]])

            obs, reward, done, info = env.step(action)
            env.render()
            print("reward:", reward)

            if done:
                print("done: {}".format(reward))
                break
