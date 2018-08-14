from MujocoManip import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector
import MujocoManip.miscellaneous.utils as U
import MujocoManip.miscellaneous.transformations as T

if __name__ == "__main__":

    # a test case: do completely random actions at each time step
    initializer = UniformRandomSampler(
        x_range=[0, 0.1],
        y_range=[0, 0.1],
        ensure_object_boundary_in_range=False,
        z_rotation=False,
    )
    env = make(
        "SawyerBinsEnv",
        ignore_done=True,
        show_gripper_visualization=True,
        use_object_obs=True,
        use_camera_obs=False,
        placement_initializer=initializer,
        reward_shaping=True,
    )

    obs = env.reset()
    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    dof = env.dof
    print("action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    while True:
        obs = env.reset()
        # env.viewer.set_camera(2)
        env.viewer.viewer._hide_overlay = True

        # rotate the gripper so we can see it easily
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        for i in range(100000):

            action = np.zeros(dof)  # 0.5 * np.random.randn(dof)
            obs, reward, done, info = env.step(action)
            # print(obs)

            ### quats are (w, x, y, z) for T, (x, y, z, w) for U ###

            print("rot")
            print(env._right_hand_orn)
            print("quat")
            print(env._right_hand_quat)
            print("testing rot")
            # print(U.quat2mat(env._right_hand_quat))
            print(
                T.euler_matrix(
                    *T.euler_from_quaternion(env._right_hand_quat[[3, 0, 1, 2]])
                )
            )
            print("testing quat")
            # print(U.mat2quat(env._right_hand_orn))
            print(T.quaternion_from_matrix(env._right_hand_orn)[[1, 2, 3, 0]])
            # print("testing rot-quat-rot")
            # print(U.quat2mat(U.mat2quat(env._right_hand_orn)))
            env.render()

            if i % 100 == 0:
                print(i)
                # break

            if done:
                print("done: {}".format(reward))
                break
