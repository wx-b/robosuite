from MujocoManip import *
import numpy as np
import time
import datetime
import pickle
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

from MujocoManip.miscellaneous.spacenavigator import SpaceNavigator
from MujocoManip.miscellaneous.ik_controller import IKController

if __name__ == "__main__":

    # a test case: do completely random actions at each time step
    env = make(
        "SawyerStackEnv",
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
    )

    # function to return robot joint angles
    def robot_jpos_getter():
        return np.array(env._joint_positions)

    obs = env.reset()

    dof = env.dof
    print("action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    spacenav = SpaceNavigator()
    ik_controller = IKController(
        bullet_data_path="../models/assets/bullet_data",
        robot_jpos_getter=robot_jpos_getter,
    )

    gripper_controls = [[1.], [-1.]]

    success_eps = 0
    while True:
        obs = env.reset()
        env.viewer.set_camera(2)
        env.viewer.viewer._run_speed /= 2.0
        # rotate the gripper so we can see it easily
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        episode_data = []

        for i in range(1000):
            state = spacenav.get_controller_state()
            dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
            velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
            action = np.concatenate([velocities, [grasp, -grasp]])

            episode_data.append(
                {
                    "dpos": dpos,
                    "rotation": rotation,
                    "grasp": grasp,
                    "action": action,
                    "state": env.sim.get_state(),
                }
            )

            obs, reward, done, info = env.step(action)
            env.render()

        if reward >= 2.0:
            t_now = time.time()
            time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y%m%d%H%M%S")
            pickle.dump(
                episode_data, open("SawyerStackEnv_{}.pkl".format(time_str), "wb")
            )
            success_eps += 1

        if success_eps >= 12:
            break
