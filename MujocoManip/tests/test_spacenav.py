from MujocoManip import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

from MujocoManip.miscellaneous.spacenavigator import SpaceNavigator
from MujocoManip.miscellaneous.ik_controller import IKController

if __name__ == '__main__':

    # a test case: do completely random actions at each time step
    env = make("SawyerStackEnv", display=True, ignore_done=True)

    # function to return robot joint angles
    def robot_jpos_getter():
        return np.array(env._joint_positions)

    obs = env.reset()

    # rotate the gripper so we can see it easily 
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    dof = 9
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))
    env.render()

    spacenav = SpaceNavigator()
    ik_controller = IKController(bullet_data_path="/Users/yukez/Research/bullet3/data/",
                                 robot_jpos_getter=robot_jpos_getter)

    gripper_controls = [[1., -1.], [-1., 1.]] 

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily 
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        for i in range(100000):
            state = spacenav.get_controller_state()
            dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
            velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
            action = np.concatenate([velocities, [grasp, -grasp]])

            obs, reward, done, info = env.step(action)
            env.render()

            # if i % 100 == 0:
            #     print(i)

            if done:
                print('done: {}'.format(reward))
                break

