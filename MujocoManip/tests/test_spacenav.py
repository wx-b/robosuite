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
    ik_controller = IKController(bullet_data_path="/Users/ajaymandlekar/Desktop/Dropbox/Stanford/ccr/bullet3/data",
                                 robot_jpos_getter=robot_jpos_getter)

    gripper_controls = [[1., -1.], [-1., 1.]] 

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily 
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        for i in range(100000):
            # print(obs[len(obs) - 6: len(obs) - 3])
            # print(obs[len(obs) - 9: len(obs) - 6])
            # action = obs[len(obs) - 3: len(obs)]
            # action[:7] = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]
            # action = 0.01 * np.random.randn(dof)
            # action = np.zeros(dof)
            # action[7:] = [-1., 1.]
            # action[2] = 0.1

            # import pdb; pdb.set_trace()
            # grip_xpos = env.sim.data.get_site_xpos('grip_site')
            # action = 0.0001 * np.random.randn(dof)
            # import pdb; pdb.set_trace()
            # action[:3] = grip_xpos
            state = spacenav.get_controller_state()
            dpos, rotation, grasp = state["dpos"], state["rotation"], state["grasp"]
            velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
            action = np.concatenate([velocities, [grasp, -grasp]])
            print(action)
            # action[:] = [0, 0, 0, 0, 0, 0, 0.01]
            # action[:6] = [0, 0, 1.0, 0, 0, 0]
            # action[7] = 1
            obs, reward, done, info = env.step(action)
            # print(obs)
            env.render()
            # time.sleep(0.2)

            if i % 100 == 0:
                print(i)
                # break
            if done:
                print('done: {}'.format(reward))
                break

