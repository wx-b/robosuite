from MujocoManip import *
import numpy as np
import time
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

if __name__ == '__main__':

    # a test case: do completely random actions at each time step
    #env = make("BaxterStackEnv", display=True, ignore_done=True, show_gripper_visualization=True, use_camera_obs=False)
    env = make("BaxterLiftEnv", display=True, ignore_done=True, show_gripper_visualization=True, use_camera_obs=False, use_eef_ctrl=not True)

    obs = env.reset()
    # rotate the gripper so we can see it easily 
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]*2)

    dof = env.dof
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))
    env.render()

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily 
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]*2)
        x = '''2.91920070e-01 -1.22491293e-01  1.42221222e-01  1.51313508e+00
          3.77805864e-03  3.13180972e-01 -2.60466116e-03 -7.69950644e-02
           -5.73647796e-01  2.45595288e-02  1.44862989e+00  5.72186995e-03
             3.07283274e-01  1.40614882e-03'''
        x = list(map(float, x.split()))
        env.set_robot_joint_positions(x)
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])


        for i in range(100000):

            action = 0.5 * np.random.randn(dof)
            action = np.zeros(9)
            action[0] = i/100000.*100
            action[1] = i/100000.*100
            action[2] = -i/100000.*100
            action = np.zeros(dof)
            #action[:3] = -np.array(list(map(float,env.model.objects[0].get('pos').split())))
            print(action[0])
            obs, reward, done, info = env.step(action)
            env.render()
            #print(env.sim, dir(env.sim))
            #print(env.sim.data.qpos)
            print(env.model.objects[0].get('pos'))

            if i % 100 == 0:
                print(i)
                # break

            if done:
                print('done: {}'.format(reward))
                break

