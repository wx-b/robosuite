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
    env = make("BaxterLiftEnv",#"BaxterHoleEnv",
               has_renderer=True,
               ignore_done=True,
               show_gripper_visualization=True,
               use_camera_obs=True,
               use_object_obs=False,
               use_eef_ctrl=False,
               )
    # env = make("BaxterHoleEnv", display=True, ignore_done=True, show_gripper_visualization=True, use_camera_obs=False, use_eef_ctrl=not True)
    # print(env.model.get_xml())
    # exit(0)
    obs = env.reset()
    # rotate the gripper so we can see it easily 
    #env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]*2)

    dof = env.dof
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))
    env.render()

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily 
        #env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]*2)
        x = '''2.91920070e-01 -1.22491293e-01  1.42221222e-01  1.51313508e+00
          3.77805864e-03  3.13180972e-01 -2.60466116e-03 -7.69950644e-02
           -5.73647796e-01  2.45595288e-02  1.44862989e+00  5.72186995e-03
             3.07283274e-01  1.40614882e-03'''
        x = list(map(float, x.split()))
        #env.set_robot_joint_positions(x)
        # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        #while True: env.render()


        for i in range(100000):
            #print((env.model.worldbody.find(".//body[@name='left_hand']")).keys())

            action = 0.5 * np.random.randn(dof)
            obs, reward, done, info = env.step(action)
            env.render()
            print(reward)
            # print(env._peg_pose_in_hole_frame())


            if i % 100 == 0:
                print(i)
                # break

            if done:
                print('done: {}'.format(reward))
                break

