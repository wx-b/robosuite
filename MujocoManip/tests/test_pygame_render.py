import pygame
from MujocoManip import *
import numpy as np
import time
from PIL import Image
from IPython import embed
from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

if __name__ == '__main__':
    width = 512
    height = 384
    screen = pygame.display.set_mode((width, height))

    env = make("BaxterLiftEnv",#"BaxterHoleEnv",
               has_renderer=False,
               ignore_done=True,
               camera_height=height,
               camera_width=width,
               show_gripper_visualization=True,
               use_camera_obs=True,
               use_object_obs=False,
               use_eef_ctrl=False,
               )
    obs = env.reset()
    # rotate the gripper so we can see it easily 
    #env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]*2)

    dof = env.dof
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))



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
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            im = np.flip(obs['image'].transpose((1,0,2)),1)
            pygame.pixelcopy.array_to_surface(screen, im)
            pygame.display.update()
            # print(obs.keys())
            # print(obs['image'].shape)
            # print(env._peg_pose_in_hole_frame())

            if i % 100 == 0:
                print(i)
                # break

            if done:
                print('done: {}'.format(reward))
                break

