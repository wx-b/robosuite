from MujocoManip import SawyerStackEnv, SawyerPushEnv, SawyerGraspEnv, SawyerReachEnvEEVel, SawyerReachEnv, make
import numpy as np
import time
from PIL import Image
if __name__ == '__main__':

    ### TODO: for some reason, when you just open a new terminal, import the env, do reset, then render, ###
    ###       it doesn't render the correct configuration. ###
    ### TODO: define observation space, action space (you can look at Julian's code for this) ###

    # a test case: do completely random actions at each time step
    # env = SawyerStackEnv(gripper='TwoFingerGripper')
    # env = SawyerStackEnv(gripper='RobotiqGripper')
    # env = SawyerStackEnv(gripper='PR2Gripper')
    # env = SawyerPushEnv()
    # env = SawyerGraspEnv(gripper="RobotiqThreeFingerGripper")
    # env = SawyerGraspEnv(gripper="RobotiqThreeFingerGripper")
    # env = SawyerReachEnv(end_effector_control=True, reward_objective_factor=500)
    # env = make("SawyerReachEnv", display=True, ignore_done=False)
    # env = make("SawyerStackEnv", display=True, ignore_done=True)
    env = make("SawyerStackEnv", display=True, ignore_done=True, use_eef_ctrl=True)
    # env = make("SawyerStackEnv", display=True, ignore_done=False, use_torque_ctrl=True)
    # env = make("SawyerPushEnv", display=True, ignore_done=False, use_torque_ctrl=True)
    obs = env._reset()
    dof = env.dof()
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))
    env._render()

    # from IPython import embed
    # embed()

    while True:
        obs = env._reset()
        for i in range(20000):
            # print(obs[len(obs) - 6: len(obs) - 3])
            # print(obs[len(obs) - 9: len(obs) - 6])
            # action = obs[len(obs) - 3: len(obs)]
            # action[:7] = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]
            # action = np.random.randn(dof)
            action = np.zeros(5)
            action[2] = 0.1

            # action[:] = [0, 0, 0, 0, 0, 0, 0.01]
            # action[:6] = [0, 0, 1.0, 0, 0, 0]
            # action[7] = 1
            obs, reward, done, info = env._step(action)
            env._render()
            # if i % 500 == 0:
            #     A = env.sim.render(width=500, height=500, camera_name='camera1')
            #     im = Image.fromarray(A)
            #     im.save('{}.jpg'.format(i))
            # print('obs: {}'.format(obs))
            # print('reward: {}'.format(reward))
            # t = time.time()
            # A = env.sim.render(500, 500)
            # dt = time.time() - t
            # print('dt: {}'.format(dt))
            if i % 100 == 0:
                print(i)
            if done:
                print('done: {}'.format(reward))
                break

