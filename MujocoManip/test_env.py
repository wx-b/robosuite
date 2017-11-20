from MujocoManip.env import SawyerStackEnv, SawyerPushEnv
import numpy as np

if __name__ == '__main__':

    ### TODO: for some reason, when you just open a new terminal, import the env, do reset, then render, ###
    ###       it doesn't render the correct configuration. ###
    ### TODO: put in action range clipping ###
    ### TODO: define observation space, action space (you can look at Julian's code for this) ###

    # a test case: do completely random actions at each time step
    env = SawyerPushEnv()
    obs = env._reset()
    print('Initial Obs: {}'.format(obs))
    while True:
        obs = env._reset()

        ### TODO: we should implement 
        ### TODO: this might need clipping ###
        action = np.random.randn(8)
        action[7] *= 0.020833
        for i in range(2000):
            action = np.random.randn(8)
            action[7] *= 0.020833
            print(action)
            obs, reward, done, info = env._step(action)
            # 
            # obs, reward, done, info = env._step([0,-1,0,0,0,0,2])
            # print(obs, reward, done, info)
            env._render()
            if done:
                print('done: {}'.format(reward))
                break