from MujocoManip import SawyerStackEnv, SawyerPushEnv
import numpy as np

if __name__ == '__main__':

    ### TODO: for some reason, when you just open a new terminal, import the env, do reset, then render, ###
    ###       it doesn't render the correct configuration. ###
    ### TODO: define observation space, action space (you can look at Julian's code for this) ###

    # a test case: do completely random actions at each time step
    # env = SawyerStackEnv(gripper='TwoFingerGripper')
    env = SawyerStackEnv(gripper='RobotiqGripper')
    # env = SawyerStackEnv(gripper='PR2Gripper')
    obs = env._reset()
    print('action space', env.action_space)
    print('Initial Obs: {}'.format(obs))
    while True:
        obs = env._reset()

        gripper_pos = -1
        ### TODO: we should implement 
        ### TODO: this might need clipping ###
        action = -1 * np.random.randn(8) / 2
        action[7] = gripper_pos
        # action[7] *= 0.020833
        for i in range(2000):
            # if i % 100 == 0:
            #     print("gripper_l_finger: {}".format(env.sim.data.qpos[env.model.get_joint_qpos_addr('r_gripper_l_finger_joint')]))
            # print("gripper: {}".format(env.sim.data.qpos[env.model.get_joint_qpos_addr('robotiq_85_left_knuckle_joint')]))
            action = -1 * np.random.randn(8) / 2
            action[7] = gripper_pos
            obs, reward, done, info = env._step(action)
            # 
            # obs, reward, done, info = env._step([0,-1,0,0,0,0,2])
            # print(obs, reward, done, info)
            env._render()
            if done:
                print('done: {}'.format(reward))
                break