import MujocoManip as MM
import numpy as np

env = MM.make("SawyerStackEnv", display=True, use_eef_ctrl=True)
obs = env.reset()
dof = env.dof()
print('action space', env.action_space)
print('Obs: {}'.format(len(obs)))
print('DOF: {}'.format(dof))
# env.render()

while True:
    obs = env.reset()

    for i in range(20000):
        action = 0.01 * np.random.randn(9)
        action[3:7] = [0., 1., 0., 0.] 
        obs, reward, done, info = env.step(action)
        # env.render()
        A = env.viewer.render_frame()
        if i % 100 == 0:
            print(i)
        if done:
            print('done: {}'.format(reward))
            break


