import MujocoManip as MM
import numpy as np
import time

### TODO: try camera id 0, 1, 2, 3
CAM_ID = 3

env = MM.make("SawyerStackEnv", display=False)
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
        t = time.time()
        A = env.physics.render(480, 480, camera_id=CAM_ID)
        print("got shape {} image in time {}".format(A.shape, time.time() - t))
        # A = env.viewer.render_frame()

# from dm_control import suite
# import numpy as np
# import time

# env = suite.load("cartpole", "swingup")
# obs = env.reset()
# action_spec = env.action_spec()

# while True:
#     obs = env.reset()

#     time_step = env.reset()
#     while not time_step.last():
#         action = np.random.uniform(action_spec.minimum,
#                                  action_spec.maximum,
#                                  size=action_spec.shape)
#         time_step = env.step(action)
#         t = time.time()
#         A = env.physics.render(480, 480)
#         print("took {} sec".format(time.time() - t))


