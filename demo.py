import time
from MujocoManip import make
import numpy as np
from IPython import embed

# env = make("SawyerLiftEnv", ignore_done=True, use_camera_obs=False)

env = make("BinsEnv", ignore_done=True, use_camera_obs=False)

while True:
    env.reset()
    env.render()
    env.viewer.set_camera(3)
    for i in range(100000):
        env.step(np.random.randn(9))
        env.render()
