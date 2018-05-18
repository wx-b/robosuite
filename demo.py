import time
from MujocoManip import make
import numpy as np

# env for training
# env = make("SawyerLiftEnv",
#             has_renderer=False,
#             ignore_done=True,
#             use_camera_obs=True,
#             camera_height=84,
#             camera_width=84,
#             camera_name='tabletop',
#             use_object_obs=False,
#             reward_shaping=True)

env = make("SawyerStackEnv", ignore_done=True, use_camera_obs=False)

while True:
    env.reset()
    env.render()
    for i in range(100000):
        env.step(np.random.randn(9) * 100)
        env.render()
