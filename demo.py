import time
from MujocoManip import make
import numpy as np
from IPython import embed


# env = make("SawyerLiftEnv", ignore_done=True, use_camera_obs=False)

env = make("SawyerBinsEnv", ignore_done=True, use_camera_obs=False)

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


while True:
    env.reset()
    env.render()
    env.viewer.set_camera(3)
    time.sleep(1)
    for i in range(100000):
        env.step(np.random.randn(9))
        env.render()
