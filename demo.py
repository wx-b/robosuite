import time
from MujocoManip import make
import numpy as np

# env = make("SawyerLiftEnv", ignore_done=True, use_camera_obs=False)
env = make("ApcEnv", ignore_done=True, use_camera_obs=False)

while True:
	env.reset()
	# env.render()
	for i in range(100000):
		env.step(np.random.randn(9) * 100)
		env.render()
