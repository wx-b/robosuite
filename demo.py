import time
from MujocoManip import make
import numpy as np

# env = make("SawyerStackEnv", use_camera_obs=False, ignore_done=True)
env = make("ApcEnv", use_camera_obs=False, ignore_done=True)

while True:
	env.reset()
	# env.render()
	for i in range(100000):
		env.step(np.random.randn(9) * 100)
		env.render()
