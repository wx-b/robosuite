import time
from MujocoManip import make
import numpy as np

# env = make("SawyerEnv", display=True, ignore_done=True)
env = make("SawyerLiftEnv", display=True, ignore_done=True)

while True:
	env.reset()
	env.render()
	for i in range(1000):
		env.step(np.random.randn(9) * 100)
		env.render()