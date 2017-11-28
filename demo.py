import time
from MujocoManip import SawyerStackEnv
import numpy as np


env = SawyerStackEnv()
while True:
	env._reset()
	env._render()
    # import pdb; pdb.set_trace()
	for i in range(1000):
		env._step(np.random.randn(8) * 100)
		env._render()
