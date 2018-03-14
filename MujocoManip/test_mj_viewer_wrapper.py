import MujocoManip as MM
from MujocoManip.environment.sawyer_viewer import SawyerViewer
import numpy as np
import time

viewer = SawyerViewer("SawyerStackEnv")
viewer.reset()
viewer.render()

while True:
    viewer.reset()

    for i in range(20000):
        action = 1. - 2. * np.random.rand(9)
        # action = 0.01 * np.random.randn(9)
        # action[3:7] = [0., 1., 0., 0.] 
        viewer.step(action)
        viewer.render()