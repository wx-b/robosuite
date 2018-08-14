# from RoboticsSuite.model import *
import sys
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np

model = load_model_from_path(sys.argv[1])
sim = MjSim(model)
viewer = MjViewer(sim)

# sim_state = sim.get_state()
while True:
    # sim.set_state(sim_state)
    for i in range(2000):
        # sim.data.ctrl[:] = -1 * np.ones(11) * 2
        # sim.step()
        viewer.render()
