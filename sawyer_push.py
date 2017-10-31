#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
import numpy as np
import os

model = load_model_from_path("robots/sawyer/arena.xml")

# import pdb; pdb.set_trace()

sim = MjSim(model)

viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0
viewer.vopt.geomgroup[1] = 1

viewer.cam.fixedcamid = 0
viewer.cam.type = const.CAMERA_FIXED
viewer.cam.azimuth = 179.7749999999999
viewer.cam.distance = 3.825077470729921
viewer.cam.elevation = -21.824999999999992
viewer.cam.lookat[:][0] = 0.09691817
viewer.cam.lookat[:][1] = 0.00164106
viewer.cam.lookat[:][2] = -0.30996464

sim_state = sim.get_state()
# viewer.render()

# Note: sim.data.ctrl contains the actuator values that we can control
while True:
    sim.set_state(sim_state)

    for i in range(500):
        sim.data.ctrl[:] = np.random.rand(7) * 3
        print(sim.data.ctrl[:])

        sim.step()
        viewer.render()
