#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

# model = load_model_from_path("robots/sawyer/main.xml")
model = load_model_from_path("robots/sawyer/sawyer_urdf.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0
viewer.vopt.geomgroup[1] = 1

sim_state = sim.get_state()


# Note: sim.data.ctrl contains the actuator values that we can control
while True:
    sim.set_state(sim_state)

    for i in range(1000):
        if i < 150:
            sim.data.ctrl[:] = [float(i) / 1000 for _ in range(7)]
        else:
            sim.data.ctrl[:] = [1.0 for _ in range(7)]
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:


