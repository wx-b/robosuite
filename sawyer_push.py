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

t = 0

while True:
    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
