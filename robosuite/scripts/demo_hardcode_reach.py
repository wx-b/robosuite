"""
Test script for trying to hard code a reaching policy
"""

import argparse
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper


if __name__ == "__main__":
    env = robosuite.make("SawyerPickPlaceCan", 
        has_renderer=True, 
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
        gripper_visualization=True,
    )

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    ### TODO: try to feed in delta quaternions that are offset from the canonical starting pose of the sawyer ###
    ### TODO: we need to grab the "correct" actions, so maybe we shouldn't use the IKWrapper but decompose it and use base jpos env ###
    ### TODO: support for different kinds of experts easily (manhattan distance, direct distance) and evaluate success ###

    while True:
        obs = env.reset()
        for _ in range(500):
            # delta_x = 0.005 * np.random.uniform(size=3, low=-1., high=1.)
            delta_x = obs['Can0_pos'] - obs["eef_pos"]
            delta_x /= np.linalg.norm(delta_x)
            delta_x *= 0.005
            delta_quat = np.array([0., 0., 0., 1.]) # unit quaternion for (x, y, z, w)
            action = np.concatenate([delta_x, delta_quat, [-1.]])
            obs, _, _, _ = env.step(action)
            env.render()
