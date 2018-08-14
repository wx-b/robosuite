"""
Testing script for playing back some data collected from the DataCollector wrapper.
"""

from MujocoManip import SawyerStackEnv, make
import numpy as np
import time
import os
from PIL import Image
from glob import glob
from IPython import embed

from MujocoManip.models import *
from MujocoManip import DataCollector


def collect_random_data(env, timesteps=1000):
    obs = env.reset()
    dof = env.dof()

    # rotate the gripper so we can see it easily
    env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    for t in range(timesteps):
        action = 0.5 * np.random.randn(dof)
        obs, reward, done, info = env.step(action)
        env.render()
        if (t + 1) % 100 == 0:
            print(t + 1)


def playback_data(env, ep_dir):
    """
    Playback data from an episode. 

    :param ep_dir: The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


if __name__ == "__main__":

    ### TODO: Handle flushing of remaining data when script terminates (i.e. freq is 1000, but we had 500 timesteps) ###

    env = make("SawyerStackEnv", ignore_done=True, use_camera_obs=False)
    direct = "./"
    env = DataCollector(env, direct)

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    dof = env.dof
    print("action space", env.action_space)
    print("DOF: {}".format(dof))
    env.render()

    print("Collecting some random data...")
    # collect some data
    collect_random_data(env, timesteps=2000)

    # playback some data
    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")

    direct = env.ep_directory
    playback_data(env, direct)
