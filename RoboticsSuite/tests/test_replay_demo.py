from MujocoManip import *
import numpy as np
import time
import datetime
import pickle
from PIL import Image
from IPython import embed

from MujocoManip.models import *
from MujocoManip.wrappers import DataCollector

from MujocoManip.miscellaneous.spacenavigator import SpaceNavigator
from MujocoManip.miscellaneous.ik_controller import IKController

if __name__ == "__main__":

    # a test case: do completely random actions at each time step
    env = make(
        "SawyerStackEnv",
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
    )

    obs = env.reset()

    dof = env.dof
    print("action space", env.action_space)
    print("Obs: {}".format(len(obs)))
    print("DOF: {}".format(dof))
    env.render()

    episode_data = pickle.load(open("SawyerStackEnv_20180604212825.pkl", "rb"))

    for i in range(1000):
        env.sim.set_state(episode_data[i]["state"])
        env.sim.forward()
        # obs, reward, done, info = env.step(action)
        env.render()
