"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import time
import argparse
import signal
import numpy as np

import RoboticsSuite
from RoboticsSuite.controllers.spacemouse import SpaceMouse
from RoboticsSuite.controllers.sawyer_ik_controller import SawyerIKController
from RoboticsSuite import DataCollectionWrapper

def collect_human_trajectory(env, space_mouse, ik_controller):
    """
    Use the SpaceNav 3D mouse to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env: environment to control
        space_mouse (instance of SpaceMouse class): to receive controls from the SpaceNav
    """

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
    ik_controller.sync_state()

    env.viewer.set_camera(camera_id=2)
    env.render()

    # episode terminates on a spacenav reset input or if task is completed
    reset = False
    space_mouse.start_control()
    while not (reset or env._check_success()):
        state = space_mouse.get_controller_state()
        dpos, rotation, grasp, reset = state["dpos"], state["rotation"], state["grasp"], state["reset"]
        velocities = ik_controller.get_control(dpos=dpos, rotation=rotation)
        grasp = grasp - 1. # map 0 -> -1, 1 -> 0 so that 0 is open, 1 is closed (halfway)
        action = np.concatenate([velocities, [grasp]])

        obs, reward, done, info = env.step(action)
        env.render()


def gather_demonstrations_as_pkl(directory, large=False):
    """
    Gathers the demonstrations saved in the current directory into a 
    single pkl file.

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        large (bool): If true, generates both a .pkl and a .bkl file. The 
            .pkl file is for indexing into the .bkl file that contains 
            all of the demonstrations. This allows for lazy loading of
            demonstrations, which is useful if there are a lot of them. 
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--directory", type=str, 
        default="/tmp/{}".format(str(time.time()).replace(".", "_")))
    args = parser.parse_args()

    # create original environment
    env = RoboticsSuite.make(
        args.environment,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        control_freq=100,
    )
    data_directory = args.directory

    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, data_directory)

    # function to return robot joint angles
    def robot_jpos_getter():
        return np.array(env._joint_positions)

    # initialize space_mouse controller
    space_mouse = SpaceMouse()

    # initialize IK controller
    ik_controller = SawyerIKController(
        bullet_data_path=os.path.join(RoboticsSuite.models.assets_root, "bullet_data"),
        robot_jpos_getter=robot_jpos_getter,
    )

    ### TODO: signal handling ###

    while True:
        collect_human_trajectory(env, space_mouse, ik_controller)



