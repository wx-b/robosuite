"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import time
import argparse
import pickle
from glob import glob
import numpy as np

import RoboticsSuite
import RoboticsSuite.utils.transform_utils as T
from RoboticsSuite.wrappers import IKWrapper
from RoboticsSuite.wrappers import DataCollectionWrapper


def collect_human_trajectory(env, device):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env: environment to control
        device (instance of Device class): to receive controls from the device
    """

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    env.viewer.set_camera(camera_id=2)
    env.render()

    # episode terminates on a spacenav reset input or if task is completed
    reset = False
    device.start_control()
    while not (reset or env._check_success()):
        state = device.get_controller_state()
        dpos, rotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["grasp"],
            state["reset"],
        )

        # convert into a suitable end effector action for the environment
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)  # relative rotation of desired from current
        dquat = T.mat2quat(drotation)
        grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
        action = np.concatenate([dpos, dquat, [grasp]])

        obs, reward, done, info = env.step(action)
        env.render()

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_pkl(directory, out_dir, large=False):
    """
    Gathers the demonstrations saved in @directory into a
    single pkl file.

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the pkl file.
        large (bool): If true, generates both a .pkl and a .bkl file. The
            .pkl file is for indexing into the .bkl file that contains
            all of the demonstrations. This allows for lazy loading of
            demonstrations, which is useful if there are a lot of them.
    """
    pickle_path = os.path.join(out_dir, "demo.pkl")
    if large:
        big = open(pickle_path.replace(".pkl", ".bkl"), "wb")
        ofs = [0]

    all_data = []
    for ep_directory in os.listdir(directory):
        # collect episode data into dictionary
        ep_data = {}
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            ep_data["model.xml"] = f.read()
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        action_infos = []
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file)
            # Note how we index the actions here. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action.
            for s, ai in zip(dic["states"][:-1], dic["action_infos"][1:]):
                states.append(s)
                action_infos.append(ai)
        ep_data["states"] = states
        ep_data["action_infos"] = action_infos
        if len(states) == 0:
            continue

        if large:
            # write episode to large pickle file as serialized string
            # and remember its offset in the small pickle file
            raw = pickle.dumps(ep_data)
            delta_ofs = big.write(raw)
            ofs.append(ofs[-1] + delta_ofs)
        else:
            # collect episode in global list
            all_data.append(ep_data)

    small = open(pickle_path, "wb")
    if large:
        # dump offsets to pickle
        pickle.dump(ofs[:-1], small)
        big.close()
    else:
        # dump actual data to pickle
        pickle.dump(all_data, small)
    small.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(RoboticsSuite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--large", type=bool, default=False)
    args = parser.parse_args()

    # create original environment
    env = RoboticsSuite.make(
        args.environment,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        control_freq=100,
        gripper_visualization=True,
    )

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from RoboticsSuite.devices import Keyboard

        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from RoboticsSuite.devices import SpaceMouse

        device = SpaceMouse()
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device)
        gather_demonstrations_as_pkl(tmp_directory, args.directory, args.large)
