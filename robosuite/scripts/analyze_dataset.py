
import argparse
import h5py
import random
import os
import numpy as np
import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite import make
from robosuite.utils.ffmpeg_gif import save_gif


def gen_gifs(args, f, env):
    demos = list(f["data"].keys())
    for key in tqdm.tqdm(demos):
        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["data/{}".format(key)].attrs["model_file"]
        model_path = os.path.join(args.demo_folder, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(key)].value

        # load the initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # load the actions and play them back open-loop
        jvels = f["data/{}/joint_velocities".format(key)].value
        grip_acts = f["data/{}/gripper_actuations".format(key)].value
        actions = np.concatenate([jvels, grip_acts], axis=1)
        num_actions = actions.shape[0]

        frames = []
        for j, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            frame = obs["image"][::-1]
            frames.append(frame)

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                assert (np.all(np.equal(states[j + 1], state_playback)))

        frames = np.stack(frames, axis=0)
        save_gif(os.path.join(args.output_path, "seq_{}.gif".format(key)), frames)


def plot_stats(args, f):
    # plot histogram of lengths
    demos = list(f["data"].keys())
    lengths = []
    for key in tqdm.tqdm(demos):
        states = f["data/{}/states".format(key)].value
        lengths.append(states.shape[0])
    lengths = np.stack(lengths)
    fig = plt.figure()
    plt.histogram(lengths)
    plt.savefig(os.path.join(args.output_path, "length_hist.png"), fig)
    fig.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_folder", type=str,
                        default=os.path.join(robosuite.models.assets_root, "demonstrations/SawyerNutAssembly"))
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--gen_gifs", type=bool, default=False)
    parser.add_argument("--plot_stats", type=bool, default=False)
    args = parser.parse_args()

    # initialize an environment with offscreen renderer
    demo_file = os.path.join(args.demo_folder, "demo.hdf5")
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env = make(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_height=args.height,
        camera_width=args.width,
    )

    if args.gen_gifs:
        gen_gifs(args, f, env)

    if args.plot_stats:
        plot_stats(args, f)

    print("Done")
