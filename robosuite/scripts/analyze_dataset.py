
import argparse
import h5py
import random
import os
import numpy as np
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite import make
from robosuite.utils.ffmpeg_gif import save_gif


def gen_gifs(args, f, env, target_length=None):
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

        # load the flattened mujoco states
        states = f["data/{}/states".format(key)].value
        seq_length = steps2length(states.shape[0])
        if target_length is not None and np.abs(seq_length - target_length) > 1:
            continue

        # force the sequence of internal mujoco states one by one
        frames = []
        for i, state in enumerate(states):
            if i % args.skip_frame == 0:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                obs = env._get_observation()
                frame = obs["image"][::-1]
                frames.append(frame)

        frames = np.stack(frames, axis=0)
        fig_file_name = os.path.join(args.output_path, "seq_{}".format(key))
        if target_length is not None:
            fig_file_name += "_len_{}".format(target_length)
        save_gif(fig_file_name + ".gif", frames, fps=15)
        if target_length is not None:
            return True
    return False if target_length is not None else True


def steps2length(steps):
    return steps/(10*15)


def plot_stats(args, f):
    # plot histogram of lengths
    seaborn.set()   # plot style
    demos = list(f["data"].keys())
    lengths = []
    for key in tqdm.tqdm(demos):
        states = f["data/{}/states".format(key)].value
        lengths.append(steps2length(states.shape[0]))
    lengths = np.stack(lengths)
    fig = plt.figure()
    plt.hist(lengths, bins=30)
    plt.xlabel("Approx. Demo Length [sec]")
    # plt.title("Peg Assembly")
    # plt.xlim(5, 75)
    # plt.ylim(0, 165)
    fig.savefig(os.path.join(args.output_path, "length_hist.png"))
    plt.close()


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
    parser.add_argument("--gif_target_length", type=int, default=-1)
    args = parser.parse_args()

    if args.gif_target_length == -1:
       args.gif_target_length = None

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
        success = gen_gifs(args, f, env)
        if not success:
            raise ValueError("Could not plot gifs successfully!")

    if args.plot_stats:
        plot_stats(args, f)

    print("Done")
