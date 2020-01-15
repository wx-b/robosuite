
import argparse
import h5py
import random
import os
import numpy as np
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
#import seaborn

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite import make
from robosuite.utils.ffmpeg_gif import save_gif
from robosuite.utils.transform_utils import mat2pose, quat_multiply, quat_conjugate

from robosuite.environments.base import make_invkin_env

def render(args, f, env):
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

        states = f["data/{}/states".format(key)].value
        d_pos = f["data/{}/right_dpos".format(key)].value
        d_quat = f["data/{}/right_dquat".format(key)].value
        gripper_actuation = f["data/{}/gripper_actuations".format(key)].value

        n_steps = states.shape[0]
        if args.target_length is not None and n_steps > args.target_length:
            continue

        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # force the sequence of internal mujoco states one by one
        frames = []
        for i, state in enumerate(states):
            if i % 50 == 0:
                t1 = time.time()
                obs = env._get_observation()
                frame = obs["image"][::-1]
                frames.append(frame)
                print('getob time', time.time() - t1)

            t1 = time.time()

            env.step(np.concatenate((d_pos[i], d_quat[i], gripper_actuation[i]), axis=-1))
            print('steptime', time.time() - t1)

        frames = np.stack(frames, axis=0)
        actions = np.concatenate((d_pos, d_quat, gripper_actuation), axis=-1)

        pad_mask = np.ones((n_steps,)) if n_steps == args.target_length \
                        else np.concatenate((np.ones((n_steps,)), np.zeros((args.target_length - n_steps,))))

        h5_path = os.path.join(args.output_path, "seq_{}.h5".format(key))
        with h5py.File(h5_path, 'w') as F:
            F['traj_per_file'] = 1
            F["traj0/images"] = frames
            F["traj0/actions"] = actions
            F["traj0/states"] = states
            F["traj0/pad_mask"] = pad_mask
            # F["traj0/joint_velocities"] = joint_velocities

        xml_path = os.path.join(args.output_path, "seq_{}.xml".format(key))
        env.model.save_model(xml_path)

        fig_file_name = os.path.join(args.output_path, "seq_{}".format(key))
        save_gif(fig_file_name + ".gif", frames, fps=15)


def steps2length(steps):
    return steps/(10*15)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_folder", type=str,
                        default=os.path.join(robosuite.models.assets_root, "demonstrations/SawyerNutAssembly"))
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--skip_frame", type=int, default=0)
    parser.add_argument("--gen_dataset", type=bool, default=False)
    parser.add_argument("--plot_stats", type=bool, default=False)
    parser.add_argument("--target_length", type=int, default=-1)
    args = parser.parse_args()

    if args.target_length == -1:
       args.target_length = None

    # initialize an environment with offscreen renderer
    demo_file = os.path.join(args.demo_folder, "demo.hdf5")
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env = make_invkin_env(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_height=args.height,
        camera_width=args.width,
        control_freq=100,  ###### important !!!
    )
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.gen_dataset:
        render(args, f, env)

    print("Done")
    f.close()



