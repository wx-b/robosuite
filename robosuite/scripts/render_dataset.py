
import argparse
import h5py
import random
import os
import numpy as np
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import seaborn

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite import make
from robosuite.utils.ffmpeg_gif import save_gif
from robosuite.utils.transform_utils import mat2pose, quat_multiply, quat_conjugate


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

        # load + subsample data
        states, _ = FixedFreqSubsampler(n_skip=args.skip_frame)(f["data/{}/states".format(key)].value)
        # d_pos, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=SumAggregator()) \
        #             (f["data/{}/right_dpos".format(key)].value, aggregate=True)
        # d_quat, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=QuaternionAggregator()) \
        #              (f["data/{}/right_dquat".format(key)].value, aggregate=True)
        gripper_actuation, _ = FixedFreqSubsampler(n_skip=args.skip_frame)(f["data/{}/gripper_actuations".format(key)].value)
        joint_velocities, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=SumAggregator()) \
                                (f["data/{}/joint_velocities".format(key)].value, aggregate=True)

        n_steps = states.shape[0]
        if args.target_length is not None and n_steps > args.target_length:
            print('traj {} too long'.format(key))
            continue

        # force the sequence of internal mujoco states one by one
        frames = []
        joint_pos = []
        gripper_pos = []
        object_pose = []

        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obs = env._get_observation()

            if args.store_images:
                frame = obs["image"][::-1]
                frames.append(frame)

            joint_pos.append(obs['joint_pos'])
            gripper_pos.append(obs["gripper_qpos"])
            object_pose.append(np.concatenate([obs[env.obj_to_use + "_pos"], obs[env.obj_to_use + "_quat"]], 0))

            print(i)

            if i == 10:
                break

        pad_mask = np.ones((n_steps,)) if n_steps == args.target_length \
                        else np.concatenate((np.ones((n_steps,)), np.zeros((args.target_length - n_steps,))))

        h5_path = os.path.join(args.output_path, "seq_{}.h5".format(key))

        joint_object_pose = np.concatenate([np.stack(joint_pos, 0), np.stack(gripper_pos, 0), np.stack(object_pose, 0)], -1)

        with h5py.File(h5_path, 'w') as F:
            F['traj_per_file'] = 1
            if args.store_images:
                frames = np.stack(frames, axis=0)
                F["traj0/images"] = frames
            F["traj0/actions"] = joint_velocities
            F["traj0/full_states"] = states
            F["traj0/states"] = joint_object_pose
            F["traj0/pad_mask"] = pad_mask
            F["traj0/joint_velocities"] = joint_velocities
            F["traj0/model_file"] = "seq_{}.xml".format(key)
            F["env_name"] = f["data"].attrs["env"]

        import pdb; pdb.set_trace()

        xml_path = os.path.join(args.output_path, "seq_{}.xml".format(key))
        env.model.save_model(xml_path)

        if args.store_images:
            fig_file_name = os.path.join(args.output_path, "seq_{}".format(key))
            save_gif(fig_file_name + ".gif", frames, fps=15)

        print('saved traj', key)


def steps2length(steps):
    return steps/(10*15)


def plot_stats(args, f):
    # plot histogram of lengths
    demos = list(f["data"].keys())
    lengths = []
    for key in tqdm.tqdm(demos):
        states = f["data/{}/states".format(key)].value
        lengths.append(states.shape[0])
    lengths = np.stack(lengths)
    fig = plt.figure()
    plt.hist(lengths, bins=30)
    plt.xlabel("Approx. Demo Length [sec]")
    # plt.title("Peg Assembly")
    # plt.xlim(5, 75)
    # plt.ylim(0, 165)
    fig.savefig(os.path.join(args.output_path, "length_hist.png"))
    plt.close()


class DataSubsampler:
    def __init__(self, aggregator):
        self._aggregator = aggregator

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class FixedFreqSubsampler(DataSubsampler):
    """Subsamples input array's first dimension by skipping given number of frames."""
    def __init__(self, n_skip, aggregator=None):
        super().__init__(aggregator)
        self._n_skip = n_skip

    def __call__(self, val, idxs=None, aggregate=False):
        """Subsamples with idxs if given, aggregates with aggregator if aggregate=True."""
        if self._n_skip == 0:
            return val, None

        if idxs is None:
            seq_len = val.shape[0]
            idxs = np.arange(0, seq_len - 1, self._n_skip + 1)

        if aggregate:
            assert self._aggregator is not None     # no aggregator given!
            return self._aggregator(val, idxs), idxs
        else:
            return val[idxs], idxs


class Aggregator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class SumAggregator(Aggregator):
    def __call__(self, val, idxs):
        return np.add.reduceat(val, idxs, axis=0)


class QuaternionAggregator(Aggregator):
    def __call__(self, val, idxs):
        # quaternions get aggregated by multiplying in order
        aggregated = [val[0]]
        for i in range(len(idxs)-1):
            idx, next_idx = idxs[i], idxs[i+1]
            agg_val = val[idx]
            for ii in range(idx+1, next_idx):
                agg_val = self.quaternion_multiply(agg_val, val[ii])
            aggregated.append(agg_val)
        return np.asarray(aggregated)

    @staticmethod
    def quaternion_multiply(Q0, Q1):
        w0, x0, y0, z0 = Q0
        w1, x1, y1, z1 = Q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


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
    parser.add_argument("--store_images", type=int, default=0)
    args = parser.parse_args()

    if args.target_length == -1:
       args.target_length = None

    # initialize an environment with offscreen renderer
    demo_file = os.path.join(args.demo_folder, "demo.hdf5")
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env = make(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=args.store_images,
        use_object_obs=True,
        camera_height=args.height,
        camera_width=args.width,
    )
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.gen_dataset:
        render(args, f, env)

    if args.plot_stats:
        plot_stats(args, f)

    print("Done")
    f.close()



