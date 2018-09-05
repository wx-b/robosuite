"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a pkl file (and optionally
a bkl file).
"""
import os
import pickle
import argparse
import random
import numpy as np

import RoboticsSuite
from RoboticsSuite.utils.mjcf_utils import postprocess_model_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join(
            RoboticsSuite.models.assets_root, "demonstrations/sawyer-lift.pkl"
        ),
    )
    parser.add_argument("--with-actions", dest='actions', action='store_true')
    args = parser.parse_args()

    env = RoboticsSuite.make(
        args.environment,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100,
    )

    with open(args.file, "rb") as f:
        d = pickle.load(f)
        # a hacky way to tell if this is a small pickle files with indices into
        # a bigger one, or if this is a pickle file with demos in it.
        if d[0] == 0:
            big = open(sys.argv[1].replace(".pkl", ".bkl"), "rb")
        else:
            big = None

    while True:
        print("Playing back random episode... (press ESC to quit)")
        # select an episode randomly
        t = random.choice(d)
        if type(t) == int:
            big.seek(t)
            t = pickle.load(big)

        env.reset()
        xml = postprocess_model_xml(t["model.xml"])
        env.reset_from_xml_string(xml)
        env.viewer.set_camera(0)

        if args.actions:
            # reset to the first state we recorded
            state = t["states"][0]
            if isinstance(state, tuple):
                state = state[0]
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            # Replay stored actions in open loop - this runs through actual mujoco simulation.
            for ai in t["action_infos"]:
                velocities, gripper_acts = ai["joint_velocities"], ai["gripper_actuation"]
                a = np.concatenate([velocities, gripper_acts])
                env.step(a)
                env.render()
        else:
            # force the sequence of internal mujoco states one by one
            for state in t["states"]:
                if isinstance(state, tuple):
                    state = state[0]
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()
