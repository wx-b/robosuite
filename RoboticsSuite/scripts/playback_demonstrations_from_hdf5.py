"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.
"""
import os
import h5py
import argparse
import random
import numpy as np

import RoboticsSuite
from RoboticsSuite.utils.mjcf_utils import postprocess_model_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=os.path.join(
            RoboticsSuite.models.assets_root, "demonstrations/Test"
        ),
    )
    parser.add_argument("--with-actions", dest='actions', action='store_true')
    args = parser.parse_args()


    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")  
    env_name = f["data"].attrs["env"]

    env = RoboticsSuite.make(
        env_name,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True:
        print("Playing back random episode... (press ESC to quit)")
        
        # # select an episode randomly
        # ep = random.choice(demos)
        ep = demos[0]

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["data/{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)].value

        if args.actions:
            # reset to the first state we recorded
            state = states[0]
            if isinstance(state, tuple):
                state = state[0]
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            # load actions
            actions = f["data/{}/actions".format(ep)].value
            joint_velocities = f["data/{}/joint_velocities".format(ep)].value
            gripper_actuations = f["data/{}/gripper_actuations".format(ep)].value

            # Replay stored actions in open loop - this runs through actual mujoco simulation.
            # for velocities, gripper_acts in zip(joint_velocities, gripper_actuations):
            #     a = np.concatenate([velocities, gripper_acts])
            #     env.step(a)
            #     env.render()

            for a in actions:
                env.step(a)
                env.render()


            # print some environment info for consistency's sake
            ob = env._get_observation()
            print("robot-state: {}".format(ob["robot-state"]))
            print("object-state: {}".format(ob["object-state"]))

        else:
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()

            # print some environment info for consistency's sake
            ob = env._get_observation()
            print("robot-state: {}".format(ob["robot-state"]))
            print("object-state: {}".format(ob["object-state"]))

    f.close()
