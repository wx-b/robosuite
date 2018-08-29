"""
This file contains a wrapper for sampling environment states
from a set of demonstrations on every reset. The main use case is for 
altering the start state distribution of training episodes for 
learning RL policies.
"""

import random
import os
import pickle
import time
from collections import deque
import numpy as np

from RoboticsSuite.utils import postprocess_model_xml
from RoboticsSuite.wrappers import Wrapper


class DemoSamplerWrapper(Wrapper):
    env = None

    def __init__(
        self,
        env,
        file_path,
        need_xml=False,
        preload=False,
        num_traj=-1,
        sampling_schemes=["uniform", "random"],
        scheme_ratios=[0.9, 0.1],
        open_loop_increment_freq=100,
        open_loop_initial_window_width=25,
        open_loop_window_increment=25,
    ):
        """
        Initializes a wrapper that provides support for resetting the environment
        state to one from a demonstration. It also supports curriculums for
        altering how often to sample from demonstration vs. sampling a reset
        state from the environment.

        Args:
            env (MujocoEnv instance): The environment to wrap.

            file_path (string): The path to the demonstrations, in pkl format.
            
            need_xml (bool): If True, the mujoco model needs to be reloaded when
                sampling a state from a demonstration. This could be because every
                demonstration was taken under varied object properties, for example.

            preload (bool): If True, fetch all demonstrations into memory at the
                beginning. Otherwise, load demonstrations as they are needed lazily.

            num_traj (int): If provided, subsample @number demonstrations from the 
                provided set of demonstrations instead of using all of them.

            sampling_schemes (list of strings): A list of sampling schemes
                to be used. The following strings are valid schemes:

                    "random" : sample a reset state directly from the wrapped environment

                    "uniform" : sample a state from a demonstration uniformly at random

                    "forward" : sample a state from a window that grows progressively from
                        the start of demonstrations

                    "reverse" : sample a state from a window that grows progressively from
                        the end of demonstrations

            scheme_ratios (list of floats): A list of probability values to
                assign to each member of @sampling_schemes. Must be non-negative and
                sum to 1.

            open_loop_increment_freq (int): How frequently to increase
                the window size in open loop schemes ("forward" and "reverse"). The
                window size will increase by @open_loop_window_increment every
                @open_loop_increment_freq samples. Only samples that are generated
                by open loop schemes contribute to this count.

            open_loop_initial_window_width (int): The width of the initial sampling
                window, in terms of number of demonstration time steps, for
                open loop schemes.

            open_loop_window_increment (int): The window size will increase by
                @open_loop_window_increment every @open_loop_increment_freq samples.
                This number is in terms of number of demonstration time steps.
        """

        super().__init__(env)

        """ Load the demo file...
            If it's a .pkl file with a list of objects, then proceed as usual.
            If it's a .pkl file with a list of numbers, open the corresponding file with all of them;
                if we want to preload the samples, then read the big file onto a list, and
                    then pretend that we just read a normal .pkl file
                if we want to lazy load from the filesystem, simply keep self.demo_big_file pointing
                    to the bigfile and self.demo_data as a list of numbers
        """
        with open(file_path, "rb") as f:
            self.demo_data = pickle.load(f)
            is_bkl = self.demo_data[0] == 0
            if num_traj > 0:
                random.seed(3141)
                self.demo_data = random.sample(self.demo_data, num_traj)
                random.seed(time.time() * 1000 * ord(os.urandom(1)))
                print("Subsampled:", self.demo_data)
            if is_bkl:
                self.demo_big_file = open(file_path.replace(".pkl", ".bkl"), "rb")
                if preload:
                    print("Preloading...")
                    data = []
                    for ofs in self.demo_data:
                        self.demo_big_file.seek(ofs)
                        data.append(pickle.load(self.demo_big_file))
                    self.demo_data = data
                    self.demo_big_file.close()
                    self.demo_big_file = None
                    print("Preloaded!")
            else:
                print("Not a bkl!")
                self.demo_big_file = None

        self.need_xml = need_xml
        self.demo_sampled = 0

        self.sample_method_dict = {
            "random": "_random_sample",
            "uniform": "_uniform_sample",
            "forward": "_forward_sample_open_loop",
            "reverse": "_reverse_sample_open_loop",
        }

        self.sampling_schemes = sampling_schemes
        self.scheme_ratios = np.asarray(scheme_ratios)

        # make sure the list of schemes is valid
        schemes = self.sample_method_dict.keys()
        assert np.all([(s in schemes) for s in self.sampling_schemes])

        # make sure the distribution is the correct size
        assert len(self.sampling_schemes) == len(self.scheme_ratios)

        # make sure the distribution lies in the probability simplex
        assert np.all(self.scheme_ratios > 0.)
        assert sum(self.scheme_ratios) == 1.0

        # open loop configuration
        self.open_loop_increment_freq = open_loop_increment_freq
        self.open_loop_window_increment = open_loop_window_increment

        # keep track of window size
        self.open_loop_window_size = open_loop_initial_window_width

    def reset(self):
        """
        Logic for sampling a state from the demonstration and resetting
        the simulation to that state. 
        """
        state = self.sample()
        if state is None:
            # None indicates that a normal env reset should occur
            return self.env.reset()
        else:
            if self.need_xml:
                # reset the simulation from the model if necessary
                state, xml = state
                self.env.reset_from_xml_string(xml)

            if isinstance(state, tuple):
                state = state[0]

            # force simulator state to one from the demo
            self.sim.set_state_from_flattened(state)
            self.sim.forward()

            return self.env._get_observation()

    def sample(self):
        """
        This is the core sampling method. Samples a state from a
        demonstration, in accordance with the configuration.
        """

        # chooses a sampling scheme randomly based on the mixing ratios
        seed = random.uniform(0, 1)
        ratio = np.cumsum(self.scheme_ratios)
        ratio = ratio > seed
        for i, v in enumerate(ratio):
            if v:
                break

        sample_method = getattr(self, self.sample_method_dict[self.sampling_schemes[i]])
        return sample_method()

    def _random_sample(self):
        """
        Sampling method.

        Return None to indicate that the state should be sampled directly
        from the environment.
        """
        return None

    def _uniform_sample(self):
        """
        Sampling method.

        First uniformly sample a demonstration from the set of demonstrations.
        Then uniformly sample a state from the selected demonstration.
        """
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        state = random.choice(episode["states"])

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml
        return state

    def _reverse_sample_open_loop(self):
        """
        Sampling method.

        Open loop reverse sampling from demonstrations. Starts by 
        sampling from states near the end of the demonstrations.
        Increases the window backwards as the number of calls to
        this sampling method increases at a fixed rate.
        """

        # sample a random demonstration
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)

        # sample uniformly in a window that grows backwards from the end of the demos
        eps_len = len(episode["states"])
        index = np.random.randint(max(eps_len - self.open_loop_window_size, 0), eps_len)
        state = episode["states"][index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state

    def _forward_sample_open_loop(self):
        """
        Sampling method.

        Open loop forward sampling from demonstrations. Starts by
        sampling from states near the beginning of the demonstrations.
        Increases the window forwards as the number of calls to
        this sampling method increases at a fixed rate.
        """

        # sample a random demonstration
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)

        # sample uniformly in a window that grows forwards from the beginning of the demos
        eps_len = len(episode["states"])
        index = np.random.randint(0, min(self.open_loop_window_size, eps_len))
        state = episode["states"][index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state
