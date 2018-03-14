"""
This file contains a wrapper for saving simulation states.
This is useful for collecting demonstrations.
"""

from MujocoManip.wrappers import Wrapper
import os
import time

class DataCollector(Wrapper):

    def __init__(self, env, directory):
        """
        :param env: The environment to monitor.
        :param directory: Where to store collected data.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states
        self.states = [] 

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = 1

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = 1000000

        if not os.path.exists(directory):
            print("DataCollector: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # TODO: save xml, create new subdirectory for logging, etc...

        # flush any data left over from the previous episode
        if self.ep_directory is not None:
            self._flush()

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split('.')
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))

        # save the model xml
        xml_path = os.path.join(self.ep_directory, 'model.xml')
        self.env.task.save_model(xml_path)

        # timesteps in current episode
        self.t = 0

    def _flush(self):
        """
        Method to flush internal state to disk. 
        """
        t1, t2 = str(time.time()).split('.')
        state_path = os.path.join(self.directory, "state_{}_{}.npz".format(t1, t2))
        np.savez(state_path, states=*self.states)
        self.states = []

    def reset(self):
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):

        ret = super().step(action)
        self.t += 1

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.physics.state()
            self.states.append(state)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret




