"""
This file contains the base wrapper class for Mujoco environments.
Wrappers are useful for data collection and logging. Highly recommended.
"""

from MujocoManip.environment.base import MujocoEnv

class Wrapper(MujocoEnv):
    env = None

    def __init__(self, env):
        self.env = env
        self.physics = self.env.physics
        self.done = False
        self.horizon = self.env.horizon

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                env = env.env
            else:
                break

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def action_space(self):
        return self.env.action_space()

    def dof(self):
        return self.env.dof()

    @property
    def unwrapped(self):
        return self.env.unwrapped



