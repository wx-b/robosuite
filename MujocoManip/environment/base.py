import numpy as np
from mujoco_py import MjSim, MjViewer
from MujocoManip.miscellaneous import SimulationError, XMLError

class MujocoEnv(object):
    def __init__(self, debug=False, display=True, control_freq=100, horizon=1200, **kwargs):
        """
            Initialize a Mujoco Environment
            @debug when True saves a model file for inspection
            @display render the environment
            @controL_freq in Hz, how many control signals to receive in every second
            TODO(extension): What about control_freq = a + bN(0,1) to simulate imperfect timing
        """
        self.debug = debug
        self.model = self._load_model()
        self.initialize_time(control_freq)
        self.sim = MjSim(self.model)
        self.display = display
        if self.display:
            self.viewer = MjViewer(self.sim)
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.set_cam()
        self.done = False
        self.t = 0
        self.horizon = horizon

        # for key in kwargs:
        #     print('Warning: Parameter {} not recognized'.format(key))


    def initialize_time(self, control_freq):
        """
            Initialize the time constants used for simulation
        """
        self.cur_time = 0
        self.model_timestep = self.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError('xml model defined non-positive time step')
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError('control frequency {} is invalid'.format(control_freq))
        self.control_timestep = 1 / control_freq

    def set_cam(self):
        pass
        # self.viewer.cam.fixedcamid = 0
        # # viewer.cam.type = const.CAMERA_FIXED
        # self.viewer.cam.azimuth = 179.7749999999999
        # self.viewer.cam.distance = 3.825077470729921
        # self.viewer.cam.elevation = -21.824999999999992
        # self.viewer.cam.lookat[:][0] = 0.09691817
        # self.viewer.cam.lookat[:][1] = 0.00164106
        # self.viewer.cam.lookat[:][2] = -0.30996464


    def _load_model(self):
        pass

    def _get_reference(self):
        pass

    def _reset(self):
        self._reset_internal()
        return self._get_observation()

    def _reset_internal(self):
        self.sim.set_state(self.sim_state_initial)
        self.cur_time = 0
        self.t=0
        self.done = False

    def _get_observation(self):
        return []

    def _step(self, action):
        reward = 0
        info = None
        if not self.done:
            self.t += 1
            self._pre_action(action)
            end_time = self.cur_time + self.control_timestep
            while self.cur_time < end_time:
                self.sim.step()
                self.cur_time += self.model_timestep
            reward, done, info = self._post_action(action)
            return self._get_observation(), reward, done, info
        else:
            return self._get_observation(), 0, True, None

    def _pre_action(self, action):
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        self.done = self._check_done() or self.t >= self.horizon
        reward = self._reward(action)
        # TODO: how to manage info?
        return reward, self.done, {}

    def _check_done(self):
        return False

    def _reward(self, action):
        return 0

    def _render(self):
        if self.display:
            self.viewer.render()
