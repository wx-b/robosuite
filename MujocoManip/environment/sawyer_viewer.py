import MujocoManip as MM
from MujocoManip.miscellaneous import MujocoPyRenderer
from MujocoManip.miscellaneous.utils import *

from mujoco_py import MjSim
import numpy as np


class SawyerViewer(object):
    def __init__(self, env_name, **kwargs):
        """
        Note that you can pass any keyword arguments to pass to the environment constructor. 

        :param env_name: the name of the environment to visualize
        """

        # make the environment
        self._env = MM.make(env_name, **kwargs)

        # grab the xml description of the environment
        self._model = self._env.task.get_model(mode='mujoco_py')

        # construct an MjSim instance
        self._sim = MjSim(self._model)

        ### TODO: change this, or introduce randomness in self._env.task construction... ###

        # save initial state of simulation to restore when calling reset
        self.initial_state = self._sim.get_state()

        # construct the MjViewer
        self._renderer = MujocoPyRenderer(self._sim)

        # initialize some timing variables
        self.initialize_time(self._env.control_freq)

        if self._env.use_eef_ctrl:
            # mocap setup
            self._setup_mocap()

        # save these super special indices
        self._ref_joint_pos_indexes = [self._model.get_joint_qpos_addr('right_j{}'.format(x)) for x in range(7)]
        self._ref_joint_vel_indexes = [self._model.get_joint_qvel_addr('right_j{}'.format(x)) for x in range(7)]
        if self._env.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [self._model.actuator_name2id(actuator) for actuator in self._model.actuator_names 
                                                                                              if actuator.startswith("gripper")]

        self.reset()

    def initialize_time(self, control_freq):
        """
            Initialize the time constants used for simulation
        """
        self.cur_time = 0
        self.model_timestep = self._model.opt.timestep
        self.control_freq = control_freq
        self.control_timestep = 1. / control_freq

    def _setup_mocap(self):
        mjpy_reset_mocap_welds(self._sim)
        self._sim.forward()

        # Move end effector into position.
        gripper_target = self._sim.data.get_body_xpos('right_hand')
        gripper_rotation = self._sim.data.get_body_xquat('right_hand')

        self._sim.data.set_mocap_pos('mocap', gripper_target)
        self._sim.data.set_mocap_quat('mocap', gripper_rotation)
        for _ in range(10):
            self._sim.step()

    def render(self):
        self._renderer.render()

    def reset(self):
        # restore old simulation state
        self._sim.set_state(self.initial_state)

        # restore joint positions
        self._sim.data.qpos[self._ref_joint_pos_indexes] = self._env.mujoco_robot.rest_pos
        if self._env.has_gripper:
            self._sim.data.qpos[self._ref_joint_gripper_actuator_indexes] = self._env.gripper.rest_pos
        self._sim.forward()

        self.cur_time = 0
        self.t = 0


    def step(self, action):
        self.t += 1
        self._pre_action(action)
        end_time = self.cur_time + self.control_timestep
        while self.cur_time < end_time:
            self._sim.step()
            self.cur_time += self.model_timestep

    def _pre_action(self, action):

        if self._env.use_eef_ctrl:
            assert len(action) == 9
            action = action.copy()  # ensure that we don't change the action outside of this scope
            # pos_ctrl, gripper_ctrl = action[:3], action[3:]

            pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:]

            # pos_ctrl *= 0.05  # limit maximum change in position
            # rot_ctrl = [0., -1./np.sqrt(2.), -1./np.sqrt(2.), 0.]

            # rot_ctrl = [0., 0., 1., 0.]  # (w, x, y, z) # fixed rotation of the end effector, expressed as a quaternion
            # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            # Apply action to simulation.
            mjpy_ctrl_set_action(self.physics, action)
            mjpy_mocap_set_action(self.physics, action)

            # gravity compensation
            self._sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self._sim.data.qfrc_bias[self._ref_joint_vel_indexes]
        
        else:
            action = np.clip(action, -1, 1)    
            if self._env.has_gripper:
                arm_action = action[:self._env.mujoco_robot.dof()]
                gripper_action_in = action[self._env.mujoco_robot.dof():self._env.mujoco_robot.dof()+self._env.gripper.dof()]
                gripper_action_actual = self._env.gripper.format_action(gripper_action_in)
                action = np.concatenate([arm_action, gripper_action_actual])

            # rescale normalized action to control ranges
            ctrl_range = self._sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:,0])
            weight = 0.5 * (ctrl_range[:,1] - ctrl_range[:,0])
            applied_action = bias + weight * action
            self._sim.data.ctrl[:] = applied_action

            # gravity compensation
            self._sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self._sim.data.qfrc_bias[self._ref_joint_vel_indexes]


