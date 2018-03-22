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

        self._get_reference()
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

    def _get_reference(self):
        # save these super special indices
        self._ref_joint_pos_indexes = [self._model.get_joint_qpos_addr('right_j{}'.format(x)) for x in range(7)]
        self._ref_joint_vel_indexes = [self._model.get_joint_qvel_addr('right_j{}'.format(x)) for x in range(7)]
        if self._env.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [self._model.actuator_name2id(actuator) for actuator in self._model.actuator_names 
                                                                                              if actuator.startswith("gripper")]
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

    def dof(self):
        if self.use_eef_ctrl:
            dof = 3
        else:
            dof = self.mujoco_robot.dof()
        if self.has_gripper:
            dof += self.gripper.dof()
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self._sim.data.get_body_xpos(name)
        rot_in_world = self._sim.data.get_body_xmat(name).reshape((3, 3))
        # # note we convert (w, x, y, z) quat to (x, y, z, w)
        # eef_rot_in_world = quat2mat(self.physics.named.data.xquat['right_hand'][[1, 2, 3, 0]])
        pose_in_world = make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self._sim.data.get_body_xpos('base')
        base_rot_in_world = self._sim.data.get_body_xmat('base').reshape((3, 3))
        # base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = pose_inv(base_pose_in_world)

        pose_in_base = pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        self._sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self._sim.forward()

    @property
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name('right_l6')

    @property 
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """

        ### TODO: check this function for correctness... ###
        ### TODO: do we want body inertia orientation, or body frame orientation? ###
        return self.pose_in_base_from_name('right_hand')

    @property
    def _right_hand_joint_cartesian_velocity(self):
        """
        Returns the current cartesian velocity of the last robot joint with respect to
        the base frame as a tuple (vel, ang_vel), each is a 3-dim numpy array
        """
        raise Exception("Not implemented yet...")

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a tuple
        """

        ### TODO: get velocity in frame, not COM (xpos vs. xipos) by translating between the orientations... ###

        ### TODO: check this function for correctness... ###
        raise Exception("This function is almost certainly wrong...")

        base_pos_in_world = self._sim.data.get_body_xpos('base')
        base_rot_in_world = self._sim.data.get_body_xmat('base').reshape((3, 3))
        # base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = pose_inv(base_pose_in_world)


        ### TODO: convert COM velocity to frame velocity here... ###
        total_vel = self.physics.named.data.cvel['right_hand']
        eef_vel_in_world = total_vel[3:]
        eef_ang_vel_in_world = total_vel[:3]

        ### TODO: should we just rotate the axis? Or is this velocity conversion below correct? ###
        return vel_in_A_to_vel_in_B(vel_A=eef_vel_in_world, ang_vel_A=eef_ang_vel_in_world, pose_A_in_B=world_pose_in_base)

    def _convert_base_force_to_world_force(self, base_force):
        """
        Utility function to convert a force measured in the base frame to one in the world frame.
        This should be used when applying force control, since all control occurs with respect
        to the base frame but all simulation happens with respect to the world frame.
        """

        raise Exception("This function is almost certainly wrong...")
        
        base_pos_in_world = self._sim.data.get_body_xpos('base')
        base_rot_in_world = self._sim.data.get_body_xmat('base').reshape((3, 3))

        # base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)

        world_force = np.zeros(6)
        lin_force_in_world, rot_force_in_world = force_in_A_to_force_in_B(force_A=base_force[:3], torque_A=base_force[3:], pose_A_in_B=base_pose_in_world)
        world_force[:3] = lin_force_in_world
        world_force[3:] = rot_force_in_world
        print("world_force: {}".format(world_force))
        print("base_rot_in_world: {}".format(base_rot_in_world))
        return world_force

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot. 
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[0]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[1]

    @property
    def _joint_positions(self):
        #return self.sim.data.qpos[self.mujoco_robot.joints]
        return self._sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        #return self.sim.data.qvel[self.mujoco_robot.joints]
        return self._sim.data.qvel[self._ref_joint_vel_indexes]






