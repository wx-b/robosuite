import numpy as np
from collections import OrderedDict
from MujocoManip.environments.base import MujocoEnv
from MujocoManip.model import SawyerRobot, gripper_factory
from MujocoManip.miscellaneous.utils import *


class SawyerEnv(MujocoEnv):

    def __init__(self, gripper=None, use_eef_ctrl=False, use_torque_ctrl=False, use_force_ctrl=False, **kwargs):
        self.has_gripper = not (gripper is None)
        self.gripper_name = gripper
        self.use_eef_ctrl = use_eef_ctrl
        self.use_torque_ctrl = use_torque_ctrl
        self.use_force_ctrl = use_force_ctrl
        super().__init__(**kwargs)

        ### TODO: any joint positions need to be set here? ###

        # self.physics.model.name2id('grip_site', 'site') # can get IDs this way

        # setup mocap stuff if necessary
        if self.use_eef_ctrl:
            self._setup_mocap()
        
    def _setup_mocap(self):
        mjpy_reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = self.sim.data.get_body_xpos('right_hand')
        gripper_rotation = self.sim.data.get_body_xquat('right_hand')

        self.sim.data.set_mocap_pos('mocap', gripper_target)
        self.sim.data.set_mocap_quat('mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = SawyerRobot(use_torque_ctrl=self.use_torque_ctrl, use_eef_ctrl=self.use_eef_ctrl)
        if self.has_gripper:
            # self.gripper = gripper_factory()
            self.mujoco_robot.add_gripper('right_hand', self.gripper_name)
            self.gripper = self.mujoco_robot.grippers['right_hand']

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

        if self.has_gripper:
            self.sim.data.qpos[self._ref_joint_gripper_actuator_indexes] = self.gripper.init_qpos

    def _get_reference(self):
        super()._get_reference()
        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints]
            self._ref_gripper_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints]


        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [self.sim.model.actuator_name2id(actuator) for actuator in self.sim.model.actuator_names 
                                                                                      if actuator.startswith("pos")]
        self._ref_joint_vel_actuator_indexes = [self.sim.model.actuator_name2id(actuator) for actuator in self.sim.model.actuator_names 
                                                                                      if actuator.startswith("vel")]

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [self.sim.model.actuator_name2id(actuator) for actuator in self.sim.model.actuator_names 
                                                                                              if actuator.startswith("gripper")]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id('grip_site')
        self.eef_cylinder_id = self.sim.model.site_name2id('grip_site_cylinder')

    # Note: Overrides super
    def _pre_action(self, action):
        if self.use_eef_ctrl:
            # assert len(action) == 5
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
            mjpy_ctrl_set_action(self.sim, action)
            mjpy_mocap_set_action(self.sim, action)

            # gravity compensation
            self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

        elif self.use_force_ctrl:

            ### TODO: convert the following in mujoco_py??? ###

            ### TODO: is this force acting in the end effector frame? If so, we need to translate to base coords... ###

            # note we convert force in base frame to force in world frame
            # self.physics.named.data.xfrc_applied['right_hand'] = self._convert_base_force_to_world_force(action[:6])
            self.physics.named.data.xfrc_applied['right_hand'] = action[:6]

            # gravity compensation
            self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]
        
        else:
            action = np.clip(action, -1, 1)    
            if self.has_gripper:
                arm_action = action[:self.mujoco_robot.dof]
                gripper_action_in = action[self.mujoco_robot.dof:self.mujoco_robot.dof+self.gripper.dof]
                gripper_action_actual = self.gripper.format_action(gripper_action_in)
                action = np.concatenate([arm_action, gripper_action_actual])

            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:,0])
            weight = 0.5 * (ctrl_range[:,1] - ctrl_range[:,0])
            applied_action = bias + weight * action
            self.sim.data.ctrl[:] = applied_action

            # gravity compensation
            self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        di = super()._get_observation()
        joint_pos = [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        joint_vel = [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        if self.has_gripper:
            joint_pos += [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            joint_vel += [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
        di['proprioception'] = np.concatenate([joint_pos, joint_vel])
        return di

    def dof(self):
        if self.use_eef_ctrl:
            dof = 3
        else:
            dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        # # note we convert (w, x, y, z) quat to (x, y, z, w)
        # eef_rot_in_world = quat2mat(self.physics.named.data.xquat['right_hand'][[1, 2, 3, 0]])
        pose_in_world = make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos('base')
        base_rot_in_world = self.sim.data.get_body_xmat('base').reshape((3, 3))
        # base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = pose_inv(base_pose_in_world)

        pose_in_base = pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def action_space(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

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

        base_pos_in_world = self.sim.data.get_body_xpos('base')
        base_rot_in_world = self.sim.data.get_body_xmat('base').reshape((3, 3))
        # base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = pose_inv(base_pose_in_world)

        ### TODO: convert COM velocity to frame velocity here... ###

        eef_vel_in_world = self.sim.data.get_body_xvelp('right_hand')
        eef_ang_vel_in_world = self.sim.data.get_body_xvelr('right_hand')

        ### TODO: should we just rotate the axis? Or is this velocity conversion below correct? ###
        return vel_in_A_to_vel_in_B(vel_A=eef_vel_in_world, ang_vel_A=eef_ang_vel_in_world, pose_A_in_B=world_pose_in_base)

    def _convert_base_force_to_world_force(self, base_force):
        """
        Utility function to convert a force measured in the base frame to one in the world frame.
        This should be used when applying force control, since all control occurs with respect
        to the base frame but all simulation happens with respect to the world frame.
        """

        base_pos_in_world = self.sim.data.get_body_xpos('base')
        base_rot_in_world = self.sim.data.get_body_xmat('base').reshape((3, 3))
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
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """
        
        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
