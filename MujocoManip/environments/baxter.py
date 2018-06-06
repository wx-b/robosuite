import numpy as np
from collections import OrderedDict
from MujocoManip.environments.base import MujocoEnv
from MujocoManip.models import BaxterRobot, gripper_factory
import MujocoManip.miscellaneous.utils as U


class BaxterEnv(MujocoEnv):

    def __init__(self,
                 gripper_right=None,
                 gripper_left=None,
                 gripper_visualization=False,
                 use_indicator_object=True, #TODO: don't do that
                 **kwargs):
        self.has_gripper_right = not (gripper_right is None)
        self.has_gripper_left = not (gripper_left is None)
        self.gripper_right_name = gripper_right
        self.gripper_left_name = gripper_left
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        super().__init__(**kwargs)

    def _setup_mocap(self):
        U.mjpy_reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effectors into position.
        gripper_target = self.sim.data.get_body_xpos('right_hand')
        gripper_rotation = self.sim.data.get_body_xquat('right_hand')

        self.sim.data.set_mocap_pos('mocap', gripper_target)
        self.sim.data.set_mocap_quat('mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = BaxterRobot()
        if self.has_gripper_right:
            self.gripper_right = gripper_factory(self.gripper_right_name)
            if not self.gripper_visualization:
                self.gripper_right.hide_visualization()
            self.mujoco_robot.add_gripper('right_hand', self.gripper_right)

        if self.has_gripper_left:
            self.gripper_left = gripper_factory(self.gripper_left_name)
            if not self.gripper_visualization:
                self.gripper_left.hide_visualization()
            self.mujoco_robot.add_gripper('left_hand', self.gripper_left)

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos + 0.03 * np.random.normal(len(self._ref_joint_pos_indexes))

        if self.has_gripper_right:
            self.sim.data.qpos[self._ref_joint_gripper_right_actuator_indexes] = self.gripper_right.init_qpos
        if self.has_gripper_left:
            self.sim.data.qpos[self._ref_joint_gripper_left_actuator_indexes] = self.gripper_left.init_qpos

    def _get_reference(self):
        super()._get_reference()
        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]
        if self.use_indicator_object:
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = self.sim.model.get_joint_qpos_addr('pos_indicator')
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = self.sim.model.get_joint_qvel_addr('pos_indicator')
            self.indicator_id = self.sim.model.body_name2id('pos_indicator')

        # indices for grippers in qpos, qvel
        if self.has_gripper_left:
            self.gripper_left_joints = list(self.gripper_left.joints)
            self._ref_gripper_left_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_left_joints]
            self._ref_gripper_left_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_left_joints]
            self.left_eef_site_id = self.sim.model.site_name2id('l_g_grip_site')

        if self.has_gripper_right:
            self.gripper_right_joints = list(self.gripper_right.joints)
            self._ref_gripper_right_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_right_joints]
            self._ref_gripper_right_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_right_joints]
            self.right_eef_site_id = self.sim.model.site_name2id('grip_site')

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names  if actuator.startswith("pos")]

        self._ref_joint_vel_actuator_indexes = [self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names if actuator.startswith("vel")]

        if self.has_gripper_left:
            self._ref_joint_gripper_left_actuator_indexes = [self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names if actuator.startswith("gripper_l")]

        if self.has_gripper_right:
            self._ref_joint_gripper_right_actuator_indexes = [self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names if actuator.startswith("gripper_r")]

        if self.has_gripper_right:
            # IDs of sites for gripper visualization
            self.eef_site_id = self.sim.model.site_name2id('grip_site')
            self.eef_cylinder_id = self.sim.model.site_name2id('grip_site_cylinder')

    def move_indicator(self, pos):
        self.sim.data.qpos[self._ref_indicator_pos_low:self._ref_indicator_pos_low+3] = pos

    # Note: Overrides super
    def _pre_action(self, action):
        #print("action is", action)
        action = np.clip(action, -1, 1)    
        last = self.mujoco_robot.dof
        arm_action = action[:last]
        if self.has_gripper_right:
            gripper_right_action_in = action[last:last+self.gripper_right.dof]
            gripper_right_action_actual = self.gripper_right.format_action(gripper_right_action_in)
            arm_action = np.concatenate([arm_action, gripper_right_action_actual])
            last = last + self.gripper_right.dof
        if self.has_gripper_left:
            gripper_left_action_in = action[last:last+self.gripper_left.dof]
            gripper_left_action_actual = self.gripper_left.format_action(gripper_left_action_in)
            arm_action = np.concatenate([arm_action, gripper_left_action_actual])
        action = arm_action

        # rescale normalized action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange
        bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:,0])
        weight = 0.5 * (ctrl_range[:,1] - ctrl_range[:,0])
        applied_action = bias + weight * action
        self.sim.data.ctrl[:] = action#applied_action

        # gravity compensation
        self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]
        if self.use_indicator_object:
            self.sim.data.qfrc_applied[self._ref_indicator_vel_low:self._ref_indicator_vel_high] = self.sim.data.qfrc_bias[self._ref_indicator_vel_low:self._ref_indicator_vel_high]

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        di = super()._get_observation()
        # proprioceptive features
        di['joint_pos'] = np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])
        di['joint_vel'] = np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])
        if self.has_gripper_right:
            di['gripper_pos'] = [self.sim.data.qpos[x] for x in self._ref_gripper_right_joint_pos_indexes]
            di['gripper_vel'] = [self.sim.data.qvel[x] for x in self._ref_gripper_right_joint_vel_indexes]
        return di

    @property
    def dof(self):
        dof = self.mujoco_robot.dof
        if self.has_gripper_left:
            dof += self.gripper_left.dof
        if self.has_gripper_right:
            dof += self.gripper_right.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = U.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos('base')
        base_rot_in_world = self.sim.data.get_body_xmat('base').reshape((3, 3))
        base_pose_in_world = U.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = U.pose_inv(base_pose_in_world)

        pose_in_base = U.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
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
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name('right_hand')

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp('right_hand').reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[:7]]

        Jr = self.sim.data.get_body_jacr('right_hand').reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[:7]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

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
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property 
    def _left_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name('left_hand')

    @property
    def _left_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp('left_hand').reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[7:]]

        Jr = self.sim.data.get_body_jacr('left_hand').reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[7:]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _left_hand_pos(self):
        """
        Returns position of eef in base frame of robot. 
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _left_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._left_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _left_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._left_hand_total_velocity[:3]

    @property
    def _left_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._left_hand_total_velocity[3:]

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
        if self.has_gripper_right:
            self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
