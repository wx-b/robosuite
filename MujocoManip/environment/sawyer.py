import numpy as np
from MujocoManip.environment.base import MujocoEnv
from MujocoManip.model import SawyerRobot, gripper_factory
from MujocoManip.miscellaneous.utils import *


class SawyerEnv(MujocoEnv):
    def __init__(self, gripper=None, end_effector_control=False, use_torque_ctrl=False, use_force_ctrl=False, **kwargs):
        self.has_gripper = not (gripper is None)
        self.gripper_name = gripper
        self.end_effector_control = end_effector_control
        self.use_torque_ctrl = use_torque_ctrl
        self.use_force_ctrl = use_force_ctrl
        super().__init__(**kwargs)
        

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = SawyerRobot(use_torque_ctrl=self.use_torque_ctrl)
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_name)
            self.mujoco_robot.add_gripper(self.gripper)

    def _reset_internal(self):
        super()._reset_internal()
        self.physics.named.data.qpos[self.mujoco_robot.joints] = self.mujoco_robot.rest_pos

        if self.has_gripper:
            self.physics.named.data.qpos[self.gripper.joints] = self.gripper.rest_pos

    def _get_reference(self):
        super()._get_reference()
        # self.joint_pos_actuators = [actuator for actuator in self.physics.named.model.actuator_gear if actuator.startswith("pos")]
        # self.joint_vel_actuators = [actuator for actuator in self.physics.named.model.actuator_gear if actuator.startswith("vel")]
        # # indices for joint pos actuation, joint vel actuation, gripper actuation
        # self._ref_joint_pos_actuator_indexes = [self.model.actuator_name2id(actuator) for actuator in self.model.actuator_names 
        #                                                                               if actuator.startswith("pos")]
        # self._ref_joint_vel_actuator_indexes = [self.model.actuator_name2id(actuator) for actuator in self.model.actuator_names 
        #                                                                               if actuator.startswith("vel")]
        
                                                                                          # if actuator.startswith("gripper")]

    # Note: Overrides super
    def _pre_action(self, action):
        if self.end_effector_control:
            raise NotImplementedError
            # jacp = self.sim.data.get_body_jacp('right_hand').reshape([3, -1])
            # jacp_joint = jacp[:, self._ref_joint_vel_indexes]
            # vel = action[0:3]
            # sol, _, _, _ = np.linalg.lstsq(jacp_joint, vel)
            # self.sim.data.ctrl[:] = np.concatenate([sol, self.gripper.rest_pos()])
        else:
            if self.use_torque_ctrl:
                # correct for gravity and add in torques
                self.physics.data.ctrl[:7] = action[:7]
                self.physics.named.data.qfrc_applied[self.mujoco_robot.joints] = self.physics.named.data.qfrc_bias[self.mujoco_robot.joints] 
            elif self.use_force_ctrl:
                self.physics.named.data.xfrc_applied['right_hand'] = action[:6]

                # gravity compensation
                self.physics.named.data.qfrc_applied[self.mujoco_robot.joints] = self.physics.named.data.qfrc_bias[self.mujoco_robot.joints]
            else:
                action = np.clip(action, -1, 1)    
                if self.has_gripper:
                    arm_action = action[:self.mujoco_robot.dof()]
                    gripper_action_in = action[self.mujoco_robot.dof():self.mujoco_robot.dof()+self.gripper.dof()]
                    gripper_action_actual = self.gripper.format_action(gripper_action_in)
                    action = np.concatenate([arm_action, gripper_action_actual])

                # rescale normalized action to control ranges
                ctrl_range = self.physics.model.actuator_ctrlrange
                bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:,0])
                weight = 0.5 * (ctrl_range[:,1] - ctrl_range[:,0])
                applied_action = bias + weight * action
                self.physics.data.ctrl[:] = applied_action

                # gravity compensation
                self.physics.named.data.qfrc_applied[self.mujoco_robot.joints] = self.physics.named.data.qfrc_bias[self.mujoco_robot.joints]

    def _get_observation(self):
        obs = super()._get_observation()
        # TODO: make sure no overwriting is happening
        joint_pos = self.physics.named.data.qpos[self.mujoco_robot.joints]
        joint_pos_sin = np.sin(joint_pos)
        joint_pos_cos = np.cos(joint_pos)
        joint_vel = self.physics.named.data.qvel[self.mujoco_robot.joints]
        # obs['joint_pos'] = [self.phycis.named.data.qpos[joint] for joint in self.joints]
        # obs['joint_pos_sin'] = np.sin(obs['joint_pos'])
        # obs['joint_pos_cos'] = np.cos(obs['joint_pos'])
        # obs['joint_vel'] = [self.phycis.named.data.qvel[joint] for joint in self.joints]
        return np.concatenate([obs, joint_pos, joint_pos_sin, joint_pos_cos, joint_vel])

    def dof(self):
        if self.end_effector_control:
            dof = 3
        else:
            dof = self.mujoco_robot.dof()
        if self.has_gripper:
            dof += self.gripper.dof()
        return dof

    @property
    #TODO: fix it
    def action_space(self):
        # TODO: I am not sure if we want to add gym dependency just for observation space and action space
        # return spaces.Box(
        low=np.ones(self.dof()) * -1.
        high=np.ones(self.dof()) * 1.
        # )
        return low, high

    @property 
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """

        ### TODO: check this function for correctness... ###

        eef_pos_in_world = self.physics.named.data.xpos['right_hand']
        # note we convert (w, x, y, z) quat to (x, y, z, w)
        eef_rot_in_world = quat2mat(self.physics.named.data.xquat['right_hand'][[1, 2, 3, 0]])
        eef_pose_in_world = make_pose(eef_pos_in_world, eef_rot_in_world)

        base_pos_in_world = self.physics.named.data.xpos['base']
        base_rot_in_world = quat2mat(self.physics.named.data.xquat['base'][[1, 2, 3, 0]])
        base_pose_in_world = make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = pose_inv(base_pose_in_world)

        eef_pose_in_base = pose_in_A_to_pose_in_B(eef_pose_in_world, world_pose_in_base)
        return eef_pose_in_base

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a tuple
        """

        ### TODO: check this function for correctness... ###

        world_pose_in_base = pose_inv(self._right_hand_pose)
        world_pos_in_base = world_pose_in_base[3, :3]
        world_rot_in_base = world_pose_in_base[:3, :3]

        # Get linear and angular velocities in world frame
        eef_vel_in_world = self.physics.named.data.subtree_linvel['right_hand']
        eef_ang_momentum_in_world = self.physics.named.data.subtree_angmom['right_hand']

        ### IMPORTANT: I'm assuming the last element in the 10-dim matrix is the point mass, could be wrong... ###
        eef_moment_of_inertia = self.physics.named.data.cinert['right_hand'][:9].reshape((3, 3))
        eef_ang_vel_in_world = (np.linalg.inv(eef_moment_of_inertia)).dot(eef_ang_momentum_in_world)

        # Convert velocities to base frame using pose.
        skew_symm = np.array([0., -world_pos_in_base[2], world_pos_in_base[1], 
                              world_pos_in_base[2], 0., -world_pos_in_base[0],
                              -world_pos_in_base[1], world_pos_in_base[0], 0.]).reshape((3, 3))
        eef_vel_in_base = world_rot_in_base.T.dot(eef_vel_in_world) + (skew_symm.dot(world_rot_in_base)).dot(eef_ang_vel_in_world)
        eef_ang_vel_in_base = world_rot_in_base.dot(eef_ang_vel_in_world)
        return eef_vel_in_base, eef_ang_vel_in_base

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
    def _right_hand_angular_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[1]

    @property
    def _joint_positions(self):
        #return self.sim.data.qpos[self.mujoco_robot.joints]
        return self.physics.named.data.qpos[self.mujoco_robot.joints]

    @property
    def _joint_velocities(self):
        #return self.sim.data.qvel[self.mujoco_robot.joints]
        return self.physics.named.data.qvel[self.mujoco_robot.joints]


