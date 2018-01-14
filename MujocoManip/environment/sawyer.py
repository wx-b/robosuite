import numpy as np
from MujocoManip.environment.base import MujocoEnv
from MujocoManip.model import SawyerRobot, gripper_factory



class SawyerEnv(MujocoEnv):
    def __init__(self, gripper=None, end_effector_control=False, **kwargs):
        self.has_gripper = not (gripper is None)
        self.gripper_name = gripper
        self.end_effector_control = end_effector_control
        super().__init__(**kwargs)
        

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = SawyerRobot()
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
            action = np.clip(action, -1, 1)    
            if self.has_gripper:
                arm_action = action[:self.mujoco_robot.dof()]
                gripper_action_in = action[self.mujoco_robot.dof():self.mujoco_robot.dof()+self.gripper.dof()]
                gripper_action_actual = self.gripper.format_action(gripper_action_in)
                action = np.concatenate([arm_action, gripper_action_actual])

            ctrl_range = self.physics.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:,1] + ctrl_range[:,0])
            weight = 0.5 * (ctrl_range[:,1] - ctrl_range[:,0])
            applied_action = bias + weight * action
            self.physics.data.ctrl[:] = applied_action

        # correct for gravity
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
    def _right_hand_pos(self):
        return self.physics.named.data.xpos['right_hand'] - self._pos_offset


    @property
    def _right_hand_vel(self):
        return self.physics.named.data.subtree_linvel['right_hand']

    @property
    def _joint_positions(self):
        return self.sim.data.qpos[self.mujoco_robot.joints]

    @property
    def _joint_velocities(self):
        return self.sim.data.qvel[self.mujoco_robot.joints]
