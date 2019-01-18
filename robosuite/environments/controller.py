"""
OSC Controller definitions.
"""

import numpy as np
from enum import Enum

import robosuite.utils.transform_utils as T
import mujoco_py

class ControllerType(str, Enum):
    POS = 'position'
    POS_ORI = 'position_orientation'
    POS_YAW = 'position_yaw'
    JOINT_IMP = 'joint_impedance'
    JOINT_TORQUE = 'joint_torque'
    JOINT_VEL = 'joint_velocity'


class controller():
    def __init__(self, control_max,
        control_min,
        max_action ,
        min_action ,
        impedance_flag = False,
        kp_max = None , 
        kp_min = None, 
        damping_max= None, 
        damping_min= None, 
        initial_joint= None):


        self.impedance_flag = impedance_flag

    
        self.initial_joint = initial_joint



        self.control_max= control_max
        self.control_min = control_min

        self.control_dim = self.control_max.shape[0]

        if self.impedance_flag:
            # kp_max = np.ones(self.control_dim) * kp_max
            # kp_min = np.ones(self.control_dim) * kp_min

            # kv_max = np.ones(self.control_dim) * damping_max
            # kv_min = np.ones(self.control_dim) * damping_min

            impedance_max = np.hstack((kp_max, damping_max))
            impedance_min = np.hstack((kp_min, damping_min))

            self.control_max = np.hstack((self.control_max, impedance_max))
            self.control_min = np.hstack((self.control_min, impedance_min))


        self.input_max = max_action
        self.input_min = min_action

        self.action_scale = abs(self.control_max-self.control_min)/abs(max_action - min_action)

        self.action_output_transform = (self.control_max+self.control_min)/2.0
        self.action_input_transform = (max_action + min_action )/2.0
    def transform_action(self, action):
        action = np.clip(action, self.input_min, self.input_max)
        
        transformed_action =    (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update_model(self, sim, joint_index, id_name='right_hand'):
        self.current_position = sim.data.body_xpos[sim.model.body_name2id(id_name)]
        # self.current_orientation = sim.data.body_xquat[sim.model.body_name2id(id_name)]
        self.current_orientation_mat = sim.data.body_xmat[sim.model.body_name2id(id_name)].reshape([3,3])
        self.current_lin_velocity = sim.data.body_xvelp[sim.model.body_name2id(id_name)]
        self.current_ang_velocity = sim.data.body_xvelr[sim.model.body_name2id(id_name)]


        self.current_joint_position = sim.data.qpos[joint_index]
        self.current_joint_velocity = sim.data.qvel[joint_index]

        self.Jx= sim.data.get_body_jacp(id_name).reshape((3, -1))[:, joint_index]
        self.Jr= sim.data.get_body_jacr(id_name).reshape((3, -1))[:, joint_index]
        self.J_full = np.vstack([self.Jx, self.Jr])

    def update_mass_matrix(self,sim, joint_index, id_name='right_hand'):
        mass_matrix =  np.ndarray(shape=(len(sim.data.qvel)**2,),dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(sim.model, mass_matrix, sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(sim.data.qvel),len(sim.data.qvel)))    
        self.mass_matrix = mass_matrix[joint_index,:][:,joint_index]


    def update_model_opspace(self,sim, joint_index, id_name='right_hand'):

        
        mass_matrix_inv = np.linalg.inv(self.mass_matrix)

        lambda_matrix_inv = np.dot(
                np.dot(self.J_full, mass_matrix_inv),
                self.J_full.transpose()
                )
        ## check for singularity here by doing np.linalg.inv with svd instead 

        svd_u, svd_s, svd_v = np.linalg.svd(lambda_matrix_inv)
        singularity_threshold = 0.00025
        svd_s_inv = [0 if x<singularity_threshold else 1./x for x in svd_s]
        self.lambda_matrix = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

        if self.initial_joint is not None:

            Jbar= np.dot(mass_matrix_inv, self.J_full.transpose()).dot(self.lambda_matrix)
            self.nullspace_matrix = np.eye(len(joint_index), len(joint_index)) - np.dot(Jbar, self.J_full)








    
    def action_to_torques(self, action, policy_step):
        raise NotImplementedError

    @property
    def action_dim(self):
        dim = self.control_dim
        if self.impedance_flag:
            dim = dim *3
        return dim

    @property

    def kp_index(self):
        start_index = self.control_dim
        end_index = start_index+ self.control_dim

        if self.impedance_flag:
            return (start_index, end_index)
        else:
            return None

    @property

    def kv_index(self):
        start_index = self.kp_index[1]
        end_index = start_index + self.control_dim

        if self.impedance_flag:
                return (start_index, end_index)
        else:
            return None
    @property
    def action_mask(self):
        raise NotImplementedError


class joint_torque_controller(controller):
    def __init__(self,control_max=np.array((30, 30, 30, 30, 30, 5, 5)), 
        control_min = -1*np.array((30, 30, 30, 30, 30, 5, 5)), 
        max_action = 1, 
        min_action = -1,
    ):

        super(joint_torque_controller, self).__init__(control_max,
            control_min,
            max_action ,
            min_action )

    def action_to_torques(self, action, policy_step):
        action = self.transform_action(action)

        return action


class joint_velocity_controller(controller):
    def __init__(self,control_max=np.ones(7)*2, 
        control_min = -1*np.ones(7)*2, 
        max_action = 1, 
        min_action = -1,
    ):

        super(joint_velocity_controller, self).__init__(control_max,
            control_min,
            max_action ,
            min_action )

        self.kv = np.array((8.0, 7.0, 6.0, 4.0, 2.0, 0.5, 0.1))

    def action_to_torques(self, action, policy_step):
        action = self.transform_action(action)



        torques = np.multiply(self.kv, (action-self.current_joint_velocity))

        return torques


class joint_impedance_controller(controller):
    def __init__(self,control_max=np.ones(7), 
        control_min = -1*np.ones(7), 
        max_action = 1, 
        min_action = -1,
        impedance_flag = False,
        kp_max = np.array((100, 100, 100, 100, 100, 100, 100)), 
        kp_min = np.array((10, 10, 10, 10, 10, 10, 10)), 
        damping_max= np.array((2, 2, 2, 2, 2, 2, 2)), 
        damping_min= np.zeros(7)
    ):

        super(joint_impedance_controller, self).__init__(control_max,
            control_min,
            max_action ,
            min_action ,
            impedance_flag,
            kp_max, kp_min, damping_max, damping_min)
            

        self.impedance_kp = np.ones(self.control_dim)*50
        self.impedance_kp[-2:] = np.ones(2)*25
        self.damping = np.ones(self.control_dim)

    def action_to_torques(self, action, policy_step):

        action = self.transform_action(action)
        if policy_step == True:
            self.goal_joint_position = self.current_joint_position + action[0:self.control_dim]

        position_joint_error = self.goal_joint_position - self.current_joint_position


        if self.impedance_flag:
            self.impedance_kp = action[self.kp_index[0]: self.kp_index[1]]
            self.damping = action[self.kv_index[0]:self.kv_index[1]]

        self.impedance_kv = 2*np.sqrt(self.impedance_kp) *self.damping
        

        if np.linalg.norm(self.current_joint_velocity)>7.0:
            self.current_joint_velocity /= (np.linalg.norm(self.current_joint_velocity)*7.0)

        torques = np.multiply(self.impedance_kp, position_joint_error) - np.multiply(self.impedance_kv, self.current_joint_velocity)

        decoupled_torques = np.dot(self.mass_matrix, torques)

        return decoupled_torques


    def update_model(self, sim, joint_index, id_name='right_hand', mass_matrix=None):

        super().update_model(sim, joint_index, id_name='right_hand')
 
        self.update_mass_matrix(sim, joint_index, id_name)


    @property
    def action_mask(self):
        return np.arange(self.control_dim)




class position_ori_controller(controller):
    def __init__(self,control_max=np.array((0.05, 0.05, 0.05, 0.1, 0.1, 0.1)), 
        control_min = -1*np.array((0.05, 0.05, 0.05, 0.1, 0.1, 0.1)), 
        max_action = 1, 
        min_action = -1,
        impedance_flag = False,
        kp_max = np.array((100, 100, 100, 100, 100, 100)), 
        kp_min = np.array((10, 10, 10, 10, 10, 10)), 
        damping_max= np.array((2, 2, 2, 2, 2, 2)), 
        damping_min= np.zeros(6), 
        initial_joint=None

        ):

        super(position_ori_controller, self).__init__(control_max,
        control_min,
        max_action ,
        min_action ,
        impedance_flag,
        kp_max, kp_min, damping_max, damping_min, initial_joint)
        

        self.impedance_kp = np.ones(6)*20
        self.damping = np.ones(6)

    def action_to_torques(self, action, policy_step):
        
        #assume your model is updated! PLEASE UPDATE MODEL!!!
        action = self.transform_action(action)

        if policy_step == True:
            self.set_goal_position(action)
            self.set_goal_orientation(action)

        position_error = self.goal_position - self.current_position

        ori_error_mat = np.dot(self.current_orientation_mat, np.linalg.inv(self.goal_orientation))
        

    
        ori_error_mat_44 = np.eye(4)
        ori_error_mat_44[0:3, 0:3] = ori_error_mat
        angle, direction, _ = T.mat2angle_axis_point(ori_error_mat_44)
        orientation_error = -angle*direction # compute "orientation error"




        if self.impedance_flag:
            self.impedance_kp[self.action_mask] = action[self.kp_index[0]: self.kp_index[1]]
            self.damping[self.action_mask] = action[self.kv_index[0]:self.kv_index[1]]

        self.impedance_kv = 2*np.sqrt(self.impedance_kp) *self.damping

        if np.linalg.norm(self.current_ang_velocity)>1.0:
            self.current_ang_velocity /= np.linalg.norm(self.current_ang_velocity)

        if np.linalg.norm(self.current_lin_velocity) > 5.0:
            self.current_lin_velocity/= (np.linalg.norm(self.current_lin_velocity)*5.0)

        return self.calculate_impedance_torques(position_error, orientation_error)

    def calculate_impedance_torques(self, position_error, orientation_error):

        desired_force = (np.multiply(np.array(position_error), np.array(self.impedance_kp[0:3])) 
            - np.multiply(np.array(self.current_lin_velocity), self.impedance_kv[0:3]))


        desired_torque = (np.multiply(np.array(orientation_error), np.array(self.impedance_kp[3:6])) 
            - np.multiply(np.array(self.current_ang_velocity), self.impedance_kv[3:6]))

        # print('orientation error new: ', orientation_error)

        # print('kp new ', self.impedance_kp)
        # print('ang vel: ', self.current_ang_velocity)
        # print('kv new: ', self.impedance_kv)
        # print('des torq new: ', desired_torque)


        desired_wrench = np.concatenate([desired_force, desired_torque])

        decoupled_wrench = np.dot(self.lambda_matrix, desired_wrench)
        torques = np.dot(self.J_full.T, decoupled_wrench)


        if self.initial_joint is not None:

            joint_kp = 10
            joint_kv = np.sqrt(joint_kp)*2

            pose_torques = np.dot(self.mass_matrix, (joint_kp*(self.initial_joint-self.current_joint_position) - joint_kv*self.current_joint_velocity))

            nullspace_torques = np.dot(self.nullspace_matrix.transpose(), pose_torques)

            torques += nullspace_torques


        # print('desired_wrench_decoupled_ new: ', decoupled_wrench)
        # print('desired_wrench_coupl new: ', desired_wrench)

        return torques
    

    def update_model(self, sim, joint_index, id_name='right_hand', mass_matrix=None):

        super().update_model(sim, joint_index, id_name='right_hand')
 
        self.update_mass_matrix(sim, joint_index, id_name)
        self.update_model_opspace(sim, joint_index, id_name='right_hand')
                 
    
    



    def set_goal_position(self, action, position=None):
        if position is not None:
            self._goal_position = position
        else:
            self._goal_position = self.current_position + action[0:3]
    def set_goal_orientation(self, action, orientation=None):
        if orientation is not  None:
            self._goal_orientation = orientation
        else:
            rotation_mat_error = T.euler2mat(action[3:6])
            self._goal_orientation = np.dot(self.current_orientation_mat, rotation_mat_error)


    @property
    def action_mask(self):
        return np.arange(self.control_dim)

    @property
    def goal_orientation(self):
        return self._goal_orientation
    


    @property
    def goal_position(self):
        return self._goal_position
    



class position_controller(position_ori_controller):

    def __init__(self, control_max=np.ones(3)*0.1, 
        control_min = np.ones(3)*-0.1, 
        max_action = 1.0, 
        min_action = -1.0,
        impedance_flag= False, 
        kp_max = np.array((100, 100, 100)), 
        kp_min = np.array((10, 10, 10)), 
        damping_max= np.array((2, 2, 2)), 
        damping_min= np.zeros(3)
        ):

        super(position_controller, self).__init__(control_max,
        control_min,
        max_action ,
        min_action ,
        impedance_flag, kp_max, kp_min, damping_max, damping_min)
        

        self.goal_orientation_set = False
        self.impedance_kp = np.ones(6)*20
        self.impedance_kp[3:] = np.ones(3)*100
        self.damping = np.ones(6)

    def set_goal_orientation(self, action, orientation=None):
        if orientation is not None:
            self._goal_orientation = orientation
        elif self.goal_orientation_set == False:
            self._goal_orientation=   np.array(self.current_orientation_mat)
            self.goal_orientation_set = True
    

    @property
    def goal_orientation(self):
        return self._goal_orientation


    @property
    def action_mask(self):

        return np.array((0,1,2))

    @property
    def goal_position(self):
        return self._goal_position
    
