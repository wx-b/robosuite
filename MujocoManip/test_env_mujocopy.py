from MujocoManip import SawyerStackEnv, SawyerPushEnv, SawyerGraspEnv, SawyerReachEnvEEVel, SawyerReachEnv, make
import numpy as np
import time
from PIL import Image
from IPython import embed
from mujoco_py import load_model_from_path, MjSim, MjViewer
from MujocoManip.model import DefaultCylinderObject, RandomCylinderObject, RandomBoxObject, DefaultBallObject, RandomBallObject, DefaultCapsuleObject, RandomCapsuleObject

if __name__ == '__main__':

    ### TODO: for some reason, when you just open a new terminal, import the env, do reset, then render, ###
    ###       it doesn't render the correct configuration. ###
    ### TODO: define observation space, action space (you can look at Julian's code for this) ###

    # a test case: do completely random actions at each time step
    # env = SawyerStackEnv(gripper='TwoFingerGripper')
    # env = SawyerStackEnv(gripper='RobotiqGripper')
    # env = SawyerStackEnv(gripper='PR2Gripper')
    # env = SawyerPushEnv()
    # env = SawyerGraspEnv(gripper="RobotiqThreeFingerGripper")
    # env = SawyerGraspEnv(gripper="RobotiqThreeFingerGripper")
    # env = SawyerReachEnv(end_effector_control=True, reward_objective_factor=500)
    # env = make("SawyerReachEnv", display=True, ignore_done=False)

    env = make("SawyerStackEnv", display=True, ignore_done=True)

    # env = make("SawyerStackEnv", display=True, ignore_done=True, use_eef_ctrl=True)
    # env = make("SawyerStackEnv", display=True, ignore_done=False, use_torque_ctrl=True)
    # env = make("SawyerPushEnv", display=True, ignore_done=False, use_torque_ctrl=True)

    ### TEST object generation ###
    # obj_arr = [RandomCapsuleObject() for _ in range(3)]
    # obj_arr.extend([RandomCylinderObject() for _ in range(5)])
    # obj_arr.extend([RandomBoxObject() for _ in range(5)])
    # obj_arr.extend([RandomBallObject() for _ in range(3)])
    # env = make("SawyerStackEnv", display=True, ignore_done=True, mujoco_objects=obj_arr)
    model = env.task.get_model(mode='mujoco_py')
    sim = MjSim(model)
    viewer = MjViewer(sim)

    sim_state = sim.get_state()
    original = sim.data.get_body_xpos('right_hand')
    while True:
        sim.set_state(sim_state)
        target = original + np.array([0, 0, 0.1]) + 0.05 * np.random.randn(3)
        sim.model.body_pos[sim.model.body_name2id('stacker_target_12')] = target
        for i in range(4000):
            
            jacp = sim.data.get_body_jacp('right_hand').reshape([3, -1])
            jacr = sim.data.get_body_jacr('right_hand').reshape([3, -1])
            _ref_joint_pos_indexes = [sim.model.get_joint_qpos_addr('right_j{}'.format(x)) for x in range(7)]
            _ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr('right_j{}'.format(x)) for x in range(7)]
            jacp_joint = jacp[:, _ref_joint_vel_indexes]
            jacr_joint = jacp[:, _ref_joint_vel_indexes]

            _ref_joint_vel_actuator_indexes = [sim.model.actuator_name2id(actuator) for actuator in sim.model.actuator_names 
                                                                                          if actuator.startswith("vel")]

            body_pos = sim.data.get_body_xpos('right_hand')
            diff = target - body_pos
            vel = diff
            sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]


            sol, _, _, _ = np.linalg.lstsq(jacp_joint, vel)
            sim.data.ctrl[_ref_joint_vel_actuator_indexes] = sol
            sim.step()
            viewer.render()
            # if i == 100:
            #   import pdb; pdb.set_trace()
