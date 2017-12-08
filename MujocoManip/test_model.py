from MujocoManip.model import *
from mujoco_py import load_model_from_path, MjSim, MjViewer

mujoco_robot = SawyerRobot()
# mujoco_robot.add_gripper(TwoFingerGripper())
# mujoco_robot.add_gripper(PR2Gripper())
mujoco_robot.add_gripper(TwoFingerGripper())
mujoco_robot.place_on([0,0,0])
mujoco_object = DefaultBallObject()
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.56,0,0])
task = SingleObjectTargetTask(mujoco_arena, mujoco_robot, mujoco_object)
model = task.get_model()
# task.save_model('sample_combined_model.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

sim_state = sim.get_state()
original = sim.data.get_body_xpos('right_hand')
while True:
    sim.set_state(sim_state)
    target = original + np.array([0, 0, 0.1]) + 0.05 * np.random.randn(3)
    sim.model.body_pos[sim.model.body_name2id('target')] = target
    for i in range(4000):
        #sim.data.ctrl[:] = np.random.rand(7) * 3
        # print(sim.data.ctrl[:])
        
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
        # 	import pdb; pdb.set_trace()
