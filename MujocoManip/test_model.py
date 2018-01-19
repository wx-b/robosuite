from MujocoManip.model import *
from MujocoManip.miscellaneous import DmControlRenderer
import sys
mujoco_robot = SawyerRobot()
# mujoco_robot.add_gripper(TwoFingerGripper())
# mujoco_robot.add_gripper(PR2Gripper())
mujoco_robot.add_gripper(TwoFingerGripper())
mujoco_robot.place_on([0,0,0])
mujoco_object = DefaultBallObject()
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.56,0,0])
task = SingleObjectTargetTask(mujoco_arena, mujoco_robot, mujoco_object)
physics = task.get_model(mode='dm_control')
render = DmControlRenderer(physics)

original = physics.named.data.xpos['right_hand']
joints = ['right_j{}'.format(x) for x in range(7)]
while True:
    target = original + np.array([0, 0, 0.1]) + 0.05 * np.random.randn(3)
    physics.named.model.body_pos['target'][:] = target
    # physics.after_reset()
    
    for i in range(4000):

        # jacp = physics.named.data.body_jacp['right_hand'].reshape([3, -1])
        # jacr = physics.named.data.body_jacr['right_hand'].reshape([3, -1])
        # _ref_joint_pos_indexes = [sim.model.get_joint_qpos_addr('right_j{}'.format(x)) for x in range(7)]
        # _ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr('right_j{}'.format(x)) for x in range(7)]
        # jacp_joint = jacp[:, _ref_joint_vel_indexes]
        # jacr_joint = jacp[:, _ref_joint_vel_indexes]

        # _ref_joint_vel_actuator_indexes = [sim.model.actuator_name2id(actuator) for actuator in sim.model.actuator_names 
        #                                                                               if actuator.startswith("vel")]

        # body_pos = sim.data.get_body_xpos('right_hand')
        # diff = target - body_pos
        # vel = diff
        for joint in joints:
            physics.named.data.qfrc_applied[joint] = physics.named.data.qfrc_bias[joint] * 1.5
        # physics.get_body_jacp('right_hand')
        if i == 20:
            if '--pdb' in sys.argv:
                import pdb; pdb.set_trace()
        # physics.after_reset()
        # with physics.reset_context():
        #     target = original + np.array([0, 0, 0.1]) + 0.05 * np.random.randn(3)
        #     physics.named.data.xpos['target'] = target
        # TODO: figure out jacobian
        
        # sol, _, _, _ = np.linalg.lstsq(jacp_joint, vel)
        # sim.data.ctrl[_ref_joint_vel_actuator_indexes] = sol
        # sim.step()
        # viewer.render()
        # if i == 100:
        # 	import pdb; pdb.set_trace()
        # arr = physics.render()
        # print('render:', arr)
        render.render()
        physics.step()