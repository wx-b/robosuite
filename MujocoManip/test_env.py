from MujocoManip import SawyerStackEnv, SawyerPushEnv, SawyerGraspEnv, SawyerReachEnvEEVel, SawyerReachEnv, make
import numpy as np
import time
from PIL import Image
from IPython import embed

from MujocoManip.model import DefautCylinderObject, RandomCylinderObject, RandomBoxObject

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
    # obj_arr = [DefautCylinderObject() for _ in range(10)]
    # obj_arr = [RandomCylinderObject() for _ in range(5)]
    # obj_arr.extend([RandomBoxObject() for _ in range(5)])
    # env = make("SawyerStackEnv", display=True, ignore_done=True, mujoco_objects=obj_arr)

    obs = env.reset()

    # rotate the gripper so we can see it easily 
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    dof = env.dof()
    print('action space', env.action_space)
    print('Obs: {}'.format(len(obs)))
    print('DOF: {}'.format(dof))
    env.render()

    target_jp = env._joint_positions
    target_jp -= 0.1
    kp = np.array([50.0, 30.0, 20.0, 15.0, 10.0, 5.0, 2.0]) # gains for PD controller
    kv = np.array([8.0, 7.0, 6.0, 4.0, 2.0, 0.5, 0.1]) # gains for PD controller

    ### TODO: try torque control with fixed joint positions that are slightly off from the start ones? ###

    #  0                  floor [ 0.56      0         0       ]
    #  1        table_collision [ 0.56      0         0.4     ]
    #  2           table_visual [ 0.56      0         0.4     ]
    # 28               top_geom [ 0.447     0.163     1.13    ]
    # 29 electric_gripper_base1 [ 0.445     0.163     1.1     ]
    # 30 electric_gripper_base2 [ 0.447     0.163     1.1     ]
    # 31             l_finger_1 [ 0.441     0.181     1.06    ]
    # 32             l_finger_2 [ 0.446     0.167     1.04    ]
    # 33             l_finger_3 [ 0.438     0.186     1.07    ]
    # 34         l_finger_tip_1 [ 0.445     0.172     1.02    ]
    # 35         l_finger_tip_2 [ 0.445     0.172     1.02    ]
    # 36             r_finger_1 [ 0.451     0.16      1.06    ]
    # 37             r_finger_2 [ 0.445     0.173     1.04    ]
    # 38             r_finger_3 [ 0.454     0.154     1.07    ]
    # 39         r_finger_tip_1 [ 0.446     0.169     1.02    ]
    # 40         r_finger_tip_2 [ 0.446     0.169     1.02    ]

    # collision: l_finger_2 with r_finger_3
    # collision: l_finger_3 with r_finger_2
    # collision: l_finger_3 with r_finger_3

    names = {}
    # names[0] = 'floor'
    # names[1] = 'table_collision'
    # names[2] = 'table_visual'
    names[28] = 'top_geom'
    names[29] = 'electric_gripper_base1'
    names[30] = 'electric_gripper_base2'
    names[31] = 'l_finger_1'
    names[32] = 'l_finger_2'
    names[33] = 'l_finger_3'
    names[34] = 'l_finger_tip_1'
    names[35] = 'l_finger_tip_2'
    names[36] = 'r_finger_1'
    names[37] = 'r_finger_2'
    names[38] = 'r_finger_3'
    names[39] = 'r_finger_tip_1'
    names[40] = 'r_finger_tip_2'

    while True:
        obs = env.reset()

        # rotate the gripper so we can see it easily 
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        for i in range(20000):
            # print(obs[len(obs) - 6: len(obs) - 3])
            # print(obs[len(obs) - 9: len(obs) - 6])
            # action = obs[len(obs) - 3: len(obs)]
            # action[:7] = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]
            # action = 0.5 * np.random.randn(dof)
            action = np.zeros(dof)
            action[7:] = [-1., 1.]
            # action[2] = 0.1

            # pos_err = env._joint_positions - target_jp
            # vel_err = env._joint_velocities
            # action = - kp * pos_err - kv * vel_err

            # action[:] = [0, 0, 0, 0, 0, 0, 0.01]
            # action[:6] = [0, 0, 1.0, 0, 0, 0]
            # action[7] = 1
            obs, reward, done, info = env.step(action)
            env.render()
            # time.sleep(0.2)

            major_keys = names.keys()
            for c in env.physics.data.contact:
                g1, g2 = c.geom1, c.geom2
                if g1 in major_keys or g2 in major_keys:
                    pass
                    # print("collision: {} with {}".format(names.get(g1, g1), names.get(g2, g2)))
            
            # if i % 500 == 0:
            #     A = env.sim.render(width=500, height=500, camera_name='camera1')
            #     im = Image.fromarray(A)
            #     im.save('{}.jpg'.format(i))
            # print('obs: {}'.format(obs))
            # print('reward: {}'.format(reward))
            # t = time.time()
            # A = env.sim.render(500, 500)
            # dt = time.time() - t
            # print('dt: {}'.format(dt))
            if i % 100 == 0:
                print(i)
            if done:
                print('done: {}'.format(reward))
                break

