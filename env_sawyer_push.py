#!/usr/bin/env python3
'''
Displays robot fetch at a disco party.
'''

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
from xml_manip import MujocoRobot, MujocoGripper, MujocoObject, PusherTask
import numpy as np
import os
import sys
import time

class MujocoEnv(object):
    def __init__(self, debug=True, display=True):
        self.debug = debug
        self.model = self._load_model()
        self.sim = MjSim(self.model)
        self.display = display
        if self.display:
            self.viewer = MjViewer(self.sim)
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.set_cam()
        self.done = False
        

    def set_cam(self):
        self.viewer.cam.fixedcamid = 0
        # viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.azimuth = 179.7749999999999
        self.viewer.cam.distance = 3.825077470729921
        self.viewer.cam.elevation = -21.824999999999992
        self.viewer.cam.lookat[:][0] = 0.09691817
        self.viewer.cam.lookat[:][1] = 0.00164106
        self.viewer.cam.lookat[:][2] = -0.30996464


    def _load_model(self):
        pass

    def _get_reference(self):
        pass

    def _reset(self):
        self._reset_internal()
        return self._get_observation()

    def _reset_internal(self):
        self.sim.set_state(self.sim_state_initial)
        self.done = False

    def _get_observation(self):
        return []

    def _step(self, action):
        # import pdb; pdb.set_trace()
        reward = 0
        info = None
        if not self.done:
            self._pre_action(action)
            self.sim.step()
            reward, done, info = self._post_action(action)
            return self._get_observation(), reward, done, info
        else:
            return self._get_observation(), 0, True, None

    def _pre_action(self, action):
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        self.done = self._check_done()
        reward = self._reward(action)
        # TODO: how to manage info?
        return reward, self.done, {}

    def _check_done(self):
        return False

    def _reward(self, action):
        return 0

    def _render(self):
        if self.display:
            self.viewer.render()

class SawyerEnv(MujocoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = MujocoRobot('robots/sawyer/robot.xml')
        self.mujoco_robot.add_gripper(MujocoGripper('robots/sawyer/gripper.xml'))

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]

    def _get_reference(self):
        super()._get_reference()
        self._ref_joint_pos_indexes = [self.model.get_joint_qpos_addr('right_j{}'.format(x)) for x in range(7)]
        self._ref_joint_vel_indexes = [self.model.get_joint_qvel_addr('right_j{}'.format(x)) for x in range(7)]

    def _pre_action(self, action):
        super()._pre_action(action)
        # correct for gravity
        self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

    def _get_observation(self):
        obs = super()._get_observation()
        joint_pos = [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        joint_pos_sin = np.sin(joint_pos)
        joint_pos_cos = np.cos(joint_pos)
        joint_vel = [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        return np.concatenate([obs, joint_pos, joint_pos_sin, joint_pos_cos, joint_vel])

    @property
    def _right_hand_pos(self):
        return self.sim.data.get_body_xpos('right_hand') - self._object_pos_rest


    @property
    def _right_hand_vel(self):
        return self.sim.data.get_body_xvelp('right_hand')



class SawyerPushEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_object = MujocoObject('robots/sawyer/object_ball.xml')
        self.arena = PusherTask(self.mujoco_robot, self.mujoco_object)
        if self.debug:
            self.arena.save_model('sample_combined_model.xml')
        return self.arena.get_model()

    def _get_reference(self):
        super()._get_reference()
        self._ref_object_pos_low, self._ref_object_pos_high = self.model.get_joint_qpos_addr('pusher_object_free_joint')
        self._ref_object_vel_low, self._ref_object_vel_high = self.model.get_joint_qvel_addr('pusher_object_free_joint')
        
        self._object_pos_rest = np.copy(self.sim.data.get_body_xpos('pusher_object'))
        self._target_pos_rest = np.copy(self.sim.model.body_pos[self.sim.model.body_name2id('pusher_target')])

    def _reset_internal(self):
        super()._reset_internal()
        # rest position of target
        target_x = np.random.uniform(high=0.3, low=-0.3)
        target_x += target_x / abs(target_x) * 0.1
        target_y = np.random.uniform(high=0.3, low=-0.3)
        target_y += target_y / abs(target_y) * 0.1
        pos = [target_x, target_y, 0]
        self._target_pos = pos
    

    def _reward(self, action):
        reward = 0
        if self._check_win():
            reward += 2
        elif self._check_lose():
            reward -= 2
        # TODO: set a good action penalty coefficient
        reward += np.exp(-2. * np.linalg.norm(self._target_pos - self._object_pos, 2))
        reward -= 0.01 * np.linalg.norm(action, 2)
        return reward

    def _get_observation(self):
        obs = super()._get_observation()

        hand_pos = self._right_hand_pos
        object_pos = self._object_pos
        target_pos = self._target_pos

        hand_vel = self._right_hand_vel
        object_vel = self._object_vel

        object_pos_rel = object_pos - hand_pos
        target_pos_rel = target_pos - hand_pos

        object_vel_rel = object_vel - hand_vel

        return np.concatenate([ obs,
                                object_pos_rel,
                                object_vel_rel,
                                target_pos_rel,
                                ])


    def _check_done(self):
        return self._check_lose() or self._check_win()

    def _check_lose(self):
        return np.max(np.abs(self._object_pos[0:2] - self._target_pos[0:2])) > 0.3

    def _check_win(self):
        return np.allclose(self._target_pos, self._object_pos, rtol=1e-2)

    ####
    # Properties for objects
    ####

    @property
    def _object_pos(self):
        return self.sim.data.get_body_xpos('pusher_object') - self._object_pos_rest

    @property
    def _target_pos(self):
        return self.sim.model.body_pos[self.sim.model.body_name2id('pusher_target')] - self._target_pos_rest

    @_target_pos.setter
    def _target_pos(self, pos):
        self.sim.model.body_pos[self.sim.model.body_name2id('pusher_target')] = pos + self._target_pos_rest

    @property
    def _object_vel(self):
        return self.sim.data.get_body_xvelp('pusher_object')
    
env = SawyerPushEnv()
obs = env._reset()
print('Initial Obs: {}'.format(obs))
while True:
    obs = env._reset()
    # print(obs)
    action = np.random.rand(16) * 2
    for i in range(2000):
        if i % 500 == 499:
            action = np.random.rand(16) * 2
        obs, reward, done, info = env._step(action)
        # 
        # obs, reward, done, info = env._step([0,-1,0,0,0,0,2])
        # print(obs, reward, done, info)
        env._render()
        if done:
            print('done: {}'.format(reward))
            break
