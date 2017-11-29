import numpy as np
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model import MujocoXMLObject, PusherTask, TableArena
from MujocoManip.model.model_util import xml_path_completion

class SawyerPushEnv(SawyerEnv):
    def __init__(self, 
                mujoco_object=None,
                table_size=(0.8, 0.8, 0.8),
                min_target_xy_distance=(0.1,0.1),
                table_friction=None,
                reward_lose=-1,
                reward_win=1,
                reward_action_norm_factor=0,
                reward_objective_factor=5,
                reward_touch_object_factor=0.001,
                reward_align_direction_factor=0.001,
                win_rel_tolerance=1e-2,
                **kwargs):
        """
            @mujoco_object(None), the object to be pushed, need that is is an MujocoObject instace
            If None, load 'object/object_ball.xml'
            @table_size, the FULL size of the table 
            @min_target_xy_distance: Minimal x/y distance between object and target
            @friction: friction coefficient of table, None for mujoco default
            @reward_win: reward given to the agent when it completes the task
            @reward_lose: reward given to the agent when it fails the task
            @reward_action_norm_factor: reward scaling factor that penalizes large actions
            @reward_objective_factor: reward scaling factor for being close to completing the objective
            @win_rel_tolerance: relative tolerance between object and target location 
                used when deciding if the agent has completed the task
            TODO(extension): table friction
        """
        # Handle parameters
        self.mujoco_object = mujoco_object
        if self.mujoco_object is None:
            self.mujoco_object = MujocoXMLObject(xml_path_completion('object/object_ball.xml'))
        self.table_size = table_size
        self.min_target_xy_distance = min_target_xy_distance
        self.table_friction = table_friction

        self.reward_lose=reward_lose
        self.reward_win=reward_win
        self.reward_action_norm_factor=reward_action_norm_factor
        self.reward_objective_factor=reward_objective_factor
        self.reward_touch_object_factor=reward_touch_object_factor
        self.reward_align_direction_factor=reward_align_direction_factor
        self.win_rel_tolerance = win_rel_tolerance

        super().__init__(**kwargs)
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
        
        self.target_bottom_offset = self.mujoco_object.get_bottom_offset()
        self.task = PusherTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_object)

        if self.debug:
            self.task.save_model('sample_combined_model.xml')
        return self.task.get_model()

    def _reset_internal(self):
        super()._reset_internal()
        # rest position of target
        table_x_half = self.table_size[0] / 2
        table_y_half = self.table_size[1] / 2
        target_x = np.random.choice([-1., 1.]) * np.random.uniform(high=table_x_half, low=self.min_target_xy_distance[0])
        target_y = np.random.choice([-1., 1.]) * np.random.uniform(high=table_y_half, low=self.min_target_xy_distance[1])
        self._set_target_xy(target_x, target_y)
        
    def _set_target_xy(self, x,y):
        self._target_pos = np.array([x,y,0]) - self.target_bottom_offset
    
    def _pre_action(self):
        super()._pre_action()
        self.pre_action_object_target_dist = np.linalg.norm(self._target_pos[:2] - self._object_pos[:2])

    def _reward(self, action):
        reward = 0
        self.post_action_object_target_dist = np.linalg.norm(self._target_pos[:2] - self._object_pos[:2])
        # Credit to jyg
        # Secret sauce to get pushing working
        if self._check_win():
            reward += self.reward_win
        elif self._check_lose():
            reward += self.reward_lose
        # TODO: set a good action penalty coefficient
        reward += self.reward_objective_factor * (self.pre_action_object_target_dist - self.post_action_object_target_dist)
        reward += self.reward_action_norm_factor * np.linalg.norm(action, 2)
        reward += self.reward_touch_object_factor * np.exp(-20. * np.linalg.norm(self._right_hand_pos - self._object_pos))
        reward += self.reward_align_direction_factor * np.dot(self._right_hand_pos - self._object_pos, 
                 self._object_pos - self._target_pos) / (np.linalg.norm(
                     self._object_pos - self._target_pos) * \
                         np.linalg.norm(self._right_hand_pos - self._object_pos))

        return reward

    # def _pre_action(self, action):
    #     # NOTE: overrides parent implementation

    #     ### TODO: reduce the number of hardcoded constants ###
    #     ### TODO: should action range scaling happen here or in RL algo? ###

    #     # action is joint vels + gripper position in range (0, 0.020833), convert to values to feed to actuator
    #     self.sim.data.ctrl[self._ref_joint_vel_actuator_indexes] = action[:7]
    #     self.sim.data.ctrl[self._ref_joint_gripper_actuator_indexes] = [-action[7], action[7]]

    #     # gravity compensation
    #     self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

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
        x_out = np.abs(self._object_pos[0]) > self.table_size[0] / 2
        y_out = np.abs(self._object_pos[1]) > self.table_size[1] / 2
        return x_out or y_out

    def _check_win(self):
        return np.allclose(self._target_pos, self._object_pos, rtol=self.win_rel_tolerance)

    @property
    def observation_space(self):
        # TODO: I am not sure if we want to add gym dependency just for observation space and action space
        # return spaces.Box(
        low=np.ones(37) * -100.
        high=np.ones(37) * 100.
        # )
        return low, high


    ####
    # Properties for objects
    ####

    @property
    def _object_pos(self):
        return self.sim.data.get_body_xpos('pusher_object') - self._pos_offset

    @property
    def _target_pos(self):
        return self.sim.model.body_pos[self.sim.model.body_name2id('pusher_target')] - self._pos_offset

    @_target_pos.setter
    def _target_pos(self, pos):
        self.sim.model.body_pos[self.sim.model.body_name2id('pusher_target')] = pos + self._pos_offset

    @property
    def _object_vel(self):
        return self.sim.data.get_body_xvelp('pusher_object')
