import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model import MujocoObject, StackerTask, RandomBoxObject, TableArena

# TODO: configure table width

class SawyerStackEnv(SawyerEnv):
    def __init__(self, 
                gripper='TwoFingerGripper',
                mujoco_objects=None,
                n_mujoco_objects=5,
                table_size=(0.8, 0.8, 0.8),
                table_friction=None,
                reward_lose=-2,
                reward_win=2,
                reward_action_norm_factor=-0.1,
                reward_objective_factor=0.1,
                win_rel_tolerance=1e-2,
                **kwargs):
        """
            @mujoco_objects(None), the objects to be stacked, need that is is an MujocoObject instace
            If None, load 'object/object_ball.xml', generate n_mujoco_objects random boxes
            @n_mujoco_objects(int), number of objects to be stacked, need an array of MujocoObject instaces
                ignored when mujoco_objects is not None
            @table_size, the FULL size of the table 
            @friction: friction coefficient of table, None for mujoco default
            @reward_win: reward given to the agent when it completes the task
            @reward_lose: reward given to the agent when it fails the task
            @reward_action_norm_factor: reward scaling factor that penalizes large actions
            @reward_objective_factor: reward scaling factor for being close to completing the objective
            @win_rel_tolerance: relative tolerance between object and target location 
                used when deciding if the agent has completed the task
        """
        # Handle parameters
        self.mujoco_objects = mujoco_objects
        if self.mujoco_objects is None:
            self.mujoco_objects = [RandomBoxObject() for i in range(n_mujoco_objects)]
        self.n_mujoco_objects = len(self.mujoco_objects)
        self.table_size = table_size
        self.table_friction = table_friction

        self.reward_lose=reward_lose
        self.reward_win=reward_win
        self.reward_action_norm_factor=reward_action_norm_factor
        self.reward_objective_factor=reward_objective_factor
        self.win_rel_tolerance = win_rel_tolerance

        super().__init__(gripper=gripper, **kwargs)
        self.max_horizontal_radius = max([di['object_horizontal_radius'] for di in self.object_metadata])
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
     
        self.task = StackerTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)

        if self.debug:
            self.task.save_model('sample_combined_model.xml')
        self.object_metadata = self.task.object_metadata
        self.n_objects = len(self.object_metadata)
        return self.task.get_model()

    def _get_reference(self):
        super()._get_reference()
        self._ref_object_pos_indexes = []
        self._ref_object_vel_indexes = []
        for di in self.object_metadata:
            self._ref_object_pos_indexes.append(self.model.get_joint_qpos_addr(di['joint_name']))
            self._ref_object_vel_indexes.append(self.model.get_joint_qvel_addr(di['joint_name']))

    
    def _reset_internal(self):
        super()._reset_internal()
        self._place_targets()
        self._place_objects()

    def _place_targets(self):
        object_ordering = [x for x in range(self.n_objects)]
        np.random.shuffle(object_ordering)
        # rest position of target
        target_x = np.random.uniform(high=self.table_size[0]/2 - self.max_horizontal_radius, low=-1 * (self.table_size[0]/2 - self.max_horizontal_radius))
        target_y = np.random.uniform(high=self.table_size[1]/2 - self.max_horizontal_radius, low=-1 * (self.table_size[1]/2 - self.max_horizontal_radius))

        contact_point = np.array([target_x, target_y, 0])

        for index in object_ordering:
            di = self.object_metadata[index]
            self._set_target_pos(index, contact_point - di['object_bottom_offset'])
            contact_point = contact_point - di['object_bottom_offset'] + di['object_top_offset']
        
    def _place_objects(self):
        placed_objects = []
        for index in range(self.n_objects):
            di = self.object_metadata[index]
            horizontal_radius = di['object_horizontal_radius']
            success = False
            for i in range(1000): # 1000 retries
                object_x = np.random.uniform(high=self.table_size[0]/2 - horizontal_radius, low=-1 * (self.table_size[0]/2 - horizontal_radius))
                object_y = np.random.uniform(high=self.table_size[1]/2 - horizontal_radius, low=-1 * (self.table_size[1]/2 - horizontal_radius))
                # objects cannot overlap
                location_valid = True
                for x, y, r in placed_objects:
                    if np.linalg.norm([object_x - x, object_y - y], 2) < r + horizontal_radius:
                        location_valid = False
                        break
                if not location_valid: # bad luck, reroll
                    continue
                # location is valid, put the object down
                # quarternions, later we can add random rotation
                pos = np.array([object_x, object_y, 0, 0, 0, 0, 0])
                self._set_object_pos(index, pos)
                self._set_object_vel(index, np.zeros(6))
                placed_objects.append((object_x, object_y, horizontal_radius))
                success = True
                break
            if not success:
                raise RandomizationError('Cannot place all objects on the desk')
    
    def _reward(self, action):
        reward = 0
        if self._check_win():
            reward += self.reward_win
        elif self._check_lose():
            reward += self.reward_lose
        # TODO: set a good action penalty coefficient
        # distance of object
        for i in range(self.n_objects):
            reward += self.reward_objective_factor * np.exp(-2. * np.linalg.norm(self._target_pos(i) - self._object_pos(i), 2))
        # Action strength
        reward += self.reward_action_norm_factor * np.linalg.norm(action, 2)
        return reward

    # def _pre_action(self, action):
    #     print('called')
    #     import pdb; pdb.set_trace()
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
        all_observations = [obs]

        hand_pos = self._right_hand_pos
        hand_vel = self._right_hand_vel
        for i in range(self.n_objects):
            all_observations += [self._object_pos(i) - hand_pos,
                                self._object_vel(i) - hand_vel,
                                self._target_pos(i) - hand_pos]

        return np.concatenate(all_observations)


    def _check_done(self):
        return self._check_lose() or self._check_win()

    def _check_lose(self):
        object_z = np.concatenate([self._object_pos(i)[2:3] for i in range(self.n_objects)])
        # Object falls off the table
        return np.min(object_z) < 0

    def _check_win(self):
        object_pos = np.concatenate([self._object_pos(i) for i in range(self.n_objects)])
        target_pos = np.concatenate([self._target_pos(i) for i in range(self.n_objects)])
        return np.allclose(object_pos, target_pos, rtol=self.win_rel_tolerance)

    @property
    def observation_space(self):
        # TODO: I am not sure if we want to add gym dependency just for observation space and action space
        # return spaces.Box(
        low=np.ones(28 + 9 * self.n_objects) * -100.
        high=np.ones(28 + 9 * self.n_objects) * 100.
        # )
        return low, high

    ####
    # Properties for objects
    ####

    def _object_pos(self, i):
        object_name = self.object_metadata[i]['object_name']
        return self.sim.data.get_body_xpos(object_name) - self._pos_offset

    def _set_object_pos(self, i, pos):
        pos[0:3] += self._pos_offset
        low, high = self._ref_object_pos_indexes[i]
        self.sim.data.qpos[low:high] = pos

    def _object_vel(self, i):
        object_name = self.object_metadata[i]['object_name']
        return self.sim.data.get_body_xvelp(object_name)

    def _set_object_vel(self, i, vel):
        low, high = self._ref_object_vel_indexes[i]
        self.sim.data.qvel[low:high] = vel

    def _target_pos(self, i):
        target_name = self.object_metadata[i]['target_name']
        return self.sim.model.body_pos[self.sim.model.body_name2id(target_name)] - self._pos_offset

    def _set_target_pos(self, i, pos):
        target_name = self.object_metadata[i]['target_name']
        self.sim.model.body_pos[self.sim.model.body_name2id(target_name)] = pos + self._pos_offset
