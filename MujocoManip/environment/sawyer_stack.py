import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model import MujocoObject, StackerTask, TableArena, DefaultCylinderObject, RandomCylinderObject, RandomBoxObject, DefaultBallObject, RandomBallObject, DefaultCapsuleObject, RandomCapsuleObject

# TODO: configure table width

class SawyerStackEnv(SawyerEnv):
    def __init__(self, 
                gripper='TwoFingerGripper',
                mujoco_objects=None,
                n_mujoco_objects=10,
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
            # self.mujoco_objects = [RandomCapsuleObject(size_max=[0.025, 0.03], size_min=[0.01, 0.01]) for _ in range(1)]
            # self.mujoco_objects.extend([RandomCylinderObject(size_max=[0.025, 0.05], size_min=[0.01, 0.01]) for _ in range(1)])
            # self.mujoco_objects.extend([RandomBoxObject(size_max=[0.025, 0.025, 0.05], size_min=[0.01, 0.01, 0.01]) for _ in range(1)])
            # self.mujoco_objects.extend([RandomBallObject(size_max=[0.03], size_min=[0.02]) for _ in range(1)])
            self.mujoco_objects = [RandomCapsuleObject(size_max=[0.025, 0.03], size_min=[0.01, 0.01]) for _ in range(3)]
            self.mujoco_objects.extend([RandomCylinderObject(size_max=[0.025, 0.05], size_min=[0.01, 0.01]) for _ in range(5)])
            self.mujoco_objects.extend([RandomBoxObject(size_max=[0.025, 0.025, 0.05], size_min=[0.01, 0.01, 0.01]) for _ in range(5)])
            self.mujoco_objects.extend([RandomBallObject(size_max=[0.03], size_min=[0.02]) for _ in range(3)])
        self.n_mujoco_objects = len(self.mujoco_objects)
        self.table_size = table_size
        self.table_friction = table_friction

        self.reward_lose=reward_lose
        self.reward_win=reward_win
        self.reward_action_norm_factor=reward_action_norm_factor
        self.reward_objective_factor=reward_objective_factor
        self.win_rel_tolerance = win_rel_tolerance

        super().__init__(gripper=gripper, **kwargs)

        # some bookkeeping
        self.max_horizontal_radius = max([di['object_horizontal_radius'] for di in self.object_metadata])
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

        self.object_names = [di['object_name'] for di in self.object_metadata]
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]

        # id of grippers for contact checking
        self.l_finger_geom_id = self.sim.model.geom_name2id("r_gripper_l_finger")
        self.r_finger_geom_id = self.sim.model.geom_name2id("r_gripper_r_finger")
        self.finger_names = ["r_gripper_l_finger", "r_gripper_r_finger"]

        # self.sim.data.contact # list, geom1, geom2
        # self.sim.model._geom_name2id # keys for named shit

        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
     
        self.task = StackerTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)

        self.object_metadata = self.task.object_metadata
        self.n_objects = len(self.object_metadata)

        self.model = self.task
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self._ref_object_pos_indexes = []
        self._ref_object_vel_indexes = []
        for di in self.object_metadata:
            self._ref_object_pos_indexes.append(self.sim.model.get_joint_qpos_addr(di['joint_name']))
            self._ref_object_vel_indexes.append(self.sim.model.get_joint_qvel_addr(di['joint_name']))
    
    def _reset_internal(self):
        super()._reset_internal()
        self.model.place_objects()

    # def _place_targets(self):
    #     object_ordering = [x for x in range(self.n_objects)]
    #     np.random.shuffle(object_ordering)
    #     # rest position of target
    #     target_x = np.random.uniform(high=self.table_size[0]/2 - self.max_horizontal_radius, low=-1 * (self.table_size[0]/2 - self.max_horizontal_radius))
    #     target_y = np.random.uniform(high=self.table_size[1]/2 - self.max_horizontal_radius, low=-1 * (self.table_size[1]/2 - self.max_horizontal_radius))

    #     contact_point = np.array([target_x, target_y, 0])

    #     for index in object_ordering:
    #         di = self.object_metadata[index]
    #         self._set_target_pos(index, contact_point - di['object_bottom_offset'])
    #         contact_point = contact_point - di['object_bottom_offset'] + di['object_top_offset']
        
    # def _place_objects(self):
    #     """
    #     Place objects randomly until no more collisions or max iterations hit.
    #     """
    #     placed_objects = []
    #     for index in range(self.n_objects):
    #         di = self.object_metadata[index]
    #         horizontal_radius = di['object_horizontal_radius']
    #         bottom_offset = di['object_bottom_offset']
    #         success = False
    #         for i in range(100000): # 1000 retries
    #             object_x = np.random.uniform(high=self.table_size[0]/2 - horizontal_radius, low=-1 * (self.table_size[0]/2 - horizontal_radius))
    #             object_y = np.random.uniform(high=self.table_size[1]/2 - horizontal_radius, low=-1 * (self.table_size[1]/2 - horizontal_radius))
    #             # objects cannot overlap
    #             location_valid = True
    #             for _, (x, y, z), r in placed_objects:
    #                 if np.linalg.norm([object_x - x, object_y - y], 2) <= r + horizontal_radius:
    #                     location_valid = False
    #                     break
    #             if not location_valid: # bad luck, reroll
    #                 continue
    #             # location is valid, put the object down
    #             # quarternions, later we can add random rotation

    #             pos = np.array([object_x, object_y, 0])
    #             pos -= bottom_offset
    #             placed_objects.append((index, pos, horizontal_radius))
    #             success = True
    #             break
    #         if not success:
    #             raise RandomizationError('Cannot place all objects on the desk')

    #     # with self.physics.reset_context():
    #     for index, pos, r in placed_objects:
    #         self._set_object_pos(index, pos)
    #         self._set_object_vel(index, np.zeros(3))
    #     self.physics.forward()
    #         # pos = np.array([object_x, object_y, 0])

    
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

    def _get_observation(self):
        obs = super()._get_observation()
        all_observations = [obs]

        hand_pos = self._right_hand_pos
        hand_vel = self._right_hand_vel
        for i in range(self.n_objects):
            all_observations += [self._object_pos(i) - hand_pos,
                                self._object_vel(i) - hand_vel,
                                self._target_pos(i) - hand_pos]

        # add in collision information
        collision = False
        for contact in self.sim.data.contact:
            print("geom1: {}".format(self.sim.model.geom_id2name(contact.geom1)))
            print("geom2: {}".format(self.sim.model.geom_id2name(contact.geom2)))
            if self.sim.model.geom_id2name(contact.geom1) in self.finger_names or \
               self.sim.model.geom_id2name(contact.geom2) in self.finger_names:
                collision = True
                print("GOT COLLISION")
                break
        all_observations += [[int(collision)]]

        return np.concatenate(all_observations)
        
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

    def _gripper_visualization(self):
        """
        Do any needed visualization here.

        Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to nearest object

        ### TODO: we could probably clean this up. Also should probably include the table site. ###

        # find closest object
        square_dist = lambda x : np.sum(np.square(x - self.sim.data.get_site_xpos('grip_site')))
        dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
        dists[self.eef_site_id] = np.inf # make sure we don't pick the same site
        dists[self.eef_cylinder_id] = np.inf
        ob_dists = dists[self.object_site_ids] # filter out object sites we care about
        min_dist = np.min(ob_dists)
        ob_id = np.argmin(ob_dists)
        # ob_name = self.sim.model.site_id2name(ob_id)
        ob_name = self.object_names[ob_id]
        # print("closest object is {} at distance {}".format(ob_name, min_dist))

        # set RGBA for the EEF site here
        max_dist = 0.1 
        scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
        rgba = np.zeros(4)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        rgba[3] = 0.5

        # rgba = np.random.rand(4)
        # rgba[-1] = 0.5
        self.sim.model.site_rgba[self.eef_site_id] = rgba

    ####
    # Properties for objects
    ####

    def _object_pos(self, i):
        object_name = self.object_metadata[i]['object_name']
        return self.sim.data.get_body_xpos(object_name) - self._pos_offset

    def _set_object_pos(self, i, pos):
        low, high = self._ref_object_pos_indexes[i]
        self.sim.data.qpos[low:high] = pos + self._pos_offset

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
