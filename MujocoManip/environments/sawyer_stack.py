import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.model import MujocoObject, TableTopTask, TableArena, DefaultCylinderObject, RandomCylinderObject, RandomBoxObject, DefaultBallObject, RandomBallObject, DefaultCapsuleObject, RandomCapsuleObject

# TODO: configure table width

class SawyerStackEnv(SawyerEnv):
    def __init__(self, 
                 gripper='TwoFingerGripper',
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 **kwargs):
        """
            @table_size, the FULL size of the table 
        """
        # Handle parameters
        self.mujoco_objects = []
        self.mujoco_objects.extend([RandomBoxObject(size_min=[0.025, 0.025, 0.03], size_max=[0.05, 0.05, 0.05]) for _ in range(2)])

        self.table_size = table_size
        self.table_friction = table_friction

        super().__init__(gripper=gripper, **kwargs)

        # some bookkeeping
        self.max_horizontal_radius = max([di['object_horizontal_radius'] for di in self.object_metadata])
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

        self.object_names = [di['object_name'] for di in self.object_metadata]
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        # self.sim.model._geom_name2id # keys for named shit

        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names]

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
     
        self.task = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)

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

    def reward(self, action):
        reward = 0
        #TODO(yukez): implementing a stacking reward
        return reward

    def _get_observation(self):
        """
            Adds hand_position, hand_velocity or 
            (current_position, current_velocity, target_velocity) of all targets
        """
        di = super()._get_observation()
        all_observations = []

        hand_pos = self._right_hand_pos
        hand_vel = self._right_hand_vel
        all_observations += [hand_pos, hand_vel]

        for i in range(self.n_objects):
            all_observations += [self._object_pos(i),
                                self._object_vel(i),
                                self._target_pos(i)]
            
        di['low-level'] = np.concatenate(all_observations)
        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False

        ### TODO: try 0:self.sim.data.ncon and :
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            # print("geom1: {}".format(self.sim.model.geom_id2name(contact.geom1)))
            # print("geom2: {}".format(self.sim.model.geom_id2name(contact.geom2)))
            if self.sim.model.geom_id2name(contact.geom1) in self.finger_names or \
               self.sim.model.geom_id2name(contact.geom2) in self.finger_names:
                collision = True
                # print("geom1: {}".format(self.sim.model.geom_id2name(contact.geom1)))
                # print("geom2: {}".format(self.sim.model.geom_id2name(contact.geom2)))
                break
        return collision

    def _check_terminated(self):
        #TODO(yukez): define termination conditions
        return False

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

    # Properties for objects

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
