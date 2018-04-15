import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.sawyer import SawyerEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion


class SawyerLiftEnv(SawyerEnv):

    def __init__(self, 
                 gripper='TwoFingerGripper',
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 camera_height=256,
                 camera_width=256,
                 camera_depth=True,
                 **kwargs):
        """
            @table_size, the full dimension of the table 
        """
        # initialize objects of interest
        cube = RandomBoxObject(size_min=[0.025, 0.025, 0.03],
                               size_max=[0.05, 0.05, 0.05])
        self.mujoco_objects = OrderedDict([('cube', cube)])

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # settings for camera observation
        self.use_camera_obs = use_camera_obs
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(gripper=gripper, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id('cube')

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

    def reward(self, action):
        reward = 0
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_size[2]
        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.10:
            reward = 1.0
        return reward

    def _get_observation(self):
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of object
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            cube_quat = self.sim.data.body_xquat[self.cube_body_id]
            di['cube_pos'] = cube_pos
            di['cube_quat'] = cube_quat

            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            di['gripper_to_cube'] = gripper_site_pos - cube_pos

        return di

    def _check_contact(self):
        """
            Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.finger_names or \
               self.sim.model.geom_id2name(contact.geom2) in self.finger_names:
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
            Returns True if task is successfully completed
        """
        #TODO(yukez): define termination conditions
        return False
