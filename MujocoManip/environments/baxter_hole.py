import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.baxter import BaxterEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion, array_to_string


class BaxterHoleEnv(BaxterEnv):

    def __init__(self, 
                 gripper_type='TwoFingerGripper',
                 use_eef_ctrl=False,
                 table_size=(0.8, 0.8, 0.8),
                 table_friction=None,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 reward_shaping=False,
                 gripper_visualization=False,
                 **kwargs):
        """
            @gripper_type, string that specifies the gripper type
            @use_eef_ctrl, position controller or default joint controllder
            @table_size, full dimension of the table
            @table_friction, friction parameters of the table
            @use_camera_obs, using camera observations
            @use_object_obs, using object physics states
            @camera_name, name of camera to be rendered
            @camera_height, height of camera observation
            @camera_width, width of camera observation
            @camera_depth, rendering depth
            @reward_shaping, using a shaping reward
        """
        # initialize objects of interest
        cube = RandomBoxObject(size_min=[0.02, 0.02, 0.02],
                               size_max=[0.025, 0.025, 0.025])
        #pot = DefaultPotObject()
        self.hole = DefaultHoleObject()
        #pot = cube
        self.mujoco_objects = OrderedDict()

        # settings for table top
        self.table_size = table_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # reward configuration
        self.reward_shaping = reward_shaping

        super().__init__(gripper_left='LeftTwoFingerGripper',
                         gripper_right='TwoFingerGripper',
                         use_eef_ctrl=use_eef_ctrl,
                         use_camera_obs=use_camera_obs,
                         camera_name=camera_name,
                         gripper_visualization=gripper_visualization,
                         **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(full_size=self.table_size,
                                       friction=self.table_friction)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.45 + self.table_size[0] / 2,0,0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.model.place_objects()


        self.hole_obj = self.hole.get_collision(name='hole', site=True)
        self.hole_obj.set('quat','0 0 0.707 0.707')
        self.hole_obj.set('pos','0.11 0 0.22')
        #self.hole_obj.append(joint(name='hole', type='slide'))
        self.model.merge_asset(self.hole)
        self.model.worldbody.find(".//body[@name='left_gripper_base']").append(self.hole_obj)
        #self.model.worldbody.append(self.hole_obj)

        
    def _get_reference(self):
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id('hole')

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()

        return
        self.sim.data.qpos[self._ref_gripper_left_joint_pos_indexes] = np.array([0.1, -0.1])
        for i in range(500):
            self.sim.step()
        for i in range(100):
            grip = (self.sim.data.site_xpos[self.eef_site_id])
            print("grip",grip,type(grip))
            hole_obj = self.model.worldbody.find(".//body[@name='hole']")

            hole_obj.set('pos', grip)
            self.sim.data.body_xpos[self.cube_body_id] = np.array([1,1,5])
            print("hmm",self.sim.data.body_xpos[self.cube_body_id])# = grip
            print(hole_obj)
            #self.sim.data.qfrc_applied[self._ref_gripper_left_joint_vel_indexes] = np.array([-1, 1])
            self.sim.step()

    def reward(self, action):
        reward = 0
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_size[2]

        # cube is higher than the table top above a margin
        if cube_height > table_height + 0.10:
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

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
        contact_geoms = self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in contact_geoms or \
               self.sim.model.geom_id2name(contact.geom2) in contact_geoms:
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_size[2]
        return (cube_height > table_height + 0.10)
