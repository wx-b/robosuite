import numpy as np
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environment.sawyer import SawyerEnv
from MujocoManip.model import MujocoXMLObject, SingleObjectTargetTask, TableArena
from MujocoManip.model.model_util import xml_path_completion

class SawyerSingleObjectTargetEnv(SawyerEnv):
    def __init__(self, 
                mujoco_object=None,
                table_size=(0.8, 0.8, 0.8),
                table_friction=None,
                reward_lose=-1,
                reward_win=1,
                reward_action_norm_factor=0,
                reward_objective_factor=5,
                win_rel_tolerance=1e-2,
                **kwargs):
        """
            @mujoco_object(None), the object to be pushed, need that is is an MujocoObject instace
            If None, load 'object/object_ball.xml'
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
        self.mujoco_object = mujoco_object
        if self.mujoco_object is None:
            self.mujoco_object = MujocoXMLObject(xml_path_completion('object/object_ball.xml'))
        self.table_size = table_size
        self.table_friction = table_friction

        self.reward_lose=reward_lose
        self.reward_win=reward_win
        self.win_rel_tolerance = win_rel_tolerance
        self.reward_action_norm_factor=reward_action_norm_factor
        self.reward_objective_factor=reward_objective_factor
        

        super().__init__(**kwargs)
        self._pos_offset = np.copy(self.sim.data.get_site_xpos('table_top'))

    def _get_reference(self):
        super()._get_reference()
        self._ref_object_pos_indexes = self.model.get_joint_qpos_addr('object_free_joint')
        self._ref_object_vel_indexes = self.model.get_joint_qvel_addr('object_free_joint')

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.place_on([0,0,0])
        
        self.mujoco_arena = TableArena(full_size=self.table_size, friction=self.table_friction)
        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_size[0] / 2,0,0])
        
        self.target_bottom_offset = self.mujoco_object.get_bottom_offset()
        self.task = SingleObjectTargetTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_object)

        if self.debug:
            self.task.save_model('sample_combined_model.xml')
        return self.task.get_model()

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset position of target and then reset position of object


    def _pre_action(self, action):
        super()._pre_action(action)
        self.pre_action_object_target_dist = np.linalg.norm(self._target_pos[:2] - self._object_pos[:2])

    def _reward(self, action):
        reward = 0
        self.post_action_object_target_dist = np.linalg.norm(self._target_pos[:2] - self._object_pos[:2])
        
        if self._check_win():
            reward += self.reward_win
        elif self._check_lose():
            reward += self.reward_lose
        reward += self.reward_objective_factor * (self.pre_action_object_target_dist - self.post_action_object_target_dist)
        reward += self.reward_action_norm_factor * np.linalg.norm(action, 2)
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

        obs = np.concatenate([ obs,
                                object_pos_rel,
                                object_vel_rel,
                                target_pos_rel,
                                ])
        # print('hand_pos', hand_pos)
        # print('object_pos_rel', object_pos_rel)
        # print('object_vel_rel', object_vel_rel)
        # print('target_pos_rel', target_pos_rel)
        emp_means = [0.17414709140143952,
        -1.0701878703264207,
        -0.46217007456936954,
        2.037620194117416,
        -0.01178486285514388,
        0.0496273615314479,
        -0.20758468336339597,
        0.1721360833988203,
        -0.8731182199179391,
        -0.41432259207243055,
        0.859941558529584,
        -0.006380524972150775,
        0.046081454354876546,
        -0.19096883055745195,
        0.9789492615169951,
        0.4776731098300898,
        0.8245281583729884,
        -0.4394451858088656,
        0.8917387477600704,
        0.9706675311962557,
        0.9214127690547003,
        -0.0039005982309288155,
        -0.005464123011624275,
        -0.08028433098725504,
        -0.07812990994705413,
        -0.04622794393548642,
        -0.03653098354772133,
        -0.05408816873457371,
        0.2064460926917751,
        0.1418058040585972,
        -0.3817655700032953,
        0.001655948482725992,
        0.03583169767162121,
        -0.040332740377748026,
        -0.13849420934640605,
        -0.005584512632434509,
        0.09615511628971267,]
        emp_stds = [0.10997010847022229,
        0.09761819245213194,
        0.39810135955838355,
        0.26591186544912304,
        0.472650802405765,
        0.23947395910314556,
        0.34696328935591886,
        0.10736986189466033,
        0.04695346391892218,
        0.3502364397902247,
        0.06824180719436772,
        0.4434466312835077,
        0.23076648440911016,
        0.32135407792126014,
        0.022343341830748965,
        0.08537181278732828,
        0.160700161765627,
        0.25046257272089284,
        0.09009095319558855,
        0.049273452480674246,
        0.10611772416446406,
        0.3616811940429793,
        0.2968132549293374,
        0.5789497150341597,
        0.5233351130470609,
        1.966899144857867,
        0.7444694648559674,
        1.8596187909385966,
        0.05697821204150268,
        0.17594845641337575,
        0.12493683758051835,
        0.12636076256316436,
        0.1621882479578597,
        0.1282257240582145,
        0.056912682275926926,
        0.1757682453071995,
        0.1249989171101154,]
        # return (obs) / emp_stds
        return obs


    def _check_done(self):
        return self._check_lose() or self._check_win()

    def _check_lose(self):
        x_out = np.abs(self._object_pos[0]) > self.table_size[0] / 2
        y_out = np.abs(self._object_pos[1]) > self.table_size[1] / 2
        return x_out or y_out or self.t > self.horizon

    def _check_win(self):
        return np.allclose(self._target_pos, self._object_pos, rtol=self.win_rel_tolerance)

    @property
    def observation_space(self):
        low=np.ones(37) * -100.
        high=np.ones(37) * 100.
        return low, high


    ####
    # Properties for objects
    ####

    @property
    def _object_pos(self):
        return self.sim.data.get_body_xpos('object') - self._pos_offset

    @_object_pos.setter
    def _object_pos(self, pos):
        pos[0:3] += self._pos_offset
        low, high = self._ref_object_pos_indexes
        self.sim.data.qpos[low:high] = pos

    @property
    def _target_pos(self):
        return self.sim.model.body_pos[self.sim.model.body_name2id('target')] - self._pos_offset

    @_target_pos.setter
    def _target_pos(self, pos):
        self.sim.model.body_pos[self.sim.model.body_name2id('target')] = pos + self._pos_offset

    @property
    def _object_vel(self):
        return self.sim.data.get_body_xvelp('object')

