"""
Defines GripperTester that is used to test the physical properties of various grippers
"""
import xml.etree.ElementTree as ET
from mujoco_py import MjSim, MjViewer

from RoboticsSuite.models.world import MujocoWorldBase
from RoboticsSuite.models.arenas.table_arena import TableArena
from RoboticsSuite.utils.mjcf_utils import new_actuator, new_joint


class GripperTester:
    """
    A class that is used to test gripper
    """

    def __init__(
        self, gripper, pos, quat, gripper_low_pos, gripper_high_pos, render=True
    ):
        """
        Initializes world and gripper positioning

        Args:
            gripper: A Gripper instance
            pos: position to place the gripper
                 e.g. '0 0 0.3'
            quat: rotation to apply to gripper
                  e.g. '0 0 1 0' to flip z axis
            gripper_low_pos (float): controls the gipper y position,
                                     larger -> higher
            gripper_high_pos (float): controls the gipper y high position
                                      larger -> higher, must be larger
                                      than gripper_low_pos
            render:
        """
        world = MujocoWorldBase()
        # Add a table
        arena = TableArena(table_full_size=(0.4, 0.4, 0.1))
        world.merge(arena)

        # Add a gripper
        # gripper = TwoFingerGripper()
        self.gripper = gripper
        gripper_body = ET.Element("body")
        for body in gripper.worldbody:
            gripper_body.append(body)
        gripper_body.set("pos", pos)
        gripper_body.set("quat", quat)  # flip z
        gripper_body.append(
            new_joint(name="gripper_z_joint", type="slide", axis="0 0 -1", damping="50")
        )
        world.merge(gripper, merge_body=False)
        world.worldbody.append(gripper_body)
        world.actuator.append(
            new_actuator(
                joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"
            )
        )

        # TODO: add object
        # # Add an object for grasping
        # mujoco_object = BoxObject(size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=1).get_collision()
        # mujoco_object.append(joint(name='object_free_joint', type='free'))
        # mujoco_object.set('pos', '0 0 0.11')
        # geoms = mujoco_object.findall('./geom')
        # for geom in geoms:
        #     if geom.get('contype'):
        #         pass
        #     geom.set('name', 'object')
        #     geom.set('density', '10000') # 1000 for water
        # world.worldbody.append(mujoco_object)

        # TODO: add reference object
        # x_ref = BoxObject(size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1]).get_visual()
        # x_ref.set('pos', '0.2 0 0.105')
        # world.worldbody.append(x_ref)
        # y_ref = BoxObject(size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1]).get_visual()
        # y_ref.set('pos', '0 0.2 0.105')
        # world.worldbody.append(y_ref)

        self.world = world
        self.render = render
        self.simulation_ready = False
        self.cur_step = 0
        if gripper_low_pos > gripper_high_pos:
            raise ValueError(
                "gripper_low_pos {} is larger "
                "than gripper_high_pos {}".format(gripper_low_pos, gripper_high_pos)
            )
        self.gripper_low_pos = gripper_low_pos
        self.gripper_high_pos = gripper_high_pos

    def start_simulation(self):
        """
            Starts simulation of the test world
        """
        # Start simulation
        model = self.world.get_model(mode="mujoco_py")

        self.sim = MjSim(model)
        if self.render:
            self.viewer = MjViewer(self.sim)
        self.sim_state = self.sim.get_state()

        # For gravity correction
        gravity_corrected = ["gripper_z_joint"]
        self._gravity_corrected_qvels = [
            self.sim.model.get_joint_qvel_addr(x) for x in gravity_corrected
        ]

        self.gripper_z_id = self.sim.model.actuator_name2id("gripper_z")
        self.gripper_z_is_low = False

        self.gripper_joint_ids = [
            self.sim.model.actuator_name2id("gripper_" + x) for x in self.gripper.joints
        ]
        self.gripper_open_action = self.gripper.format_action([1])  # [0.0115, -0.0115]
        self.gripper_closed_action = self.gripper.format_action(
            [-1]
        )  # [-0.020833, 0.020833]
        self.gripper_is_closed = True

        self.reset()
        self.simulation_ready = True

    def reset(self):
        """
            Resets the simulation to the initial state
        """
        self.sim.set_state(self.sim_state)
        self.cur_step = 0

    def step(self):
        """
        Forward the simulation by one timestep

        Raises:
            RuntimeError: if start_simulation is not yet called.
        """
        if not self.simulation_ready:
            raise RuntimeError("Call start_simulation before calling step")
        if self.gripper_z_is_low:
            self.sim.data.ctrl[self.gripper_z_id] = self.gripper_low_pos
        else:
            self.sim.data.ctrl[self.gripper_z_id] = self.gripper_high_pos
        if self.gripper_is_closed:
            self.sim.data.ctrl[self.gripper_joint_ids] = self.gripper_closed_action
        else:
            self.sim.data.ctrl[self.gripper_joint_ids] = self.gripper_open_action
        self._apply_gravity_compensation()

    def _apply_gravity_compensation(self):
        self.sim.data.qfrc_applied[
            self._gravity_corrected_qvels
        ] = self.sim.data.qfrc_bias[self._gravity_corrected_qvels]
        self.sim.step()
        if self.render:
            self.viewer.render()
        self.cur_step += 1

    def loop(self, T=300, total_iters=1):
        """
        Performs lower, grip, raise and release actions of a gripper,
                each separated with T timesteps
        Args:
            T: The interfal between two gripper actions
            total_iters: Iterations to perform before exiting
        """
        seq = [(False, False), (True, False), (True, True), (False, True)]
        step = 0
        cur_plan = 0
        for cur_iter in range(total_iters):
            for cur_plan in seq:
                self.gripper_z_is_low, self.gripper_is_closed = cur_plan
                for step in range(T):
                    self.step()

        # These commented code are for contact testing
        # while True:
        #     # if step % 100 == 0:
        #     #     pass
        #     #     # print('step: {}'.format(step))
        #     # if step % 100 == 0:
        #     #     pass
        #     #     # for contact in sim.data.contact[0:sim.data.ncon]:
        #         #     if sim.model.geom_id2name(contact.geom1) == 'floor' \
        #         #         and sim.model.geom_id2name(contact.geom2) == 'floor':
        #         #         continue
        #         # if not gripper_is_closed and sim.model.geom_id2name(contact.geom1) == 'r_finger_g0' and sim.model.geom_id2name(contact.geom2) == 'object':
        #         # if sim.model.geom_id2name(contact.geom1) == 'r_finger_g0' and sim.model.geom_id2name(contact.geom2) == 'object':
        #         # print("geom1: {}, geom2: {}".format(sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)))
        #         # print("contact id {}".format(id(contact)))
        #         # print("friction: {}".format(contact.friction))
        #         # print("normal: {}".format(contact.frame[0:3]))
        #     if step % T == 0:
        #         plan = seq[int(step / T) % len(seq)]
        #         gripper_z_is_low, gripper_is_closed = plan
        #         self.gripper_z_is_low = gripper_z_is_low
        #         self.gripper_is_closed = gripper_is_closed
        #         # print('changing plan: gripper low: {}, gripper closed {}'.format(gripper_z_is_low, gripper_is_closed))
        #     self.step()
        # step += 1
