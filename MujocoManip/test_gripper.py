from MujocoManip.model import *
from mujoco_py import load_model_from_path, MjSim, MjViewer
import xml.etree.ElementTree as ET
from MujocoManip.model.model_util import *

world = MujocoWorldBase()

# Add a table
arena = TableArena(full_size=(0.4, 0.4, 0.1))
world.merge(arena)

# Add a gripper
gripper = TwoFingerGripper()
gripper_body = ET.Element('body')
for body in gripper.worldbody:
    gripper_body.append(body)
gripper_body.set('pos', '0 0 0.3')
gripper_body.set('quat', '0 0 1 0') # flip z
gripper_body.append(joint(name='gripper_z_joint', type='slide', axis='0 0 1', damping='50'))
world.merge(gripper, merge_body=False)
world.worldbody.append(gripper_body)
world.actuator.append(actuator(joint='gripper_z_joint', act_type='position', name='gripper_z', kp='100'))

# Add an object for grasping
mujoco_object = BoxObject(size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 1]).get_full()
mujoco_object.append(joint(name='object_free_joint', type='free'))
mujoco_object.set('pos', '0 0 0.11')
world.worldbody.append(mujoco_object)

# Start simulation
model = world.get_model(mode='mujoco_py')

sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()

# For gravity correction
gravity_corrected = ['gripper_z_joint']
_ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

gripper_z_id = sim.model.actuator_name2id('gripper_z')
gripper_z_low = 0.07
gripper_z_high = -0.02
gripper_z_is_low = False

gripper_joint_ids = [sim.model.actuator_name2id('gripper_' + x) for x in gripper.joints]
gripper_open = [0.0115, -0.0115]
gripper_closed = [-0.020833, 0.020833]
gripper_is_closed = True

seq = [(False, False), (True, False), (True, True), (False, True), (False, False)]

sim.set_state(sim_state)
step = 0
T = 1000
while True:
    if step % 100 == 0:
        print('step: {}'.format(step))
    if step % T == 0:
        plan = seq[int(step / T) % len(seq)]
        gripper_z_is_low, gripper_is_closed = plan
        print('changing plan: gripper low: {}, gripper closed {}'.format(gripper_z_is_low, gripper_is_closed))
    if gripper_z_is_low:
        sim.data.ctrl[gripper_z_id] = gripper_z_low
    else:
        sim.data.ctrl[gripper_z_id] = gripper_z_high
    if gripper_is_closed:
        sim.data.ctrl[gripper_joint_ids] = gripper_closed
    else:
        sim.data.ctrl[gripper_joint_ids] = gripper_open
    sim.step()
    sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]
    viewer.render()
    step += 1