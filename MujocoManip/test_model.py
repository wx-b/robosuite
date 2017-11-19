from MujocoManip.model import *

mujoco_robot = MujocoRobot('model/assets/robot/sawyer/robot.xml')
mujoco_robot.add_gripper(MujocoGripper('model/assets/gripper/two_finger_gripper.xml'))
mujoco_object = MujocoXMLObject('model/assets/object/object_box.xml')
task = PusherTask(mujoco_robot, mujoco_object)
model = task.get_model()
# task.save_model('sample_combined_model.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

sim_state = sim.get_state()
while True:
    sim.set_state(sim_state)

    for i in range(2000):
        #sim.data.ctrl[:] = np.random.rand(7) * 3
        # print(sim.data.ctrl[:])

        sim.step()
        viewer.render()
