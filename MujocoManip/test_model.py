from MujocoManip.model import *
from mujoco_py import load_model_from_path, MjSim, MjViewer

mujoco_robot = SawyerRobot()
# mujoco_robot.add_gripper(TwoFingerGripper())
# mujoco_robot.add_gripper(PR2Gripper())
mujoco_robot.add_gripper(RobotiqGripper())
mujoco_robot.place_on([0,0,0])
mujoco_object = DefaultBallObject()
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.56,0,0])
task = StackerTask(mujoco_arena, mujoco_robot, [mujoco_object])
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
