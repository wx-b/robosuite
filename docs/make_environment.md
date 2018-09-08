# How to build a custom environment in Mujoco
We provide a variety of templating tools to build an environment in a modular way. Here we break down the creation of `SawyerLift` to demonstrate these functionalities. The code cited can be found in `_load_model` methods of classes `SawyerEnv` (creates the robot) and `SawyerLift` (creates the table) plus the code of `TableTopTask`.

# Modeling
Here we use RoboticsSuite's mujoco xml builders to create a robotics task
## Creating the world
All mujoco object definitions are housed in an xml. We create a `MujocoWorldBase` class to do it.
```
from RoboticsSuite.models import MujocoWorldBase

world = MujocoWorldBase()
```

## Creating the robot
The class housing the xml of a robot can be created as follows
```
from RoboticsSuite.models.robots import Sawyer

mujoco_robot = Sawyer()
```

We can add a gripper to the robot by creating a gripper instance and calling the `add_gripper` method on a robot
```
from RoboticsSuite.models.grippers import gripper_factory

gripper = gripper_factory('TwoFingerGripper')
gripper.hide_visualization()
mujoco_robot.add_gripper("right_hand", gripper)
```

To add the robot to the world, we place the robot on to a desired position and merge it into the world
```
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)
```

## Creating the table
We can initialize the `TableArena` instance that creates a table and the floorplane
```
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.16, 0, 0])
world.merge(mujoco_arena)
```

## Adding the object
For details of mujoco object, refer to [TODO](), we can create a ball and add it to the world. It is a bit more complicated than before because we are adding a free joint to the object (so it can move) and we want to place the object properly
```
from RoboticsSuite.models.objects import BoxObject
from RoboticsSuite.utils.mjcf_utils import new_joint

object_mjcf = BoxObject()
world.merge_asset(obj_mjcf)

obj = obj_mjcf.get_collision(name=obj_name, site=True)
obj.append(new_joint(name=obj_name, type="free"))
obj.set("pos", [0, 0, 0.5])
world.worldbody.append(obj)
```

# Simulation
Once we have created the object, we can obtain a [mujoco_py](https://github.com/openai/mujoco-py) model by running
```
model = world.get_model(mode="mujoco_py")
```
This is a mujoco_py `MjModel` instance than can then be used for simulation.

For example, 
```
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
sim.data.ctrl[:] = [1,2,3,4,5]
sim.step()
viewer.render()
```

For details, refer to [mujoco_py](https://github.com/openai/mujoco-py)'s documentation or look at our implementations in the environments.
