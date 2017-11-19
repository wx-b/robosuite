from MujocoManip.model.gripper import MujocoGripper 
from MujocoManip.model.robot import MujocoRobot
from MujocoManip.model.mujoco_object import MujocoXMLObject, BoxObject, RandomBoxObject
from MujocoManip.model.pusher_task import PusherTask
from MujocoManip.model.stacker_task import StackerTask
import os.path
assets_root = os.path.join(os.path.dirname(__file__), 'assets')