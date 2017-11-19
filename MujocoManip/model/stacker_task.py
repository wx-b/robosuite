import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.world import MujocoWorldBase

# TODO: Add a table arena to wrap around table offset

class StackerTask(MujocoWorldBase):
    def __init__(self, mujoco_robot, mujoco_objects):
        super().__init__()
        self.table_offset = np.array([0.5, 0, -0.2])
        self.object_metadata = []
        arena_xml = MujocoXML(xml_path_completion('robots/sawyer/table_arena.xml'))
        self.merge(arena_xml)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)

    def merge_robot(self, mujoco_robot):
        self.merge(mujoco_robot)

    def merge_objects(self, mujoco_objects):
        for i, mujoco_object in enumerate(mujoco_objects):
            object_name = 'stacker_object_{}'.format(i)
            joint_name = 'stacker_object_free_joint_{}'.format(i)
            target_name = 'stacker_target_{}'.format(i)

            self.merge_asset(mujoco_object)
            # Load object
            stacker_object = mujoco_object.get_full()
            stacker_object.set('name', object_name)
            object_bottom_offset = mujoco_object.get_bottom_offset()
            object_center_offset = self.table_offset - object_bottom_offset
            stacker_object.set('pos', array_to_string(object_center_offset))
            stacker_object.append(joint(name=joint_name, type='free'))
            self.worldbody.append(stacker_object)

            # Load target
            stacker_target = mujoco_object.get_visual()
            stacker_target.set('name', target_name)
            stacker_target.set('pos', array_to_string(object_center_offset))
            set_alpha(stacker_target, 0.2)
            self.worldbody.append(stacker_target)

            self.object_metadata.append({
                'object_name': object_name,
                'target_name': target_name,
                'joint_name': joint_name,
                'object_bottom_offset': mujoco_object.get_bottom_offset(),
                'object_top_offset': mujoco_object.get_top_offset(),
                'object_horizontal_radius': mujoco_object.get_horizontal_radius(),
                })
