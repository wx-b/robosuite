import xml.etree.ElementTree as ET
from mujoco_py import load_model_from_path, MjSim, MjViewer
# import xml.etree.ElementTree.Element as Element
import os.path as path
import os
import copy
import numpy as np
import xml.dom.minidom

def array_to_string(array):
    return ' '.join(['{}'.format(x) for x in array])

def string_to_array(string):
    return np.array([float(x) for x in string.split(' ')])

def joint(**kwargs):
    element = ET.Element('joint', attrib=kwargs)
    return element
    
class MujocoXML(object):
    def __init__(self, fname):
        self.file = fname
        self.folder = path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.name = self.root.get('model')
        self.worldbody = self.create_default_element('worldbody')
        self.actuator = self.create_default_element('actuator')
        self.asset = self.create_default_element('asset')
        # self.config = self.create_default_element('surreal_config')

    def create_default_element(self,name):
        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    # Default merge method
    def merge(self, other):
        if not isinstance(other, MujocoXML):
            print('Error: {} is not a MujocoXML instance.'.format(type(other)))
            raise TypeError
        for body in other.worldbody:
            self.worldbody.append(body)
        for one_asset in other.asset:
            self.asset.append(one_asset)
        for one_actuator in other.actuator:
            self.actuator.append(one_actuator)
        # self.config.append(other.config)

    def get_model(self):
        tempfile = path.join(self.folder, '.surreal_temp_model.xml')
        with open(tempfile, 'w') as f:
            f.write(ET.tostring(self.root, encoding='unicode'))
        model = load_model_from_path(tempfile)
        os.remove(tempfile)
        return model

    def save_model(self, fname, pretty=False):
        
        with open(fname, 'w') as f:
            xml_str = ET.tostring(self.root, encoding='unicode')
            if pretty:
                # TODO: get a good pretty print library
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl='')
            f.write(xml_str)

### Base class for all objects
class MujocoObject(MujocoXML):
    def __init__(self, fname):
        super().__init__(fname)

    def get_bottom_offset(self):
        bottom_site = self.worldbody.find("./site[@name='bottom_site']")
        return string_to_array(bottom_site.get('pos'))

    def get_top_offset(self):
        top_site = self.worldbody.find("./site[@name='top_site']")
        return string_to_array(top_site.get('pos'))

    # returns a copy, Returns xml body node
    def get_collision(self):
        collision = copy.deepcopy(self.worldbody.find("./body[@name='collision']"))
        collision.attrib.pop('name')
        return collision

    # returns a copy, Returns xml body node
    def get_visual(self):
        visual = copy.deepcopy(self.worldbody.find("./body[@name='visual']"))
        visual.attrib.pop('name')
        return visual

    # returns a copy, xml of collision plus visual
    def get_full(self):
        collision = self.get_collision()
        visual = self.get_visual()
        collision.append(visual)
        return collision


### Base class for all robots
### Since we will only be having sawyer for a while, all sawyer methods are put in here.
class MujocoRobot(MujocoXML):
    def __init__(self, fname):
        super().__init__(fname)
        self.right_hand = self.worldbody.find(".//body[@name='right_hand']")
        self.has_gripper = False
        if self.right_hand is None:
            print('Error: body with name "right_hand" not found.')
            raise ValueError

    def add_gripper(self, gripper):
        if not isinstance(gripper, MujocoGripper):
            print('Error: {} is not a MujocoGripper instance.'.format(type(other)))
            raise TypeError
        for actuator in gripper.actuator:
            self.actuator.append(actuator)
        for asset in gripper.asset:
            self.asset.append(asset)
        for body in gripper.worldbody:
            self.right_hand.append(body)
        self.has_gripper = True

### Base class for grippers
class MujocoGripper(MujocoXML):
    def __init__(self, fname):
        super().__init__(fname)

### Base class to inherit all mujoco worlds from
class MujocoWorldBase(MujocoXML):
    def __init__(self):
        super().__init__('robots/sawyer/base.xml')

    def merge_asset(self, other):
        for asset in other.asset:
            asset_name = asset.get('name')
            asset_type = asset.tag
            # Avoids duplication
            if self.asset.find('./{}[@name={}]'.format(asset_type, asset_name)) is None:
                self.asset.append(asset)


class PusherTask(MujocoWorldBase):
    def __init__(self, mujoco_robot, mujoco_object):
        super().__init__()
        self.table_offset = np.array([0.5, 0, -0.2])
        arena_xml = MujocoXML('robots/sawyer/pusher_task/pusher_task.xml')
        self.merge(arena_xml)
        self.merge_robot(mujoco_robot)
        self.merge_object(mujoco_object)

    def merge_robot(self, mujoco_robot):
        self.merge(mujoco_robot)

    def merge_object(self, mujoco_object):
        self.merge_asset(mujoco_object)
        # Load object
        pusher_object = mujoco_object.get_full()
        pusher_object.set('name', 'pusher_object')
        object_bottom_offset = mujoco_object.get_bottom_offset()
        object_center_offset = self.table_offset - object_bottom_offset
        pusher_object.set('pos', array_to_string(object_center_offset))
        pusher_object.append(joint(name='pusher_object_free_joint', type='free'))
        self.worldbody.append(pusher_object)

        # Load target
        pusher_target = mujoco_object.get_visual()
        pusher_target.set('name', 'pusher_target')
        pusher_target.set('pos', array_to_string(object_center_offset))
        self.worldbody.append(pusher_target)

class StackerTask(MujocoWorldBase):
    def __init__(self, mujoco_robot, mujoco_objects):
        super().__init__()
        self.table_offset = np.array([0.5, 0, -0.2])
        arena_xml = MujocoXML('robots/sawyer/pusher_task/pusher_task.xml')
        self.merge(arena_xml)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)

    def merge_robot(self, mujoco_robot):
        self.merge(mujoco_robot)

    def merge_object(self, mujoco_objects):
        for i, mujoco_object in enumerate(mujoco_objects):
            self.merge_asset(mujoco_object)
            # Load object
            stacker_object = mujoco_object.get_full()
            stacker_object.set('name', 'pusher_object')
            object_bottom_offset = mujoco_object.get_bottom_offset()
            object_center_offset = self.table_offset - object_bottom_offset
            stacker_object.set('pos', array_to_string(object_center_offset))
            stacker_object.append(joint(name='pusher_object_free_joint', type='free'))
            self.worldbody.append(pusher_object)

            # Load target
            stacker_target = mujoco_object.get_visual()
            stacker_target.set('name', 'pusher_target')
            stacker_target.set('pos', array_to_string(object_center_offset))
            self.worldbody.append(pusher_target)


if __name__ == '__main__':
    mujoco_robot = MujocoRobot('robots/sawyer/robot.xml')
    mujoco_robot.add_gripper(MujocoGripper('robots/sawyer/gripper.xml'))
    mujoco_object = MujocoObject('robots/sawyer/object_box.xml')
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

    # import pdb; pdb.set_trace()
