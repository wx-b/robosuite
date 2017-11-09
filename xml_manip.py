import xml.etree.ElementTree as ET
from mujoco_py import load_model_from_path, MjSim, MjViewer
# import xml.etree.ElementTree.Element as Element
from mujoco_py import load_model_from_path
import os.path as path
import os

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

    def save_model(self, fname):
        with open(fname, 'w') as f:
            f.write(ET.tostring(self.root, encoding='unicode'))

class MujocoModelBase(MujocoXML):
    def __init__(self):
        super().__init__('robots/sawyer/base.xml')
        self.components = []

    def merge(self, other):
        super().merge(other)
        self.components.append(other.name)

class PuhserTask(MujocoModelBase):
    def __init__(self, robot_xml, object_xml='robots/sawyer/pusher_task/pusher_object_default.xml'):
        super().__init__()
        arena_xml = MujocoXML('robots/sawyer/pusher_task/pusher_task.xml')
        self.merge(arena_xml)
        if not isinstance(robot_xml, MujocoXML):
            robot_xml = MujocoXML(robot_xml)
        self.merge_robot(robot_xml)
        if not isinstance(object_xml, MujocoXML):
            object_xml = MujocoXML(object_xml)
        self.merge_object(object_xml)

    def merge_robot(self, robot_xml):
        # re-center stuff, etc.
        self.merge(robot_xml)

    def merge_object(self, object_xml):
        pusher_object = object_xml.worldbody.find("./body[@name='pusher_object']")
        if pusher_object is None:
            print('Error: Malformed object file {}. Body "pusher_object" not found'.format(object_xml.file))
            raise ValueError
        if pusher_object.find("./joint[@name='pusher_object_free_joint']") is None:
            print('Error: Malformed object file {}. Joint "pusher_object_free_joint" not found'.format(object_xml.file))
            raise ValueError

        pusher_target_g0 = object_xml.worldbody.find("./geom[@name='pusher_target_g0']")
        pusher_target_g1 = object_xml.worldbody.find("./geom[@name='pusher_target_g1']")
        if pusher_target_g0 is None:
            print('Error: Malformed object file {}. Geom "pusher_target_g0" not found'.format(object_xml.file))
        if pusher_target_g1 is None:
            print('Error: Malformed object file {}. Geom "pusher_target_g1" not found'.format(object_xml.file))
        
        # Fix object offset
        pusher_object.set('pos', '0.5 0 -0.2')
        self.worldbody.append(pusher_object)

        pusher_target = self.worldbody.find(".//body[@name='pusher_target']")
        pusher_target.append(pusher_target_g0)
        pusher_target.append(pusher_target_g1)

        # Currently no assets or actuators from objects and targets

if __name__ == '__main__':
    task = PuhserTask('robots/sawyer/robot.xml', 'robots/sawyer/pusher_task/pusher_object_default.xml')
    model = task.get_model()
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
