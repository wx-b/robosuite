import copy
import time
import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML, MujocoXMLFile
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import *


class MujocoObject(MujocoXML):
    """
        Base class for all objects
        We use Mujoco Objects to implement all objects that 
        1) may appear for multiple times in a task
        2) can be swapped between different tasks
        Typical methods return copy so the caller can all joints/attributes as wanted
    """
    def __init__(self, name=None):
        """creates a root directory <mujoco model="sawyer">"""
        if name is None:
            name = 'mujoco_object_{}'.format(np.random.randint(0,1000))
        self.site_appended = False
        super().__init__(name)

    def get_bottom_offset(self):
        """
            Returns vector from object center to object bottom
            Helps us put objects on a surface
            returns numpy array
            e.g. return np.array([0, 0, -2])
        """
        raise NotImplementedError
        

    def get_top_offset(self):
        """
            Returns vector from object center to object top
            Helps us put other objects on this object
            returns numpy array
            e.g. return np.array([0, 0, 2])
        """
        raise NotImplementedError

    def get_horizontal_radius(self):
        """
            Returns scalar 
            If object a,b has horizontal distance d
                a.get_horizontal_radius() + b.get_horizontal_radius() < d 
                should mean that a, b has no contact 
            Helps us put objects programmatically without them flying away due to 
            a huge initial contact force
        """
        raise NotImplementedError
        # return 2
    
    def set_name(self, name):
        """
            Changes the name of this object to something else
        """

class MujocoFileObject(MujocoXMLFile, MujocoObject): 
    """
        MujocoObjects that are loaded from xml files
    """
    def __init__(self, fname):
        """
            The object must follow the following structure:
            <mujoco model="box">
            ...
            <worldbody>
                <body name="main_body">
                    <site name=bottom_site/>
                    <site name=top_site/>
                    <site name=horizontal_radius_site/>
                    <site name="main_site">
                <body/>
            <worldbody/>
        """
        MujocoXMLFile.__init__(self, fname)
        self.build()

    def build(self):
        self.bottom_site = self.worldbody.find("../site[@name='bottom_site']")
        self.bottom_offset = string_to_array(bottom_site.get('pos'))
        self.top_site = self.worldbody.find("../site[@name='top_site']")
        self.top_offset = string_to_array(top_site.get('pos'))
        self.horizontal_radius_site = self.worldbody.find("../site[@name='horizontal_radius_site']")
        self.horizontal_radius = string_to_array(horizontal_radius_site.get('pos'))[0]
        self.main_site = self.worldbody.find("../site[@name='main_site']")
        self.body = self.worldbody.find("./body[@name='main_body']")

        self.set_name(self.name)

    def get_bottom_offset(self):
        return self.bottom_offset

    def get_top_offset(self):
        return self.top_offset

    def get_horizontal_radius(self):
        return self.horizontal_radius

    def set_name(self, name):
        """
            Rename all named elements to follow the new name
        """
        old_name = self.name
        self.name = name
        self.bottom_site.set('name', '{}_{}'.format(self.name, 'bottom_site'))
        self.top_site.set('name', '{}_{}'.format(self.name, 'top_site'))
        self.horizontal_radius_site.set('name', '{}_{}'.format(self.name, 'horizontal_radius_site'))
        self.center_cite.set('name', self.name)
        self.body.set('name', self.name)

class DefaultBoxObject(MujocoFileObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_box.xml'))

class DefaultBallObject(MujocoFileObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_ball.xml'))

class DefaultCylinderObject(MujocoFileObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_cylinder.xml'))

class DefaultCapsuleObject(MujocoFileObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_capsule.xml'))


class MujocoGeneratedObject(MujocoObject):
    """
        Base class for all programmatically generated mujoco object
        i.e., every MujocoObject that does not have an corresponding xml file 

        Every generated object must have a name, which will be used to index 
        the enclosing body, and the cite at the center (pos = "0 0 0") of the body
    """
    def __init__(self, name=None):
        if name is None:
            name = 'mujoco_generated_object_{}'.format(np.random.randint(0,1000))
        super().__init__(name)
        self.body = ET.Element('body', attrib={'name': self.name})
        self.worldbody.append(self.body)
        self.build()
        self.append_site()

    # returns a copy, Returns xml body node
    def build(self):
        """
            Build the object under <worldbody>
                                        <body>
        """
        raise NotImplementedError

    def append_site(self):
        if self.site_appended:
            return
        site_attributes = {
                'pos': '0 0 0',
                'size': '0.002 0.002 0.002',
                'rgba': '1 0 0 1',
                'type': 'sphere',
                }
        site_attributes['name'] = self.name
        self.site = ET.Element('site', attrib=site_attributes)
        self.body.append(self.site)
        self.site_appended = True

    def set_name(self, name):
        self.name = name
        self.body.set('name', name)
        self.site.set('name', name)


class BoxObject(MujocoGeneratedObject):
    """
        An object that is a box
    """
    # TODO: friction, etc
    def __init__(self, name=None, size=None, rgba=None):
        if name is None:
            name = 'box_{}'.format(np.random.randint(0,1000))
        if size is None:
            size = [0.01, 0.01, 0.01]
        if rgba is None:
            rgba = [1, 0, 0, 1]
        assert(len(size) == 3)
        self.size = np.array(size)
        self.rgba = np.array(rgba)
        super().__init__(name)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def build(self):
        template = {}
        template['pos']  = '0 0 0'
        template['type'] = 'box'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        self.body.append(ET.Element('geom', attrib=template))


class CylinderObject(MujocoGeneratedObject):
    """
        An object that is a cylinder
    """
    # TODO: friction, etc
    def __init__(self, name=None, size=None, rgba=None):
        if size is None:
            size = [0.01, 0.01]
        if rgba is None:
            rgba = [1, 0, 0, 1]
        if name is None:
            name = 'cylinder_{}'.format(np.random.randint(0,1000))
        assert(len(size) == 2)
        self.size = np.array(size)
        self.rgba = np.array(rgba)
        super().__init__(name)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    def build(self):
        template = {}
        template['pos']  = '0 0 0'
        template['type'] = 'cylinder'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        self.body.append(ET.Element('geom', attrib=template))


class BallObject(MujocoGeneratedObject):
    """
        An object that is a ball (sphere)
    """
    # TODO: friction, etc
    def __init__(self, name=None, size=None, rgba=None):
        print('---')
        print(size)
        print('---')
        if size is None:
            size = [0.01]
        if rgba is None:
            rgba = [1, 0, 0, 1]
        if name is None:
            name = 'ball_{}'.format(np.random.randint(0,1000))
        assert(len(size) == 1)
        self.size = np.array(size)
        self.rgba = np.array(rgba)
        super().__init__(name)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    def build(self):
        template = {}
        template['pos']  = '0 0 0'
        template['type'] = 'sphere'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        self.body.append(ET.Element('geom', attrib=template))


class CapsuleObject(MujocoGeneratedObject):
    """
        An object that is a capsule 
    """
    # TODO: friction, etc
    def __init__(self, name=None, size=None, rgba=None):
        if size is None:
            size = [0.01, 0.01]
        if rgba is None:
            rgba = [1, 0, 0, 1]
        if name is None:
            name = 'capsule_{}'.format(np.random.randint(0,1000))
        assert(len(size) == 2)
        self.size = np.array(size)
        self.rgba = np.array(rgba)
        super().__init__(name)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    def build(self):
        template = {}
        template['pos']  = '0 0 0'
        template['type'] = 'capsule'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        self.body.append(ET.Element('geom', attrib=template))


class RandomBoxObject(BoxObject):
    """
        A random box
    """
    def __init__(self, name=None, size_max=[0.07, 0.07, 0.07], size_min=[0.03, 0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(3)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        
        # # create a custom name depending on system time
        # t1, t2 = str(time.time()).split('.')
        # name = "random_box_{}_{}".format(t1, t2)
        # print("creating object with name: {}".format(name))
        super().__init__(name, size, rgba)

class RandomCylinderObject(CylinderObject):
    """
        A random cylinder
    """
    def __init__(self, name=None, size_max=[0.07, 0.07], size_min=[0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(2)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])

        # # create a custom name depending on system time
        # t1, t2 = str(time.time()).split('.')
        # name = "random_cylinder_{}_{}".format(t1, t2)
        # print("creating object with name: {}".format(name))
        super().__init__(name, size, rgba)

class RandomBallObject(BallObject):
    """
        A random ball (sphere)
    """
    def __init__(self, name=None, size_max=[0.07], size_min=[0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(1)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        
        # # create a custom name depending on system time
        # t1, t2 = str(time.time()).split('.')
        # name = "random_ball_{}_{}".format(t1, t2)
        # print("creating object with name: {}".format(name))
        super().__init__(size, name, rgba)

class RandomCapsuleObject(CapsuleObject):
    """
        A random ball (sphere)
    """
    def __init__(self, name=None, size_max=[0.07, 0.07], size_min=[0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(2)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        
        # # create a custom name depending on system time
        # t1, t2 = str(time.time()).split('.')
        # name = "random_capsule_{}_{}".format(t1, t2)
        # print("creating object with name: {}".format(name))
        super().__init__(name, size, rgba)

