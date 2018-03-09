import copy
import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import *


class MujocoObject():
    """
        Base class for all objects
        We use Mujoco Objects to implement all objects that 
        1) may appear for multiple times in a task
        2) can be swapped between different tasks
        Typical methods return copy so the caller can all joints/attributes as wanted
    """
    def __init__(self):
        self.asset = ET.Element('asset')

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

    def get_collision(self):
        """
            Returns a ET.Element
            It is a <body/> subtree that defines all collision related stuff of this object
            Return is a copy
        """
        raise NotImplementedError
        
    def get_visual(self):
        """
            Returns a ET.Element
            It is a <body/> subtree that defines all visual related stuff of this object
            Return is a copy
        """
        raise NotImplementedError

    def get_full(self):
        """
            Returns a ET.Element
            It is a <body/> subtree that defines all collision and visual related stuff of this object
            Return is a copy
        """
        collision = self.get_collision()
        visual = self.get_visual()
        collision.append(visual)
        return collision

class MujocoXMLObject(MujocoXML, MujocoObject):
    """
        MujocoObjects that are loaded from xml files
    """
    def __init__(self, fname):
        MujocoXML.__init__(self, fname)

    def get_bottom_offset(self):
        bottom_site = self.worldbody.find("./site[@name='bottom_site']")
        return string_to_array(bottom_site.get('pos'))

    def get_top_offset(self):
        top_site = self.worldbody.find("./site[@name='top_site']")
        return string_to_array(top_site.get('pos'))

    def get_horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find("./site[@name='horizontal_radius_site']")
        return string_to_array(horizontal_radius_site.get('pos'))[0]

    def get_collision(self):
        collision = copy.deepcopy(self.worldbody.find("./body[@name='collision']"))
        collision.attrib.pop('name')
        return collision

    def get_visual(self):
        visual = copy.deepcopy(self.worldbody.find("./body[@name='visual']"))
        visual.attrib.pop('name')
        return visual

class DefaultBoxObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_box.xml'))

class DefaultBallObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_ball.xml'))

class DefaultCylinderObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_cylinder.xml'))

class DefaultCapsuleObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion('object/object_capsule.xml'))

class MujocoGeneratedObject(MujocoObject):
    """
        Base class for all programmatically generated mujoco object
        i.e., every MujocoObject that does not have an corresponding xml file 
    """
    def get_collision_attrib_template(self):
        return {'pos': '0 0 0'}

    def get_visual_attrib_template(self):
        return {'conaffinity': "0", 'contype': "0"}

class BoxObject(MujocoGeneratedObject):
    """
        An object that is a box
    """
    # TODO: friction, etc
    def __init__(self, size, rgba):
        super().__init__()
        assert(len(size) == 3)
        self.size = np.array(size)
        self.rgba = np.array(rgba)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def get_collision(self):
        body = ET.Element('body')
        template = self.get_collision_attrib_template()
        template['type'] = 'box'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        body.append(ET.Element('geom', attrib=template))
        return body
    
    # returns a copy, Returns xml body node
    def get_visual(self):
        body = ET.Element('body')
        template = self.get_visual_attrib_template()
        template['type'] = 'box'
        template['rgba'] = array_to_string(self.rgba)
        # shrink so that we don't see flickering when showing both visual and collision
        template['size'] = array_to_string(self.size * visual_size_shrink_ratio) 
        template['group'] = '0'
        body.append(ET.Element('geom', attrib=template))
        template_gp1 = copy.deepcopy(template)
        template_gp1['group'] = '1'
        body.append(ET.Element('geom', attrib=template_gp1))
        return body

class CylinderObject(MujocoGeneratedObject):
    """
        An object that is a cylinder
    """
    # TODO: friction, etc
    def __init__(self, size, rgba):
        super().__init__()
        assert(len(size) == 2)
        self.size = np.array(size)
        self.rgba = np.array(rgba)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self):
        body = ET.Element('body')
        template = self.get_collision_attrib_template()
        template['type'] = 'cylinder'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        body.append(ET.Element('geom', attrib=template))
        return body
    
    # returns a copy, Returns xml body node
    def get_visual(self):
        body = ET.Element('body')
        template = self.get_visual_attrib_template()
        template['type'] = 'cylinder'
        template['rgba'] = array_to_string(self.rgba)
        # shrink so that we don't see flickering when showing both visual and collision
        template['size'] = array_to_string(self.size * visual_size_shrink_ratio) 
        template['group'] = '0'
        body.append(ET.Element('geom', attrib=template))
        template_gp1 = copy.deepcopy(template)
        template_gp1['group'] = '1'
        body.append(ET.Element('geom', attrib=template_gp1))
        return body

class BallObject(MujocoGeneratedObject):
    """
        An object that is a ball (sphere)
    """
    # TODO: friction, etc
    def __init__(self, size, rgba):
        super().__init__()
        assert(len(size) == 1)
        self.size = np.array(size)
        self.rgba = np.array(rgba)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self):
        body = ET.Element('body')
        template = self.get_collision_attrib_template()
        template['type'] = 'sphere'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        body.append(ET.Element('geom', attrib=template))
        return body
    
    # returns a copy, Returns xml body node
    def get_visual(self):
        body = ET.Element('body')
        template = self.get_visual_attrib_template()
        template['type'] = 'sphere'
        template['rgba'] = array_to_string(self.rgba)
        # shrink so that we don't see flickering when showing both visual and collision
        template['size'] = array_to_string(self.size * visual_size_shrink_ratio) 
        template['group'] = '0'
        body.append(ET.Element('geom', attrib=template))
        template_gp1 = copy.deepcopy(template)
        template_gp1['group'] = '1'
        body.append(ET.Element('geom', attrib=template_gp1))
        return body

class CapsuleObject(MujocoGeneratedObject):
    """
        An object that is a capsule 
    """
    # TODO: friction, etc
    def __init__(self, size, rgba):
        super().__init__()
        assert(len(size) == 2)
        self.size = np.array(size)
        self.rgba = np.array(rgba)

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self):
        body = ET.Element('body')
        template = self.get_collision_attrib_template()
        template['type'] = 'capsule'
        template['rgba'] = array_to_string(self.rgba)
        template['size'] = array_to_string(self.size)
        body.append(ET.Element('geom', attrib=template))
        return body
    
    # returns a copy, Returns xml body node
    def get_visual(self):
        body = ET.Element('body')
        template = self.get_visual_attrib_template()
        template['type'] = 'capsule'
        template['rgba'] = array_to_string(self.rgba)
        # shrink so that we don't see flickering when showing both visual and collision
        template['size'] = array_to_string(self.size * visual_size_shrink_ratio) 
        template['group'] = '0'
        body.append(ET.Element('geom', attrib=template))
        template_gp1 = copy.deepcopy(template)
        template_gp1['group'] = '1'
        body.append(ET.Element('geom', attrib=template_gp1))
        return body


class RandomBoxObject(BoxObject):
    """
        A random box
    """
    def __init__(self, size_max=[0.07, 0.07, 0.07], size_min=[0.03, 0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(3)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        super().__init__(size, rgba)

class RandomCylinderObject(CylinderObject):
    """
        A random cylinder
    """
    def __init__(self, size_max=[0.07, 0.07], size_min=[0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(2)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        super().__init__(size, rgba)

class RandomBallObject(BallObject):
    """
        A random ball (sphere)
    """
    def __init__(self, size_max=[0.07], size_min=[0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(1)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        super().__init__(size, rgba)

class RandomCapsuleObject(CapsuleObject):
    """
        A random ball (sphere)
    """
    def __init__(self, size_max=[0.07, 0.07], size_min=[0.03, 0.03], seed=None):
        if seed is not None:
            np.random.seed(seed)
        size = np.array([np.random.uniform(size_min[i], size_max[i]) for i in range(2)])
        rgba = np.array([np.random.uniform(0, 1) for i in range(3)] + [1])
        super().__init__(size, rgba)


