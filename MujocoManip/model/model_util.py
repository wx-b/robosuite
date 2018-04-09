import xml.etree.ElementTree as ET
import os
import numpy as np
import MujocoManip.model

visual_size_shrink_ratio = 0.99

def xml_path_completion(xml_path):
    """
        Takes in a local xml path and returns a full path. 
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(MujocoManip.model.assets_root, xml_path)
    return full_path

def array_to_string(array):
    """
        Converts a numeric array into the string format in mujoco
        [0, 1, 2] => "0 1 2"
    """
    return ' '.join(['{}'.format(x) for x in array])

def string_to_array(string):
    """
        Converts a array string in mujoco xml to np.array
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(' ')])

def set_alpha(node, alpha = 0.1):
    """
        Sets all a(lpha) field of the rgba attribute to be @alpha
        for @node and all subnodes
        used for managing display
    """
    for child_node in node.findall('.//*[@rgba]'):
        rgba_orig = string_to_array(child_node.get('rgba'))
        child_node.set('rgba', array_to_string(list(rgba_orig[0:3]) + [alpha]))

def joint(**kwargs):
    """
        Create a joint tag with attributes specified by @**kwargs
    """
    element = ET.Element('joint', attrib=kwargs)
    return element

def actuator(joint, act_type, **kwargs):
    """
        Create a joint tag with attributes specified by @**kwargs
    """
    element = ET.Element(act_type, attrib=kwargs)
    element.set('joint', joint)
    return element
    