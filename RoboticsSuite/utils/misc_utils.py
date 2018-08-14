import os
import xml.etree.ElementTree as ET


def postprocess_model_xml(xml_str):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.
    """

    import RoboticsSuite
    path = os.path.split(RoboticsSuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    e = root.find("asset")
    meshes = e.findall("mesh")
    textures = e.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        ind = old_path_split.index("RoboticsSuite")
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def range_to_reward(x, r1, r2, y):
    """
    A function f(y) such that:
        f(y) = 1 for y in [x - r_1, x + r_1]
        f(y) = 0 for y in [x - r_2, x + r_2]
    And the function decreases linearly between that
    """
    if abs(y - x) <= r1:
        return 1
    elif abs(y - x) >= r2:
        return 0
    else:
        return 1 - (abs(y - x) - r1) / (r2 - r1)
