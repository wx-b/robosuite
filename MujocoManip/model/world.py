import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXML
from MujocoManip.miscellaneous import XMLError
from MujocoManip.model.model_util import xml_path_completion

### Base class to inherit all mujoco worlds from
class MujocoWorldBase(MujocoXML):
    def __init__(self):
        super().__init__(xml_path_completion('base.xml'))

    def merge_asset(self, other):
        """Useful for merging other files in a custom logic"""
        for asset in other.asset:
            asset_name = asset.get('name')
            asset_type = asset.tag
            # Avoids duplication
            if self.asset.find('./{}[@name={}]'.format(asset_type, asset_name)) is None:
                self.asset.append(asset)
