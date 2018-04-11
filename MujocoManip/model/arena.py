import numpy as np
import xml.etree.ElementTree as ET
from MujocoManip.model.base import MujocoXMLFile
from MujocoManip.model.model_util import *
from MujocoManip.miscellaneous import XMLError

class Arena(MujocoXMLFile):
	def set_origin(self, pos):
		pos = np.array(pos)
		for node in self.worldbody.findall('./*[@pos]'):
			cur_pos = string_to_array(node.get('pos'))
			new_pos = cur_pos + pos
			node.set('pos', array_to_string(new_pos))

class TableArena(Arena):
	def __init__(self, full_size=(0.8,0.8,0.8), friction=(1, 0.005, 0.0001)):
		self.full_size = np.array(full_size)
		self.half_size = self.full_size / 2
		if friction is None:
			friction = np.array([1, 0.005, 0.0001])
		self.friction = friction

		super().__init__(xml_path_completion('arena/table_arena.xml'))
		self.floor = self.worldbody.find("./geom[@name='floor']")
		self.table_body = self.worldbody.find("./body[@name='table']")
		self.table_collision = self.table_body.find("./geom[@name='table_collision']")
		self.table_visual = self.table_body.find("./geom[@name='table_visual']")
		self.table_top = self.table_body.find("./site[@name='table_top']")

		self.configure_location()

	def configure_location(self):
		self.bottom_pos = np.array([0,0,0])
		self.floor.set('pos', array_to_string(self.bottom_pos))
		
		self.center_pos = self.bottom_pos + np.array([0,0,self.half_size[2]])
		self.table_body.set('pos', array_to_string(self.center_pos))
		self.table_collision.set('size', array_to_string(self.half_size))
		self.table_collision.set('friction', array_to_string(self.friction))
		self.table_visual.set('size', array_to_string(self.half_size * visual_size_shrink_ratio))

		self.table_top.set('pos', array_to_string(np.array([0,0,self.half_size[2]])))
	
	@property
	def table_top_abs(self):
		"""Returns the absolute position of table top"""
		return string_to_array(self.floor.get('pos')) + np.array([0,0,self.full_size[2]])
