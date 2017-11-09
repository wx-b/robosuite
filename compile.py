import os
import sys
from mujoco_py import load_model_from_path
from shutil import copyfile



def print_usage():
	print("""python compile.py input_file output_file""")

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print_usage()
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	input_folder = os.path.dirname(input_file)

	# Create a tempfile at the same folder of input file.
	# This avoids mujoco-py from complaining about .urdf extension
	# Also allows assets to be compiled properly
	tempfile = os.path.join(input_folder, '.surreal_temp_model.xml')
	copyfile(input_file, tempfile)

	model = load_model_from_path(tempfile)
	xml_string = model.get_xml()
	with open(output_file, 'w') as f:
		f.write(xml_string)

	os.remove(tempfile)