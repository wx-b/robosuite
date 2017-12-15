from setuptools import setup, find_packages
import sys

setup(name='MujocoManip',
      packages=[package for package in find_packages()
                if package.startswith('MujocoManip')],
      install_requires=['glfw',
						'mujoco-py==1.50.1',
						'numpy',
						],
      description="MujocoManipulation: Provides training envs for continuous control on robots",
      author="JG, AJ, JZ, YZ",
      url='https://github.com/StanfordVL/MujocoManipulation',
      author_email="",
      version="0.1.0")
