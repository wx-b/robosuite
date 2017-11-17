from setuptools import setup, find_packages
import sys

setup(name='MujocoManip',
      packages=[package for package in find_packages()
                if package.startswith('MujocoManip')],
      install_requires=[],
      description="OpenAI baselines: high quality implementations of reinforcement learning algorithms",
      author="AJ",
      url='https://github.com/openai/baselines',
      author_email="gym@openai.com",
      version="0.1.4")
