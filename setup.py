from setuptools import setup, find_packages
import sys

setup(
    name="MujocoManip",
    packages=[
        package for package in find_packages() if package.startswith("MujocoManip")
    ],
    install_requires=[
        "glfw",
        # 'mujoco-py==1.50.1',
        "numpy",
    ],
    description="Stanford Robotics Suite: Extensive Simulated Benchmarks for Robot Manipulation",
    author="Yuke Zhu, Jiren Zhu, Ajay Mandlekar, Joan Creus-Costa, Anchit Gupta",
    url="https://github.com/StanfordVL/MujocoManipulation",
    author_email="yukez@cs.stanford.edu",
    version="0.1.0",
)
