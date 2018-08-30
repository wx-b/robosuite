# Stanford Robotics Suite

![gallery of_environments](resources/gallery.png)

Stanford Robotics Suite is a tookit and simulation benchmark powered by the [MuJoCo physics engine](http://mujoco.org/) for reproducible robotics research. The current release concentrates on reinforcement learning for robot manipulation.

Reinforcement learning has been a powerful and generic tool in robotics. Reinforcement learning combined with deep neural networks, i.e. *deep reinforcement learning* (DRL), has achieved some exciting successes in a variety of robot control problems. However, the challenges of reproducibility and replicability in DRL and robotics have impaired research progress. Our goal is to provide an accessible set of benchmarking tasks that facilitates a fair and rigorus evaluation and improves our understanding of new methods.

This framework was originally developed since late 2017 by researchers in [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL) as an internal tool for robot learning research. Today it is actively maintained and used for robotics research projects in SVL.

This release of Stanford Robotics Suite contains a set of benchmarking manipulation tasks and a modularized design of APIs for building new environments. We highlight these primary features below:

* [**standardized tasks**](RoboticsSuite/environments): a set of single-arm and bimanual manipulation tasks of large diversity and varying complexity.
* [**procedural generation**](RoboticsSuite/models): modularized APIs for programmatically creating new scenes and new tasks as a combinations of robot models, arenas, and parameterized 3D objects;
* [**controller modes**](RoboticsSuite/controllers): a selection of controller types to command the robots, such as joint velocity control, inverse kinematics control, and 3D motion devices for teleoperation;
* **multi-modal sensors**: heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
* **human demonstrations**: utilities for collecting human demonstrations, replaying demonstration datasets, and leveraging demonstration data for learning.

## Installation
TODO(Joan): Talk about system requirement, library dependencies, installation commands on Linux and Mac OS X.

## Quick Start
TODO(Anchit): A demo of how to import the framework and run the environment.

## Building Your Own Environments
TODO(Jiren): A short example of how to create a new environment.

## Human Demonstrations
TODO(Ajay): Talk about how to collect, download, and replay human demonstrations.

## Citations
TODO(Yuke): Add bibtex for citing RoboticsSuite.
