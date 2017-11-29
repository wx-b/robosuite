# MujocoManipulation

## Installation on mac
* This repo officially supports python 3.5.2 on Mac OS X.
* (Optionally set up virtual env) 
```bash
conda create -n mujocomanip python=3.5.2
source activate mujocomanip
```
* Install Mujoco 1.5.0 following the direction in [the mujoco-py repo](https://github.com/openai/mujoco-py)
* Run in project root directory
```bash
pip install -e .
```
* Run the demo
```bash
python demo.py
```
## Prototyping new environments

- It is often necessary to prototype new environments. To make this simple, we rely on the [VRTeleop](https://github.com/StanfordVL/VRTeleop/tree/master) repo.
- You can control the Sawyer through **Keyboard** or **iPhone** (must install the [SawyerTeleop](https://github.com/StanfordVL/SawyerApp) app).
- Edit [config.py](https://github.com/StanfordVL/VRTeleop/blob/master/config.py) to configure the settings and also choose which environment to debug. 