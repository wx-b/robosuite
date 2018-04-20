# MujocoManipulation

## Installation on mac
* This repo officially supports python 3.5.2 on Mac OS X.
* (Optionally set up virtual env) 
```bash
conda create -n mujocomanip python=3.5.2
source activate mujocomanip
```
* Install Mujoco 1.5.0 following the direction in the [mujoco-py](https://github.com/openai/mujoco-py) repo
* Run in project root directory
```bash
pip install -e .
```
* Run the demo
```bash
python demo.py
```


## Creating a new environment

- Take a look at [sawyer\_lift.py](https://github.com/StanfordVL/MujocoManipulation/blob/master/MujocoManip/environments/sawyer_lift.py) and [sawyer\_stack.py](https://github.com/StanfordVL/MujocoManipulation/blob/master/MujocoManip/environments/sawyer_stack.py) for inspiration.

- When you are done implementing your environment, add the following line to [\_\_init\_\_.py](https://github.com/StanfordVL/MujocoManipulation/blob/master/MujocoManip/__init__.py) 

  ```python
  from MujocoManip.environment.your_env_file import YourEnv
  ```

- Also register the environment in [base.py](https://github.com/StanfordVL/MujocoManipulation/blob/master/MujocoManip/environments/base.py) by importing the environment and adding it to the REGISTERED_ENVS dictionary.

## Prototyping new environments

- It is often necessary to prototype new environments. To make this simple, we rely on the [RobotTeleop](https://github.com/StanfordVL/RobotTeleop) repo.
- You can control the Sawyer through **Keyboard** or **iPhone** (must install the [SawyerApp](https://github.com/StanfordVL/SawyerApp) app).
- Edit [config.py](https://github.com/StanfordVL/RobotTeleop/blob/master/RobotTeleop/config.py) to configure the settings and also choose which environment to debug. 