# Controllers

Controllers are used to perform inverse kinematics computations on different robots. Inverse kinematics is a procedure that takes a desired end effector pose and backs out a set of joint positions that can achieve that end effector pose. It should be noted that there are often (infinitely) many joint positions that can achieve an end effector pose, and additional constraints are often used to favor certain solutions over others. 



We currently support an inverse kinematics controller for the Sawyer robot ([SawyerIKController](sawyer_ik_controller.py)) and one for the Baxter robot ([BaxterIKController](baxter_ik_controller.py)). These controllers allow us to support an end effector action space instead of a joint velocity action space via the [IKWrapper](../wrappers/ik_wrapper.py) environment wrapper. 



In order to use these controllers, you must run the following command to install PyBullet.

```bash
pip install pybullet==1.9.5
```

