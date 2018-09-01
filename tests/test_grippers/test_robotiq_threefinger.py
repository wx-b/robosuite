from RoboticsSuite.models.grippers import GripperTester, RobotiqThreeFingerGripper


def test_robotiq_three_finger():
    robotiq_three_finger_tester(False)


def robotiq_three_finger_tester(render):
    gripper = RobotiqThreeFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.02,
        gripper_high_pos=0.1,
        render=render,
    )
    tester.start_simulation()
    tester.loop()


if __name__ == "__main__":
    robotiq_three_finger_tester(True)
