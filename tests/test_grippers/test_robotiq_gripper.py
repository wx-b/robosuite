from RoboticsSuite.models.grippers import GripperTester, RobotiqGripper


def test_robotiq():
    robotiq_tester(False)


def robotiq_tester(render):
    gripper = RobotiqGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.07,
        gripper_high_pos=0.02,
        render=render,
    )
    tester.start_simulation()
    tester.loop()


if __name__ == "__main__":
    robotiq_tester(True)
