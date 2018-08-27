from RoboticsSuite.models import GripperTester, PR2Gripper


def test_pr2():
    pr2_tester(False)


def pr2_tester(render):
    gripper = PR2Gripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.02,
        gripper_high_pos=0.05,
        render=render,
    )
    tester.start_simulation()
    tester.loop()


if __name__ == "__main__":
    pr2_tester(True)
