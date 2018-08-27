from RoboticsSuite.models import GripperTester, PushingGripper


def test_pushing():
    pushing_tester(False)


def pushing_tester(render):
    gripper = PushingGripper()
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
    pushing_tester(True)
