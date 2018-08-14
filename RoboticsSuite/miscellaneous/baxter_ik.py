import numpy as np
import random
import sys
import pybullet as p
from os.path import join as pjoin

import RoboticsSuite.miscellaneous.utils as U


class BaxterIKController(object):
    """
    This controller is used to control a robot using Inverse Kinematics
    to map end effector motions to joint motions. 
    """

    def __init__(self, bullet_data_path, robot_jpos_getter, rest_poses):

        # path to data folder of bullet repository
        self.bullet_data_path = bullet_data_path

        # returns current robot joint positions
        self.robot_jpos_getter = robot_jpos_getter

        # Do any setup needed for Inverse Kinematics.
        self.setup_inverse_kinematics()

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 1.0

        self.rest_poses = rest_poses

        self.sync_state()

    def get_control(self, dpos, rotation):
        """
        This function is repeatedly called in the control loop. 
        It takes the current user information and uses it to return joint velocities.

        :param user_info: a dictionary containing information on user controls

        :return velocities: The joint velocity commands to apply.
        """

        # Sync joint positions for IK.
        self.sync_ik_robot(self.robot_jpos_getter())

        # ### TODO: scale down dpos appropriately based on current distance to IK target ###
        # cur_pos = robot.eef_position()
        # new_pos = user_info["dpos"] + self.ik_robot_target_pos

        # norm = np.linalg.norm(new_pos - cur_pos)
        # if norm > 0.1:
        #     user_info["dpos"] = 0.1 * (user_info["dpos"] / norm)

        self.commanded_joint_positions = self.joint_positions_for_user_displacement(
            dpos, rotation
        )

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(7)
        deltas = self._get_current_error(
            self.robot_jpos_getter(), self.commanded_joint_positions
        )
        # print("%.3f"%np.linalg.norm(deltas), ', '.join(map(lambda x: "%.3f"%x, deltas)))
        # print(self.robot_jpos_getter())
        # print(deltas, self.commanded_joint_positions, self.robot_jpos_getter())
        # print(self.robot_jpos_getter())
        for i, delta in enumerate(deltas):
            velocities[i] = -2. * delta  # -2. * delta
        velocities = self.clip_joint_velocities(velocities)
        # print(velocities)
        return velocities

    def sync_state(self):
        """
        This function is called when the user is not controlling the robot.
        It takes a TeleopEnv instance (which is a robot to control) and 
        does internal bookkeeping to maintain consistent state. 

        :param robot: An instance of TeleopEnv. 
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot(self.robot_jpos_getter())

        ### TODO: Why is the bullet eef z-position so wrong??? ###

        # make sure target pose is up to date
        self.ik_robot_target_pos, self.ik_robot_target_orn = (
            self.ik_robot_eef_joint_cartesian_pose()
        )

    def setup_inverse_kinematics(self):
        """
        This function is responsible for doing any setup for inverse kinematics.
        Inverse Kinematics maps end effector (EEF) poses to joint angles that
        are necessary to achieve those poses. The user controls the robot in
        EEF space, but our controller must send joint angle commands to the robot.
        """

        # Use PyBullet to handle inverse kinematics.

        # Set up a connection to the PyBullet simulator.
        try:
            p.resetSimulation()
            p.disconnect()
        except:
            pass
        p.connect(p.DIRECT)
        p.resetSimulation()

        # get paths to urdfs
        # self.robot_urdf = pjoin(self.bullet_data_path, "sawyer_description/urdf/sawyer_arm.urdf")
        # self.robot_urdf = pjoin(self.bullet_data_path, "baxter_custom_ikfast/baxter_arm.accurate.left.urdf")
        self.robot_urdf = self.bullet_data_path

        # load the urdfs
        self.ik_robot = p.loadURDF(self.robot_urdf, (0, 0, 0.9), useFixedBase=1)
        print(p.getNumJoints(self.ik_robot))
        print()
        for i in range(26):
            print(i, "joints", p.getJointInfo(self.ik_robot, i))
        # exit()

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1)

    def sync_ik_robot(self, joint_positions, simulate=False, sync_last=True):
        """
        Force the robot model used to compute IK to match the provided joint angles.

        :param joint_positions: A list or numpy array.
        :param simulate: If True, actually use physics simulation, else 
                         write to physics state directly.
        :param sync_last: If False, don't sync the last joint angle (for directly controlling roll)
        """

        num_joints = len(joint_positions)
        if not sync_last:
            num_joints -= 1
        mp = [11, 12, 13, 14, 15, 17, 18]
        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    self.ik_robot,
                    mp[i],
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                p.resetJointState(self.ik_robot, mp[i], joint_positions[i], 0)

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a x-y-z-w quaternion
        """
        eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, 18)[0])
        eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, 18)[1])
        eef_pose_in_world = U.pose2mat((eef_pos_in_world, eef_orn_in_world))

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = U.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = U.pose_inv(base_pose_in_world)

        eef_pose_in_base = U.pose_in_A_to_pose_in_B(
            pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
        )

        return U.mat2pose(eef_pose_in_base)

    def inverse_kinematics(self, target_position, target_orientation, rest_poses=None):
        """
        Helper function to do inverse kinematics for a given target position and 
        orientation in the PyBullet world frame.

        :param target_position: A tuple, list, or numpy array of size 3 for position.
        :param target_orientation: A tuple, list, or numpy array of size 4 for 
                                   a orientation quaternion.
        :param rest_poses: (optional) A list of size @num_joints to favor ik solutions close by.

        :return: A list of size @num_joints corresponding to the joint angle solution.
        """

        ### TODO: experiment with NULL space, removing damping, rest poses, etc... ###
        ### TODO: should rest poses final value be the initial roll or commanded roll? ###

        if rest_poses is None:
            raise SystemExit
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    6,
                    target_position,
                    targetOrientation=target_orientation,
                    restPoses=[0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161],
                    jointDamping=[0.1] * 7,
                )
            )
        else:
            # ik_solution = list(p.calculateInverseKinematics(self.ik_robot, 6,
            #                    target_position, targetOrientation=target_orientation,
            #                    restPoses=rest_poses, jointDamping=[0.1] * 7))

            ### TODO: experiment with joint damping in simulation... ###
            ### TODO: if this is at link 6, how is the code working? target orn is eef frame, not link 6... ###
            ### TODO: test mujoco eef orientation vs. link 6 orientation in bullet vs. link 6 orn in mujoco... ###

            mp = [11, 12, 13, 14, 15, 17, 18]

            lowerLimits = [-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]
            upperLimits = [1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059]

            def actual(a):
                z = np.zeros(26)
                for i in range(7):
                    z[mp[i]] = a[i]
                return z

            # trying nullspace IK here
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    18,
                    target_position,
                    targetOrientation=target_orientation,
                    # solver = 0,
                    # lowerLimits=actual(lowerLimits), upperLimits=actual(upperLimits),
                    restPoses=[actual(self.robot_jpos_getter())],
                    jointDamping=[1] * 26,
                )
            )  # restPoses=rest_poses,
            """eps = 1e-1
            for i in range(7):
                if ik_solution[i] < lowerLimits[i]:
                    ik_solution[i] = lowerLimits[i]
                elif ik_solution[i] > upperLimits[i]:
                    ik_solution[i] = upperLimits[i]
                    #print(ik_solution)
                    #raise ValueError
            """
        # print("ik solution", ik_solution, len(ik_solution))
        self.ik_solution = ik_solution
        return ik_solution

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        :param pose_in_base: a (pos, orn) tuple
        :return pose_in world: a (pos, orn) tuple
        """
        pose_in_base = U.pose2mat(pose_in_base)

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = U.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = U.pose_in_A_to_pose_in_B(
            pose_A=pose_in_base, pose_A_in_B=base_pose_in_world
        )
        return U.mat2pose(pose_in_world)

    def joint_positions_for_user_displacement(self, dpos, rotation):
        """
        This function runs inverse kinematics to back out target joint positions
        from the user displacement information.

        :param user_displacement: a dictionary with keys "dpos" and "rotation"

        :return: A list of size @num_joints corresponding to the target joint angles.
        """

        self.ik_robot_target_pos += dpos * self.user_sensitivity

        # we are rotating the initial simulation configuration for easy stacking
        rotation = rotation.dot(
            U.rotation_matrix(angle=-np.pi / 2, direction=[0., 0., 1.], point=None)[
                :3, :3
            ]
        )

        # print(rotation,dpos,rotation,self.ik_robot_target_pos)
        # print("tgt", self.ik_robot_target_orn)
        self.ik_robot_target_orn = U.mat2quat(rotation)
        # self.ik_robot_target_orn = np.array([-0.15246771, -0.01503118, 0.13649071, 0.97872261])

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos, self.ik_robot_target_orn)
        )

        rest_poses = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]
        rest_poses = (
            self.rest_poses
        )  # [3.10821564e-09, -5.50000000e-01, 1.38161579e-09, 1.28400000e+00, 4.89875129e-11, 2.61600000e-01, -2.71076012e-11]

        for bullet_i in range(100):

            arm_joint_pos = self.inverse_kinematics(
                world_targets[0], world_targets[1], rest_poses=rest_poses
            )

            # arm_joint_pos = self.inverse_kinematics(world_targets[0], world_targets[1])

            # Update pybullet joint position for its iterative IK solver
            # jpos = self.robot.joint_positions()
            # self.sync_ik_robot(jpos, sync_last=False)

            self.sync_ik_robot(arm_joint_pos, sync_last=True)

        # print("target pos: {}".format(self.ik_robot_target_pos))
        # print("target orn: {}".format(self.ik_robot_target_orn))
        # print("commanded jpos: {}".format(arm_joint_pos))
        # from IPython import embed
        # embed()

        # print("Bullet IK took {} seconds".format(time.time() - t1))
        return arm_joint_pos

    def _get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current joint positions.
        Useful for PID control.

        :param current: the current joint positions
        :param set_point: the joint positions that are desired as a numpy array
        :return: the current error in the joint positions
        """
        error = current - set_point
        return error

    def clip_joint_velocities(self, velocities):
        """
        Clips joint velocities into a valid range.
        """
        for i in range(len(velocities)):
            if velocities[i] >= 1.0:
                # print("CLIPPED joint {} at {}".format(i, velocities[i]))
                velocities[i] = 1.0
            elif velocities[i] <= -1.0:
                # print("CLIPPED joint {} at {}".format(i, velocities[i]))
                velocities[i] = -1.0
        return velocities
