"""
Some utility functions.

NOTE: convention for quaternions is (x, y, z, w)
"""

import numpy as np
import math
import mujoco_py

pi = np.pi
EPS = np.finfo(float).eps * 4.

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def vec(values):
    """
    Convert value tuple into a vector
    :param values: a tuple of numbers
    :return: vector of given values
    """
    return np.array(values, dtype=np.float32)

def mat4(array):
    """
    Convert an array to 4x4 matrix
    :param array: the array in form of vec, list, or tuple
    :return: 4x4 numpy matrix
    """
    return np.array(array, dtype=np.float32).reshape((4, 4))

def mat2pose(hmat):
    """
    Convert a homogeneous 4x4 matrix into pose
    :param hmat: a 4x4 homogeneous matrix
    :return: (pos, orn) tuple where pos is 
    vec3 float in cartesian, orn is vec4 float quaternion
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn

def mat2quat(rmat, precise=False):
    """
    Convert given rotation matrix to quaternion
    :param rmat: 3x3 rotation matrix
    :param precise: If isprecise is True,
    the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    :return: vec4 float quaternion angles
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if precise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q[[1,2,3,0]]

def mat2euler(rmat, axes='sxyz'):
    """
    Convert given rotation matrix to euler angles in radian.
    :param rmat: 3x3 rotation matrix
    :param axes: One of 24 axis sequences as string or encoded tuple
    :return: converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az)) 

def pose2mat(pose):
    """
    Convert pose to homogeneous matrix
    :param pose: a (pos, orn) tuple where
    pos is vec3 float cartesian, and
    orn is vec4 float quaternion.
    :return:
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat

def quat2mat(quaternion):
    """
    Convert given quaternion  (x, y, z, w) to matrix
    :param quaternion: vec4 float angles
    :return: 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float32, copy=True)[[3,0,1,2]]
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])

def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A 
    to a homogenous matrix corresponding to the same point C in frame B.

    :param pose_A: numpy array of shape (4,4) corresponding to the pose of C in frame A
    :param pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B
    
    :return: numpy array of shape (4,4) corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)

def pose_inv(pose):
    """
    Computes the inverse of a homogenous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    :param pose: numpy array of shape (4,4) for the pose to inverse

    :return: numpy array of shape (4,4) for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense. 
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by 
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by 
    # R-1 to align the axis again. 
    
    pose_inv = np.zeros((4,4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.
    """
    return np.array([0., -pos_A_in_B[2], pos_A_in_B[1],
                     pos_A_in_B[2], 0., -pos_A_in_B[0],
                     -pos_A_in_B[1], pos_A_in_B[0], 0.]).reshape((3, 3))

def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    :param vel_A: 3-dim iterable for linear velocity in A
    :param ang_vel_A: 3-dim iterable for angular velocity in A
    :param pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    :return vel_B, ang_vel_B: two numpy arrays of shape (3,) for the velocities in B

    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B.dot(vel_A) + skew_symm.dot(rot_A_in_B.dot(ang_vel_A))
    ang_vel_B = rot_A_in_B.dot(ang_vel_A)
    return vel_B, ang_vel_B

def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    :param force_A: 3-dim iterable for linear force in A
    :param torque_A: 3-dim iterable for rotational force (moment) in A
    :param pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    :return force_B, torque_B: two numpy arrays of shape (3,) for the forces in B
    """

    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T.dot(force_A)
    torque_B = -rot_A_in_B.T.dot(skew_symm.dot(force_A)) + rot_A_in_B.T.dot(torque_A)
    return force_B, torque_B

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def make_pose(translation, rotation):
    """
    Make a homogenous pose matrix from a translation vector and a rotation matrix.

    :param translation: a 3-dim iterable
    :param rotation: a 3x3 matrix

    :return pose: a 4x4 homogenous matrix
    """
    pose = np.zeros((4,4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def get_orientation_error(target_orn, current_orn):
    """
    Returns the difference between two quaternion orientations as a 3 DOF numpy array.
    For use in an impedance controller / task-space PD controller.

    :param target_orn: 4-dim iterable, desired orientation as a (x, y, z, w) quaternion
    :param current_orn: 4-dim iterable, current orientation as a (x, y, z, w) quaternion

    :return orn_error: 3-dim numpy array for current orientation error, corresponds to (target_orn - current_orn)
    """

    current_orn = np.array([current_orn[3], current_orn[0], current_orn[1], current_orn[2]])
    target_orn = np.array([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])

    pinv = np.zeros((3, 4))
    pinv[0, :] = [-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]]
    pinv[1, :] = [-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]]
    pinv[2, :] = [-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]]
    orn_error = 2.0 * pinv.dot(np.array(target_orn))
    return orn_error

def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    :param target_pose: a 4x4 homogenous matrix for the target pose
    :param current_pose: a 4x4 homogenous matrix for the current pose

    :return: A 6-dim numpy array for the pose error.
    """

    error = np.zeros(6)

    # compute translational error 
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos

    # compute rotational error
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (np.cross(r1, r1d) + np.cross(r2, r2d) + np.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error

### support for mocap ###

# def ctrl_set_action(physics, action):
#     """For torque actuators it copies the action into mujoco ctrl field.
#     For position actuators it sets the target relative to the current qpos.
#     """

#     if physics.model.nmocap > 0:
#         ### note: this grabs gripper ctrl only... ###
#         _, action = np.split(action, (physics.model.nmocap * 7, ))
#     if physics.data.ctrl is not None:
#         for i in range(action.shape[0]):
#             if physics.model.actuator_biastype[i] == 0:
#                 physics.data.ctrl[i] = action[i]
#             else:
#                 idx = physics.model.jnt_qposadr[physics.model.actuator_trnid[i, 0]]
#                 physics.data.ctrl[i] = physics.data.qpos[idx] + action[i]


# def mocap_set_action(physics, action):
#     """The action controls the robot using mocaps. Specifically, bodies
#     on the robot (for example the gripper wrist) is controlled with
#     mocap bodies. In this case the action is the desired difference
#     in position and orientation (quaternion), in world coordinates,
#     of the of the target body. The mocap is positioned relative to
#     the target body according to the delta, and the MuJoCo equality
#     constraint optimizer tries to center the welded body on the mocap.
#     """
#     if physics.model.nmocap > 0:
#         ### note: this grabs pos/orn control only... ###
#         action, _ = np.split(action, (physics.model.nmocap * 7, ))
#         action = action.reshape(physics.model.nmocap, 7)

#         pos_delta = action[:, :3]
#         quat_delta = action[:, 3:]

#         reset_mocap2body_xpos(physics)
#         physics.data.mocap_pos[:] = physics.data.mocap_pos + pos_delta
#         physics.data.mocap_quat[:] = quat_delta #physics.data.mocap_quat + quat_delta


# def reset_mocap_welds(physics):
#     """Resets the mocap welds that we use for actuation.
#     """
#     if physics.model.nmocap > 0 and physics.model.eq_data is not None:
#         for i in range(physics.model.eq_data.shape[0]):
#             if physics.model.eq_type[i] == enums.mjtEq.mjEQ_WELD:
#                 physics.model.eq_data[i, :] = np.array(
#                     [0., 0., 0., 1., 0., 0., 0.])
#     physics.forward()


# def reset_mocap2body_xpos(physics):
#     """Resets the position and orientation of the mocap bodies to the same
#     values as the bodies they're welded to.
#     """

#     if (physics.model.eq_type is None or
#         physics.model.eq_obj1id is None or
#         physics.model.eq_obj2id is None):
#         return

#     for eq_type, obj1_id, obj2_id in zip(physics.model.eq_type,
#                                          physics.model.eq_obj1id,
#                                          physics.model.eq_obj2id):
#         if eq_type != enums.mjtEq.mjEQ_WELD: 
#             continue

#         mocap_id = physics.model.body_mocapid[obj1_id]
#         if mocap_id != -1:
#             # obj1 is the mocap, obj2 is the welded body
#             body_idx = obj2_id
#         else:
#             # obj2 is the mocap, obj1 is the welded body
#             mocap_id = physics.model.body_mocapid[obj2_id]
#             body_idx = obj1_id

#         assert (mocap_id != -1)
#         physics.data.mocap_pos[mocap_id][:] = physics.data.xpos[body_idx]
#         physics.data.mocap_quat[mocap_id][:] = physics.data.xquat[body_idx]

def mjpy_ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        ### note: this grabs gripper ctrl only... ###
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mjpy_mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        mjpy_reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = quat_delta #sim.data.mocap_quat + quat_delta


def mjpy_reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def mjpy_reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

def range_to_reward(x, r1, r2, y):
    """
    A function f(y) such that:
        f(y) = 1 for y in [x - r_1, x + r_1]
        f(y) = 0 for y in [x - r_2, x + r_2]
    And the function decreases linearly between that
    """
    if abs(y - x) <= r1:
        return 1
    elif abs(y - x) >= r2:
        return 0
    else:
        return 1 - (abs(y - x) - r1) / (r2 - r1)


