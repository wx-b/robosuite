import MujocoManip as MM
import MujocoManip.miscellaneous.utils as U
import numpy as np

def rotate_and_return_quat(rot, rot_angle, direction):
    """
    Takes a rotation matrix, an angle in degrees and a direction, and 
    returns a quaternion after applying a post-rotation.
    """
    rad = np.pi * rot_angle / 180.0
    R = U.rotation_matrix(rad, direction, point=None)
    mat = rot.dot(R[:3, :3])
    return U.mat2quat(mat)

if __name__ == "__main__":

    base_quat = np.array([0.653, 0.271, 0.271, 0.653])
    base_rot = U.quat2mat(base_quat)
    # base_rot = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])

    quat = rotate_and_return_quat(base_rot, 15., [0., 1., 0.])
    print("\"{} {} {} {}\"".format(*quat))