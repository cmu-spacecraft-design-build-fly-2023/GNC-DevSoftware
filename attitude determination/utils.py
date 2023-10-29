import numpy as np


def left_quaternion_multiply(q1, q2):
    """
    left Quaternion multiplication
    q1 * q2 = [(s1s2 - v1^Tv2);
              (s1v2 + s2v1 + v1*v2)]
    """
    s1, x1, y1, z1 = q1
    s2, x2, y2, z2 = q2

    s = s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2
    x = s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2
    y = s1 * y2 - x1 * z2 + y1 * s2 + z1 * x2
    z = s1 * z2 + x1 * y2 - y1 * x2 + z1 * s2

    return np.array([s, x, y, z])


def right_quaternion_multiply(q1, q2):
    """
    Right multiplication is left multiplication of q2,q1
    """
    return left_quaternion_multiply(q2, q1)


def quaternion_conjugate(q):
    """
    returns conjuagte of quaternion [s ; -v]
    """
    s, x, y, z = q
    return np.array([s, -x, -y, -z])


def quaternion_norm(q):
    """
    return norm of a quaternion
    """
    x, y, z = q
    quat_norm = np.sqrt(x * x + y * y + z * z)
    return quat_norm


def quaternion_kinematic(q, w):
    """
    returns qdot = 0.5*q*W
    """
    w1, w2, w3 = w
    q_w = np.array([0, w1, w2, w3])
    qdot = 0.5 * left_quaternion_multiply(q, q_w)
    return qdot


def quaternion_to_rp(q):
    s, x, y, z = q
    return np.array([x / s, y / s, z / s])


def rp_quaternion(r):
    r1, r2, r3 = r
    norm = quaternion_norm(np.array([r1, r2, r3]))
    norm = np.sqrt(1 + norm**2)
    q = np.array([1 / norm, r1 / norm, r2 / norm, r3 / norm])
    return q


def get_left_matrix(q):
    s, x, y, z = q
    v = np.array([[x], [y], [z]])
    v_skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    L = np.block([[s, -v.T], [v, s * np.eye(3) + v_skew]])
    return L


def get_right_matrix(q):
    s, x, y, z = q
    v = np.array([[x], [y], [z]])
    v_skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.block([[s, -v.T], [v, s * np.eye(3) - v_skew]])
    return R


# Perform operations
q1 = np.array([1.95399968e-17, 9.52312365e-01, -2.85535588e-01, 1.07566664e-01])
q2 = np.array([0.707, 0, 0.707, 0])
# print(left_quaternion_multiply(q1, q2))
# print(get_right_matrix(q2) @ q1)
