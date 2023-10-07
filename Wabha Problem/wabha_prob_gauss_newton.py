import numpy as np
from utils import *
import sympy as sp
from scipy.spatial.transform import Rotation
from scipy.linalg import logm


def get_rot_from_quat(q):
    L = get_left_matrix(q)
    R = get_right_matrix(q)
    H = np.block([[0, 0, 0], [np.eye(3)]])
    return H.T @ L @ R.T @ H


def residual_function(inertial_vector, body_vector, q):
    A = get_rot_from_quat(q)
    r = inertial_vector - A @ body_vector
    return r


init_quat = np.array(
    [-3.05550767e-17, -9.18178275e-01, 3.47828924e-01, -1.89640966e-01]
)
# init_quat = np.array([0.5, 0.5, 0.5, 1])
# init_quat = init_quat / np.linalg.norm(init_quat)
# init_quat = np.array([0, 0, 0, 1])
ground_truth_rot = get_rot_from_quat(init_quat)
print("ground truth rotation matrix: \n", ground_truth_rot)

r_dash = Rotation.from_matrix(ground_truth_rot)
rot = r_dash.as_rotvec()
axis = rot[:3]  # Extract the axis
angle = np.linalg.norm(rot)

noisy_axis = axis + np.random.normal(0, 0.1, 3)
noisy_angle = angle + np.random.normal(0, np.radians(10))
noisy_axis /= np.linalg.norm(noisy_axis)
noisy_rotation = Rotation.from_rotvec(noisy_axis * noisy_angle)
noisy_rotation_matrix = noisy_rotation.as_matrix()


inertial_measure_gt = np.array([[-1.0], [0.0], [0.0]])
body_measure_gt = ground_truth_rot.T @ inertial_measure_gt
# noisy_body_vector = np.random.normal(body_measure_gt, 0.25, size=(3, 1))
noisy_body_vector = noisy_rotation_matrix.T @ inertial_measure_gt
# print("true body vector: \n", body_measure_gt)
# print("noisy sensor measurement : \n", noisy_body_vector)

q0, q1, q2, q3 = sp.symbols("q0 q1 q2 q3")
q = sp.Matrix([q0, q1, q2, q3])
r = residual_function(inertial_measure_gt, noisy_body_vector, q)


R = sp.Matrix([r[0], r[1], r[2]])
dR_dq0 = R.jacobian([q0, q1, q2, q3])

f = sp.lambdify((q0, q1, q2, q3), dR_dq0, "numpy")
H = np.block([[0, 0, 0], [np.eye(3)]])


k = 0
norm_phi = 1000
q_hat = np.zeros([4, 100])
q_hat[:, 0] = init_quat

# q_hat[:, 0] = np.array([0, -1, 0.3, 0.1])

while norm_phi > 0.00000001:
    qk = q_hat[:, k]
    jacobian_r = f(*qk) @ (get_left_matrix(qk) @ H)
    phi = (
        -np.linalg.pinv(jacobian_r.T @ jacobian_r)
        @ jacobian_r.T
        @ residual_function(inertial_measure_gt, noisy_body_vector, qk)
    )
    phi_quat = rp_quaternion(phi)
    phi_quat = phi_quat.reshape((4, 1))
    q_hat[:, k + 1] = (get_left_matrix(qk) @ phi_quat).flatten()
    norm_phi = np.linalg.norm(phi)
    # print(norm_phi)
    k = k + 1

quat_est = q_hat[:, k]
print("iteration count : ", k)
print(np.linalg.norm(quat_est))
rot_matrix_est = get_rot_from_quat(quat_est)
print("estimated matrix : \n", rot_matrix_est)
E = ground_truth_rot.T @ rot_matrix_est
e = logm(E)
print(e)
theta_error = np.linalg.norm(e) * 180 / np.pi
print(theta_error)
