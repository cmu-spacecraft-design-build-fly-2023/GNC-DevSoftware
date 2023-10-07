import numpy as np
from utils import *
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
print("true body vector: \n", body_measure_gt)
print("noisy sensor measurement : \n", noisy_body_vector)
# print("inertial frame sensor measurement: \n", body_vector_gt)
# print("body frame sensor measurement: \n", body_vector)

weight = 1
B = weight * (np.matmul(noisy_body_vector, inertial_measure_gt.T))
# B = weight * (np.matmul(inertial_vector, body_vector.T))

# print(B)
sigma = np.trace(B)
S_matrix = B + B.T
z = np.array([[B[1][2] - B[2][1]], [B[2][0] - B[0][2]], [B[0][1] - B[1][0]]])
# print(z)
S_sigma = S_matrix - sigma * np.eye(3)
K = np.block([[sigma, z.T], [z, S_matrix]])
# K = np.block([[S_matrix, z], [z.T, sigma]])
eigenvalues, eigenvectors = np.linalg.eig(K)
max_eigenvalue_index = np.argmax(eigenvalues)

# Get the eigenvector corresponding to the maximum eigenvalue
max_eigenvalue_vector = eigenvectors[:, max_eigenvalue_index]
est_quat = max_eigenvalue_vector
print("quaternion estimate : ", est_quat)
H = np.block([[0, 0, 0], [np.eye(3)]])
L = get_left_matrix(est_quat)
R = get_right_matrix(est_quat)
# rot_matrix_Q = H.T @ R.T @ L @ H
rot_matrix_Q = H.T @ L @ R.T @ H

print("estimated rotation matrix : \n")
print(rot_matrix_Q)
# print("inertial vector estimated:  ", rot_matrix_Q @ body_vector)
# print("Q * Q.T : ", rot_matrix_Q @ rotation_matrix_ground_truth.T)
E = ground_truth_rot.T @ rot_matrix_Q
e = logm(E)
theta_error = np.linalg.norm(e) * 180 / np.pi
print(theta_error)
