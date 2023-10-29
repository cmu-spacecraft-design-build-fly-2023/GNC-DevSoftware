import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


class attitude_estimation:
    def __init__(self):
        # initialise
        tperiod = 60 * 60 * 1  # 10 hour time period in seconds
        self.timestep = 1
        self.tout = np.arange(0, tperiod, self.timestep, dtype=int)
        self.omega_init = np.array([0.0174532925, 0, 0]).T
        self.omega_ground_truth = np.zeros([3, len(self.tout)])
        self.omega_with_bias = np.zeros([3, len(self.tout)])
        self.gt_init_states = np.array(
            [-3.05550767e-17, -9.18178275e-01, 3.47828924e-01, -1.89640966e-01]
        )
        self.gt_attitude_states = np.zeros([4, len(self.tout)])
        self.gt_attitude_states[:, 0] = self.gt_init_states
        self.omega_ground_truth[:, 0] = self.omega_init
        wdot = np.array([0, 0, 0]).T
        w_bias = np.array([0.001, 0.001, 0.001]).T
        self.omega_with_bias[:, 0] = self.omega_init + w_bias
        self.sensor_world_vector = np.array([1, 1, 2]).T
        self.sensor_body_vector = np.zeros([3, len(self.tout)])

        for i in range(len(self.tout) - 1):
            # get ground truth omega and omega with bias
            self.omega_ground_truth[:, i + 1] = (
                self.omega_ground_truth[:, i] + wdot * self.timestep
            )
            self.omega_with_bias[:, i + 1] = self.omega_ground_truth[:, i + 1] + w_bias
            # get ground truth quaternions
            self.gt_attitude_states[:, i + 1], _ = self.state_propogation(
                self.gt_attitude_states[:, i], self.omega_ground_truth[:, i]
            )
        for i in range(len(self.tout)):
            # convert quat to rotation matrix and add noise
            rot_matrix = self.get_rot_from_quat(self.gt_attitude_states[:, i])
            r_dash = Rotation.from_matrix(rot_matrix)
            rot = r_dash.as_rotvec()
            axis = rot[:3]  # Extract the axis
            angle = np.linalg.norm(rot)
            noisy_axis = axis + np.random.normal(0, 0.01, 3)
            noisy_angle = angle + np.random.normal(0, np.radians(1))
            noisy_axis /= np.linalg.norm(noisy_axis)
            noisy_rotation = Rotation.from_rotvec(noisy_axis * noisy_angle)
            noisy_rotation_matrix = noisy_rotation.as_matrix()

            # get body vector measurments
            self.sensor_body_vector[:, i] = (
                noisy_rotation_matrix.T @ self.sensor_world_vector
            )

    def get_rot_from_quat(self, q):
        L = get_left_matrix(q)
        R = get_right_matrix(q)
        H = np.block([[0, 0, 0], [np.eye(3)]])
        return H.T @ L @ R.T @ H

    def state_propogation(self, q, w):
        dt = 1
        theta = np.linalg.norm(w) * dt
        r = w / np.linalg.norm(w)
        dq = np.array(
            [
                np.cos(theta / 2),
                r[0] * np.sin(theta / 2),
                r[1] * np.sin(theta / 2),
                r[2] * np.sin(theta / 2),
            ]
        )
        next_state = get_left_matrix(q) @ dq
        return next_state, dq

    def jacobian(self, curr_state, next_state, dq, dt):
        """
        input : current state
        output : discretised A matrix
        """

        A = np.zeros((6, 6))
        H = np.block([[0, 0, 0], [np.eye(3)]])
        del_phi = (
            H.T
            @ get_left_matrix(next_state).T
            @ get_left_matrix(curr_state)
            @ get_right_matrix(dq)
            @ H
        )
        del_phi_b = -np.eye(3) * dt
        A[:3, :3] = del_phi
        A[:3, 3:] = del_phi_b
        A[3:, 3:] = np.eye(3)

        return A

    # EKF
    def EKF_estimation(self):
        self.filter_states = np.zeros([7, len(self.tout)])  # quat and bias
        self.filter_error_states = np.zeros([6, len(self.tout)])  # rp vector and bias
        self.filter_states[:4, 0] = self.gt_attitude_states[:, 0]
        self.est_attitude_states = np.zeros([7, len(self.tout) + 1])
        x = self.filter_states[:, 0]  # with init quat and 0 bias
        self.est_attitude_states[:, 0] = x
        P = np.eye(6) * 1e-2
        W = np.eye(6) * 1e-3  # process noise
        V = np.eye(3) * 1000  # measurment noise
        delx = self.filter_error_states[:, 0]
        delx[:3] = quaternion_to_rp(x[:4])  # rp for quat and 0 del bias
        steps = 0
        while steps < len(self.tout):
            # state prediction
            x_pred, dq = self.state_propogation(
                x[0:4], self.omega_with_bias[:, steps] + delx[3:]
            )  # current quat with omega bias and delB
            A = self.jacobian(x[:4], x_pred[:4], dq, self.timestep)
            P_pred = A @ P @ A.T + W
            # innovation
            Q_pred = self.get_rot_from_quat(x_pred[:4])  # Q at K+1|K
            z = self.sensor_body_vector[:, steps] - Q_pred.T @ self.sensor_world_vector
            dely_delphi = np.diag(self.sensor_body_vector[:, steps])
            C = np.block([dely_delphi, np.zeros([3, 3])])
            S = C @ P_pred @ C.T + V
            # gain
            K = P_pred @ C.T @ np.linalg.inv(S)
            # update
            delx = K @ z
            theta = np.linalg.norm(delx[:3])
            r = delx[:3] / theta
            delq = np.array(
                [
                    np.cos(theta / 2),
                    r[0] * np.sin(theta / 2),
                    r[1] * np.sin(theta / 2),
                    r[2] * np.sin(theta / 2),
                ]
            )
            x[:4] = get_left_matrix(x_pred[:4]) @ delq.T
            x[4:] = x[4:] + delx[3:]
            P = (np.eye(6) - K @ C) @ P_pred @ (np.eye(6) - K @ C).T + K @ V @ K.T

            self.est_attitude_states[:, steps + 1] = x
            steps = steps + 1

        # RMSE_position = np.sqrt(
        #     (np.linalg.norm(self.orbit_states[0:3, :] - self.estimated_orbit[0:3, :]))
        #     / len(self.tout)
        # )
        # print("RMSE for position in  : ", RMSE_position)
        # RMSE_vel = np.sqrt(
        #     (np.linalg.norm(self.orbit_states[3:6, :] - self.estimated_orbit[3:6, :]))
        #     / len(self.tout)
        # )
        # print("RMSE for velocity in  : ", RMSE_vel)

    def error_plot(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            (self.gt_attitude_states[0, :] - self.est_attitude_states[0, :]),
            label="error in q1",
            color="b",
        )
        # axs[0].plot(
        #     self.tout,
        #     3 * self.sigma_values[0, :],
        #     label="3σ upper bound",
        #     color="g",
        #     linestyle="--",
        # )
        # axs[0].plot(
        #     self.tout,
        #     -3 * self.sigma_values[0, :],
        #     label="3σ lower bound",
        #     color="r",
        #     linestyle="--",
        # )
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("value")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            (self.gt_attitude_states[1, :] - self.est_attitude_states[1, :]),
            label="error in q2",
            color="b",
        )
        # axs[1].plot(
        #     self.tout,
        #     3 * self.sigma_values[1, :],
        #     label="3σ upper bound",
        #     color="g",
        #     linestyle="--",
        # )
        # axs[1].plot(
        #     self.tout,
        #     -3 * self.sigma_values[1, :],
        #     label="3σ lower bound",
        #     color="r",
        #     linestyle="--",
        # )
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("value")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            (self.gt_attitude_states[2, :] - self.est_attitude_states[2, :]),
            label="error in q3",
            color="b",
        )
        # axs[2].plot(
        #     self.tout,
        #     3 * self.sigma_values[2, :],
        #     label="3σ upper bound",
        #     color="g",
        #     linestyle="--",
        # )
        # axs[2].plot(
        #     self.tout,
        #     -3 * self.sigma_values[2, :],
        #     label="3σ lower bound",
        #     color="r",
        #     linestyle="--",
        # )
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("value")
        axs[2].legend()

        axs[3].plot(
            self.tout,
            (self.gt_attitude_states[3, :] - self.est_attitude_states[3, :]),
            label="error in q4",
            color="b",
        )
        # axs[3].plot(
        #     self.tout,
        #     3 * self.sigma_values[3, :],
        #     label="3σ upper bound",
        #     color="g",
        #     linestyle="--",
        # )
        # axs[3].plot(
        #     self.tout,
        #     -3 * self.sigma_values[3, :],
        #     label="3σ lower bound",
        #     color="r",
        #     linestyle="--",
        # )
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Value")
        axs[3].legend()
        plt.show()

    def quat_plot(self):
        fig, axs = plt.subplots(5, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            self.est_attitude_states[0, :-1],
            label="q1 estimates",
            color="b",
        )
        axs[0].plot(
            self.tout,
            self.gt_attitude_states[0, :],
            label="q1 ",
            color="c",
        )

        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("q1")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            self.est_attitude_states[1, :-1],
            label="q2 estimates",
            color="b",
        )
        axs[1].plot(
            self.tout,
            self.gt_attitude_states[1, :],
            label="q2",
            color="c",
        )

        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("q2")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            self.est_attitude_states[2, :-1],
            label="q3 estimates",
            color="b",
        )
        axs[2].plot(
            self.tout,
            self.gt_attitude_states[2, :],
            label="q3",
            color="c",
        )

        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("q4")
        axs[2].legend()

        axs[3].plot(
            self.tout,
            self.est_attitude_states[3, :-1],
            label="q4 estimates",
            color="b",
        )
        axs[3].plot(
            self.tout,
            self.gt_attitude_states[3, :],
            label="q4",
            color="c",
        )

        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("q4")
        axs[3].legend()

        axs[4].plot(
            self.tout,
            self.est_attitude_states[4, :-1],
            label="b1 estimates",
            color="b",
        )
        axs[4].plot(
            self.tout,
            self.est_attitude_states[5, :-1],
            label="b2",
            color="c",
        )
        axs[4].plot(
            self.tout,
            self.est_attitude_states[6, :-1],
            label="b3",
            color="r",
        )

        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("bias")
        axs[4].legend()
        plt.show()


if __name__ == "__main__":
    attitude_EKF = attitude_estimation()
    attitude_EKF.EKF_estimation()
    attitude_EKF.quat_plot()
    # attitude_EKF.error_plot()
