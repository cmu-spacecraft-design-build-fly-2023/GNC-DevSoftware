import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import sympy as sp
from scipy.linalg import expm

# import transforms3d.quaternions as quaternions
from transforms3d.euler import euler2quat, quat2euler


class orbit_estimation:
    def __init__(self):
        R = 6.371e6  # 6  # meters
        M = 5.972e24  # kg
        G = 6.67e-11
        self.mu = G * M
        altitude = 600 * 1000  # meters
        x0 = R + altitude
        y0 = 0
        z0 = 0
        xdot0 = 0
        inclination = 51.6 * np.pi / 180
        semi_major = np.linalg.norm([x0, y0, z0])

        vcircular = np.sqrt(self.mu / semi_major)
        ydot0 = vcircular * np.cos(inclination)
        zdot0 = vcircular * np.sin(inclination)
        period = 2 * np.pi / (np.sqrt(self.mu)) * semi_major ** (3 / 2)
        self.init_orbit = np.array([x0, y0, z0, xdot0, ydot0, zdot0])

        period = 2 * np.pi / (np.sqrt(self.mu)) * semi_major ** (3 / 2)
        # print("period", period)
        number_of_orbits = 1
        tfinal = period * number_of_orbits
        self.timestep = 1
        self.tout = np.arange(0, tfinal, self.timestep, dtype=int)

    def two_body_ode(self, state, mu):
        vel = state[3:]
        position = state[0:3]
        a = -(mu * position) / (
            np.linalg.norm(position)
            * np.linalg.norm(position)
            * np.linalg.norm(position)
        )
        state = np.array([vel[0], vel[1], vel[2], a[0], a[1], a[2]]).T
        return state

    def RK4_step(self, f, state):
        M = 5.972e24  # kg
        G = 6.67e-11 * (1e-9)
        self.mu = G * M
        k1 = f(state, self.mu)
        k2 = f(state + 0.5 * k1 * self.timestep, self.mu)
        k3 = f(state + 0.5 * k2 * self.timestep, self.mu)
        k4 = f(state + k3 * self.timestep, self.mu)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state + self.timestep * k

    def get_A_matrix(self, x):
        epsilon = 1e-6
        next_x = self.two_body_ode(x, self.mu)
        Jacobian = np.zeros((len(next_x), len(x)))

        # Numerical differentiation to approximate A
        for i in range(len(x)):
            # Create a perturbed state vector
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            x_next_perturbed = self.two_body_ode(x_perturbed, self.mu)
            # Calculate the perturbed ẋ (xdot)
            xdot_perturbed = (x_next_perturbed - next_x) / epsilon

            # Calculate the i-th column of A
            Jacobian[:, i] = xdot_perturbed
        return Jacobian

    def jacobian(self, state):
        next_state = self.two_body_ode(state, self.mu)
        a = next_state[3:]
        A = np.zeros((6, 6))
        A[0, 3] = A[1, 4] = A[2, 5] = 1
        A[3, 0] = a[0]
        A[4, 1] = a[1]
        A[5, 2] = a[2]
        return expm(A)

    # EKF
    def EKF_estimation(self):
        # get orbit propogation
        self.orbit_states = np.zeros([6, len(self.tout)])
        self.orbit_states[:, 0] = self.init_orbit / 1000
        self.orbit_states_with_errors = np.zeros([6, len(self.tout)])
        self.estimated_orbit = np.zeros([6, len(self.tout)])

        for i in range(len(self.tout) - 1):
            self.orbit_states[:, i + 1] = self.RK4_step(
                self.two_body_ode, self.orbit_states[:, i]
            )

        for i in range(len(self.tout)):
            noise = np.random.normal(0, 0.01, 3)
            self.orbit_states_with_errors[:3, i] = self.orbit_states[:3, i] + noise

        P_sensor = np.cov(self.orbit_states_with_errors - self.orbit_states)
        print("first sensor measurement : \n", self.orbit_states_with_errors[:, 0])
        print("first gt measurement : \n", self.orbit_states[:, 0])
        # initialise

        x = self.orbit_states_with_errors[:, 0] + 10
        P = P_sensor  # Initial covariance matrix
        C = np.block([np.eye(3), np.zeros([3, 3])])  # C = [I 0]
        # # dt = 1  # Time step
        W = np.eye(6) * 10000  # Process noise covariance
        V = np.eye(3) * 100  # measurement noise cov

        steps = 0

        while steps < len(self.tout):
            A = self.jacobian(x)  # 6x6
            x_pred = self.RK4_step(self.two_body_ode, x)
            P_pred = A @ P @ A.T + W  # 6x6

            # innovation
            z = self.orbit_states_with_errors[:3, steps] - C @ x_pred  # 3x1
            S = C @ P_pred @ C.T + V  # 3x3
            # gain
            K = P_pred @ C.T @ np.linalg.inv(S)  # 6x3
            # update

            x = x_pred + K @ z  # 6x1
            P = (np.eye(6) - K @ C) @ P_pred @ (
                np.eye(6) - K @ C
            ).T + K @ V @ K.T  # 6x6
            self.estimated_orbit[:, steps] = x
            steps = steps + 1

    def orbit_plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            self.estimated_orbit[0, :],
            label="x estimates",
            color="b",
        )
        axs[0].plot(
            self.tout,
            self.orbit_states[0, :],
            label="x positions",
            color="c",
        )
        axs[0].scatter(
            self.tout,
            self.orbit_states_with_errors[0, :],
            label="sensor measurement",
            color="g",
            s=1,
        )
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("x")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            self.estimated_orbit[1, :],
            label="y estimates",
            color="b",
        )
        axs[1].plot(
            self.tout,
            self.orbit_states[1, :],
            label="y positions",
            color="c",
        )
        axs[1].scatter(
            self.tout,
            self.orbit_states_with_errors[1, :],
            label="sensor measurement",
            color="g",
            s=1,
        )
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("y")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            self.estimated_orbit[2, :],
            label="z estimates",
            color="b",
        )
        axs[2].plot(
            self.tout,
            self.orbit_states[2, :],
            label="z positions",
            color="c",
        )
        axs[2].scatter(
            self.tout,
            self.orbit_states_with_errors[2, :],
            label="sensor measurement",
            color="g",
            s=1,
        )
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("z")
        axs[2].legend()
        plt.show()

    def error_plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            (self.orbit_states[0, :] - self.estimated_orbit[0, :]),
            label="error in x",
            color="b",
        )
        axs[0].axhline(0.01 * 2, color="r", linestyle="--", label="2σ Upper Bound")
        axs[0].axhline(-0.01 * 2, color="g", linestyle="--", label="2σ Lower Bound")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("x")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            (self.orbit_states[1, :] - self.estimated_orbit[1, :]),
            label="error in y",
            color="b",
        )
        axs[1].axhline(0.01 * 2, color="r", linestyle="--", label="2σ Upper Bound")
        axs[1].axhline(-0.01 * 2, color="g", linestyle="--", label="2σ Lower Bound")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("y")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            (self.orbit_states[2, :] - self.estimated_orbit[2, :]),
            label="error in z",
            color="b",
        )
        axs[2].axhline(0.01 * 2, color="r", linestyle="--", label="2σ Upper Bound")
        axs[2].axhline(-0.01 * 2, color="g", linestyle="--", label="2σ Lower Bound")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("z")
        axs[2].legend()
        plt.show()


if __name__ == "__main__":
    orbit_EKF = orbit_estimation()
    orbit_EKF.EKF_estimation()
    # orbit_EKF.orbit_plot( )
    orbit_EKF.error_plot()
