import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm


class orbit_estimation:
    def __init__(self):
        R = 6371  # Km
        M = 5.972e24  # kg
        G = 6.67e-11 * 1e-9
        self.mu = G * M
        altitude = 600  # Km
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
        self.init_orbit = np.array([x0, y0, z0, xdot0, ydot0, zdot0, 0, 0, 0])

        period = 2 * np.pi / (np.sqrt(self.mu)) * semi_major ** (3 / 2)
        number_of_orbits = 1
        tfinal = period * number_of_orbits
        self.timestep = 1
        self.tout = np.arange(0, tfinal, self.timestep, dtype=int)
        self.ag = np.zeros([3, 1])
        self.ad = np.zeros([3, 1])
        self.aj2 = np.zeros([3, 1])

    def two_body_ode(self, state, mu, Qe):
        """
        process model with gravity , darg and unmodelled accelerations
        input : current state
        output : change in state
        """
        vel = state[3:6]
        position = state[0:3]
        ag = -(mu * position) / (
            np.linalg.norm(position)
            * np.linalg.norm(position)
            * np.linalg.norm(position)
        )
        Cd = 1
        A = 0.00001 * 0.00001  # 10cm x 10cm
        p = 5e-11 * 1000000000  # kg/km3
        We = 7.2921159e-5
        W = np.array([0, 0, We])
        Vrel = vel - np.cross(W, position)
        ad = -0.5 * Cd * A * p * np.linalg.norm(Vrel) * Vrel
        a = ag + ad + state[6:]
        e = np.random.multivariate_normal(np.array([0, 0, 0]).T, Qe)
        state = np.array([vel[0], vel[1], vel[2], a[0], a[1], a[2], e[0], e[1], e[2]]).T
        return state

    def RK4_step(self, f, state, Qe):
        """
        input : state and dynamic function to be used
        output : next state
        """
        M = 5.972e24
        G = 6.67e-11 * (1e-9)
        self.mu = G * M
        k1 = f(state, self.mu, Qe)
        k2 = f(state + 0.5 * k1 * self.timestep, self.mu, Qe)
        k3 = f(state + 0.5 * k2 * self.timestep, self.mu, Qe)
        k4 = f(state + k3 * self.timestep, self.mu, Qe)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state + self.timestep * k

    def jacobian(self, state, Qe):
        """
        input : curretn state
        output : discretised A matrix
        """
        vel = state[3:6]
        position = state[0:3]
        ag = -(self.mu * position) / (
            np.linalg.norm(position)
            * np.linalg.norm(position)
            * np.linalg.norm(position)
        )
        Cd = 1
        A = 0.00001 * 0.00001  # 10cm x 10cm
        p = 5e-11 * 1000000000  # kg/km3
        We = 7.2921159e-5
        W = np.array([0, 0, We])
        Vrel = vel - np.cross(W, position)
        ad = -0.5 * Cd * A * p * np.linalg.norm(Vrel) * Vrel
        a = ag
        self.ag = np.hstack([self.ag, np.array([ag]).T])
        self.ad = np.hstack([self.ad, np.array([ad]).T])
        A = np.zeros((9, 9))
        A[0, 3] = A[1, 4] = A[2, 5] = 1
        A[3, 0] = a[0]
        A[4, 1] = a[1]
        A[5, 2] = a[2]
        A[3, 6] = 1
        A[4, 7] = 1
        A[5, 8] = 1
        A[6, 6] = 0
        A[7, 7] = 0
        A[8, 8] = 0

        return expm(A)

    # EKF
    def EKF_estimation(self):
        data = np.load("orbit_states.npy", allow_pickle=True)
        self.orbit_states, self.orbit_states_with_errors = data

        self.estimated_orbit = np.zeros([6, len(self.tout)])
        self.unmod_acc_est = np.zeros([3, len(self.tout)])
        self.sigma_values = np.zeros([9, len(self.tout)])
        Qe = np.eye(3) * 1e-10  # cov for unmodelled acc

        P_sensor = np.cov(self.orbit_states_with_errors - self.orbit_states)

        # initialise
        P_v2 = P_sensor
        x_init = self.orbit_states_with_errors[:, 0]
        x_init[6:] = 1e-6  # putting unmodelled acc in states

        x = x_init
        C = np.block([np.eye(3), np.zeros([3, 6])])  # C = [I 0]
        W = np.zeros([9, 9])  # process noise cov
        W[0:3, 0:3] = np.eye(3) * 1e-5  # tune for position
        W[3:6, 3:6] = np.eye(3) * 1e-10  # tune for velocity
        W[6:, 6:] = np.eye(3) * 1e-10  # tune for unmodeled acc
        V = np.eye(3) * 1e-3  # measurement noise cov was 100

        steps = 0

        while steps < len(self.tout):
            A = self.jacobian(x, Qe)
            if np.isnan(A).any():
                print("************* found nan in A  value *************** ")
                break
            x_next = self.RK4_step(self.two_body_ode, x, Qe)

            wk = np.random.multivariate_normal(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).T, W
            )
            x_pred = x_next  # + wk
            P_pred = A @ P_v2 @ A.T + W  # 9x9

            if np.isnan(x_next).any():
                print("************* found nan value in x pred *************** ")
                break

            if np.isnan(P_v2).any():
                print("************* found nan value in P_v2 *************** ")
                break
            # innovation
            z = (
                self.orbit_states_with_errors[:3, steps] - C @ x_pred
            ) + np.random.multivariate_normal(np.array([0, 0, 0]).T, V)
            S = C @ P_pred @ C.T + V
            # gain
            K = P_pred @ C.T @ np.linalg.inv(S)

            # update

            x = x_pred + K @ z
            P_v2 = (np.eye(9) - K @ C) @ P_pred @ (np.eye(9) - K @ C).T + K @ V @ K.T
            self.estimated_orbit[:, steps] = x[:6]
            self.unmod_acc_est[:, steps] = x[6:]
            self.sigma_values[:, steps] = np.sqrt(np.diagonal(P_v2))
            steps = steps + 1

        RMSE_position = np.sqrt(
            (np.linalg.norm(self.orbit_states[0:3, :] - self.estimated_orbit[0:3, :]))
            / len(self.tout)
        )
        print("RMSE for position in  : ", RMSE_position)
        RMSE_vel = np.sqrt(
            (np.linalg.norm(self.orbit_states[3:6, :] - self.estimated_orbit[3:6, :]))
            / len(self.tout)
        )
        print("RMSE for velocity in  : ", RMSE_vel)

    def orbit_plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        # axs[0].plot(
        #     self.tout,
        #     self.estimated_orbit[0, :],
        #     label="x estimates",
        #     color="b",
        # )
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

        # axs[1].plot(
        #     self.tout,
        #     self.estimated_orbit[1, :],
        #     label="y estimates",
        #     color="b",
        # )
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

        # axs[2].plot(
        #     self.tout,
        #     self.estimated_orbit[2, :],
        #     label="z estimates",
        #     color="b",
        # )
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
        fig, axs = plt.subplots(6, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            (self.orbit_states[0, :] - self.estimated_orbit[0, :]),
            label="error in x",
            color="b",
        )
        axs[0].plot(
            self.tout,
            3 * self.sigma_values[0, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[0].plot(
            self.tout,
            -3 * self.sigma_values[0, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[0].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[0].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("distance in x")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            (self.orbit_states[1, :] - self.estimated_orbit[1, :]),
            label="error in y",
            color="b",
        )
        axs[1].plot(
            self.tout,
            3 * self.sigma_values[1, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[1].plot(
            self.tout,
            -3 * self.sigma_values[1, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[1].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[1].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("distance in y")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            (self.orbit_states[2, :] - self.estimated_orbit[2, :]),
            label="error in z",
            color="b",
        )
        axs[2].plot(
            self.tout,
            3 * self.sigma_values[2, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[2].plot(
            self.tout,
            -3 * self.sigma_values[2, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[2].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[2].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("distance in z")
        axs[2].legend()

        axs[3].plot(
            self.tout,
            (self.orbit_states[3, :] - self.estimated_orbit[3, :]),
            label="error in Vx",
            color="b",
        )
        axs[3].plot(
            self.tout,
            3 * self.sigma_values[3, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[3].plot(
            self.tout,
            -3 * self.sigma_values[3, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[0].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[0].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Velocity in x")
        axs[3].legend()

        axs[4].plot(
            self.tout,
            (self.orbit_states[4, :] - self.estimated_orbit[4, :]),
            label="error in Vy",
            color="b",
        )
        axs[4].plot(
            self.tout,
            3 * self.sigma_values[4, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[4].plot(
            self.tout,
            -3 * self.sigma_values[4, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[1].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[1].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Velocity in y")
        axs[4].legend()

        axs[5].plot(
            self.tout,
            (self.orbit_states[5, :] - self.estimated_orbit[5, :]),
            label="error in z",
            color="b",
        )
        axs[5].plot(
            self.tout,
            3 * self.sigma_values[5, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[5].plot(
            self.tout,
            -3 * self.sigma_values[5, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[2].axhline(0.01 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[2].axhline(-0.01 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[5].set_xlabel("Time (s)")
        axs[5].set_ylabel("Velocity in z")
        axs[5].legend()
        plt.show()

    def unmod_acc_plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(
            self.tout,
            (self.unmod_acc_est[0, :]),
            label="acc in x",
            color="b",
        )
        axs[0].plot(
            self.tout,
            3 * self.sigma_values[6, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[0].plot(
            self.tout,
            -3 * self.sigma_values[6, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[0].axhline(1e-7 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[0].axhline(-1e-7 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("unmodeled acc in x")
        axs[0].legend()

        axs[1].plot(
            self.tout,
            (self.unmod_acc_est[1, :]),
            label="acc in y",
            color="b",
        )
        axs[1].plot(
            self.tout,
            3 * self.sigma_values[6, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )

        axs[1].plot(
            self.tout,
            -3 * self.sigma_values[7, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[1].axhline(1e-7 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[1].axhline(-1e-7 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("unmodeled acc in y")
        axs[1].legend()

        axs[2].plot(
            self.tout,
            (self.unmod_acc_est[2, :]),
            label="acc in z",
            color="b",
        )
        axs[2].plot(
            self.tout,
            3 * self.sigma_values[8, :],
            label="3σ upper bound",
            color="g",
            linestyle="--",
        )
        axs[2].plot(
            self.tout,
            -3 * self.sigma_values[8, :],
            label="3σ lower bound",
            color="r",
            linestyle="--",
        )
        # axs[2].axhline(1e-7 * 3, color="r", linestyle="--", label="3σ Upper Bound")
        # axs[2].axhline(-1e-7 * 3, color="g", linestyle="--", label="3σ Lower Bound")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("unmodeled acc in z")
        axs[2].legend()
        plt.show()

    def process_dynamic_plot(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        axs[0, 0].plot(
            self.tout,
            (self.ag[0, 1:]),
            label="ag in x",
            color="b",
        )

        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("gravity in x")
        axs[0, 0].legend()

        axs[0, 1].plot(
            self.tout,
            (self.ad[0, 1:]),
            label="drag in x",
            color="b",
        )

        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("drag in x")
        axs[0, 1].legend()

        axs[1, 0].plot(
            self.tout,
            (self.ag[1, 1:]),
            label="ag in y",
            color="b",
        )

        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("gravity in y")
        axs[1, 0].legend()
        axs[1, 1].plot(
            self.tout,
            (self.ad[1, 1:]),
            label="drag in y",
            color="b",
        )

        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("drag in y")
        axs[1, 1].legend()

        axs[2, 0].plot(
            self.tout,
            (self.ag[2, 1:]),
            label="ag in z",
            color="b",
        )

        axs[2, 0].set_xlabel("Time (s)")
        axs[2, 0].set_ylabel("gravity in z")
        axs[2, 0].legend()

        axs[2, 1].plot(
            self.tout,
            (self.ad[0, 1:]),
            label="drag in z",
            color="b",
        )

        axs[2, 1].set_xlabel("Time (s)")
        axs[2, 1].set_ylabel("drag in z")
        axs[2, 1].legend()

        plt.show()


if __name__ == "__main__":
    orbit_EKF = orbit_estimation()
    orbit_EKF.EKF_estimation()
    # orbit_EKF.orbit_plot()
    # orbit_EKF.error_plot()
    # orbit_EKF.unmod_acc_plot()
    # orbit_EKF.process_dynamic_plot()
