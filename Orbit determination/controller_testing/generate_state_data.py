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

    def two_body_ode_ground_truth(self, state, mu, Qe):
        """
        ground truth model with acceleration due to gravity, drag and J2
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
        p = 5e-11 * 1000000000  # kg/m3
        We = 7.2921159e-5
        W = np.array([0, 0, We])
        Vrel = vel - np.cross(W, position)
        J2 = 1082 * 1e-6
        R = 6371
        r = np.linalg.norm(position)
        X, Y, Z = position
        aj2 = (
            3
            * J2
            * mu
            * R**2
            / r**5
            * (
                (5 * Z**2 / r**2 - 1) * np.array([X, Y, 0])
                + Z * (5 * Z**2 / r**2 - 3) * np.array([0, 0, 1])
            )
        )
        ad = -0.5 * Cd * A * p * np.linalg.norm(Vrel) * Vrel
        a = ag + ad + aj2
        # self.aj2 = np.hstack([self.aj2, np.array([aj2]).T])
        state = np.array([vel[0], vel[1], vel[2], a[0], a[1], a[2]]).T
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

    def EKF_estimation(self):
        self.orbit_states = np.zeros([9, len(self.tout)])
        self.orbit_states[:, 0] = self.init_orbit
        self.orbit_states_with_errors = np.zeros([9, len(self.tout)])

        Qe = np.eye(3) * 1e-10  # cov for unmodelled acc

        # get ground truth orbit propagation
        for i in range(len(self.tout) - 1):
            self.orbit_states[:6, i + 1] = self.RK4_step(
                self.two_body_ode_ground_truth, self.orbit_states[:6, i], Qe
            )
        # orbit sensor measurement
        self.orbit_states_with_errors = self.orbit_states
        for i in range(len(self.tout)):
            noise = np.random.normal(0, 0.01, 3)  # add noise
            self.orbit_states_with_errors[:3, i] = self.orbit_states[:3, i] + noise

        np.save("orbit_states.npy", [self.orbit_states, self.orbit_states_with_errors])


if __name__ == "__main__":
    orbit_EKF = orbit_estimation()
    orbit_EKF.EKF_estimation()
