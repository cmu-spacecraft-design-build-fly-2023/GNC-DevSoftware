import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pyIGRF
import math


def EulerAngles2Quaternions(phi_theta_psi):
    phi = phi_theta_psi[0]
    theta = phi_theta_psi[1]
    psi = phi_theta_psi[2]

    q_0 = np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    q_1 = np.sin(phi/2) * np.cos(theta/2) * np.cos(psi/2) - np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    q_2 = np.cos(phi/2) * np.sin(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2)
    q_3 = np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.sin(theta/2) * np.cos(psi/2)

    quaternion = np.array([q_0, q_1, q_2, q_3])
    return quaternion

def Quaternions2EulerAngles(q0123_temp):
    q_0 = q0123_temp[:, 0]
    q_1 = q0123_temp[:, 1]
    q_2 = q0123_temp[:, 2]
    q_3 = q0123_temp[:, 3]
    euler = np.zeros((q0123_temp.shape[0], 3))
    euler[:,0] = np.arctan2(2 * (q_0 * q_1 + q_2 * q_3), 1 - 2 * (q_1**2 + q_2**2))  # phi
    euler[:,1] = np.arcsin(2 * (q_0 * q_2 - q_3 * q_1))  # theta
    euler[:,2] = np.arctan2(2 * (q_0 * q_3 + q_1 * q_2), 1 - 2 * (q_2**2 + q_3**2))  # psi
    euler = np.real(euler)
    return euler

def Quaternions2EulerAngles_single(q0123_temp):
    q_0 = q0123_temp[0]
    q_1 = q0123_temp[1]
    q_2 = q0123_temp[2]
    q_3 = q0123_temp[3]
    euler = np.zeros(3)
    euler[0] = np.arctan2(2 * (q_0 * q_1 + q_2 * q_3), 1 - 2 * (q_1**2 + q_2**2))  # phi
    euler[1] = np.arcsin(2 * (q_0 * q_2 - q_3 * q_1))  # theta
    euler[2] = np.arctan2(2 * (q_0 * q_3 + q_1 * q_2), 1 - 2 * (q_2**2 + q_3**2))  # psi
    euler = np.real(euler)
    return euler
def TIB(phi,theta,psi):
    ct = np.cos(theta)
    st = np.sin(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    ss = np.sin(psi)
    cs = np.cos(psi)

    out = np.array([[ct*cs,sp*st*cs-cp*ss,cp*st*cs+sp*ss],
        [ct*ss,sp*st*ss+cp*cs,cp*st*ss-sp*cs],
        [-st,sp*ct,cp*ct]])
    return out

def TIBquat(q0123_temp):
    q_0 = q0123_temp[0]
    q_1 = q0123_temp[1]
    q_2 = q0123_temp[2]
    q_3 = q0123_temp[3]

    q0s = q_0 ** 2
    q1s = q_1 ** 2
    q2s = q_2 ** 2
    q3s = q_3 ** 2

    R = np.array([[(q0s + q1s - q2s - q3s),2 * (q_1 * q_2 - q_0 * q_3), 2 * (q_0 * q_2 + q_1 * q_3)],
    [2 * (q_1 * q_2 + q_0 * q_3), (q0s - q1s + q2s - q3s), 2 * (q_2 * q_3 - q_0 * q_1)],
    [2 * (q_1 * q_3 - q_0 * q_2), 2 * (q_0 * q_1 + q_2 * q_3), (q0s - q1s - q2s + q3s)]])
    return R

k = 67200
n_turns = 84  # no. of turns
Area = 0.02  # Area in meters ^ 2
k_nA = 67200/(84*0.02)
nA = 84*0.02
maxCurrent = 120  # mAmps
R = 6.371e6 #meters
M = 5.972e24 #kg
G = 6.67e-11
mu = G*M
print('mu',mu)
#inertia and mass
ms = 2.6 #kilograms
lx = 10/100 #meters
ly = 10/100 #meters
lz = 20/100 #meters
l = np.array((lx,ly,lz))
CD = 1.0
Is = np.array([[0.9,0,0],[0,0.9,0],[0,0,0.3]])
invI = LA.inv(Is)

#Initial Conditions Position and Velocity
altitude = 600*1000 #meters
x0 = R + altitude
y0 = 0
z0 = 0
xdot0 = 0
inclination = 51.6*np.pi/180
semi_major = LA.norm([x0,y0,z0])
vcircular = np.sqrt(mu/semi_major)
ydot0 = vcircular*np.cos(inclination)
zdot0 = vcircular*np.sin(inclination)
period = 2*np.pi/(np.sqrt(mu))*semi_major**(3/2)

#Intitial Conditions for Attitude and Angular Velocity
phi0 = 0
theta0 = 0
psi0 = 0
ptp0 = np.array([phi0,theta0,psi0])
q0123_0 = EulerAngles2Quaternions(ptp0)
p0 = 0.5
q0 = -0.02
r0 = 0.03

#initial_state = np.array([[x0],[y0],[z0],[xdot0],[ydot0],[zdot0]])

initial_state_f = np.array([x0,y0,z0,xdot0,ydot0,zdot0,q0123_0[0],q0123_0[1],q0123_0[2],q0123_0[3],p0,q0,r0])
print('init_state',initial_state_f.shape)
Bx_list = []
By_list = []
Bz_list = []
BBx_list = []
BBy_list = []
BBz_list = []
BB_meas_x_list = []
BB_meas_y_list = []
BB_meas_z_list = []
p_meas_list = []
q_meas_list = []
r_meas_list = []
nextSensorUpdate = 1
lastSensorUpdate = 0.0
fsensor=1

MagscaleNoise = 1e-5 * fsensor
MagFieldNoise = MagscaleNoise * (2 * np.random.rand() - 1)  # Random number between -1 and 1

AngscaleNoise = 0.001 * fsensor
AngFieldNoise = AngscaleNoise * (2 * np.random.rand() - 1)  # Random number between -1 and 1

EulerScaleNoise = 1 * np.pi / 180 * fsensor
EulerNoise = EulerScaleNoise * (2 * np.random.rand() - 1)  # Random number between -1 and 1

MagscaleBias = 4e-7 * fsensor
MagFieldBias = MagscaleBias * (2 * np.random.rand() - 1)  # Random number between -1 and 1

AngscaleBias = 0.01 * fsensor
AngFieldBias = AngscaleBias * (2 * np.random.rand() - 1)  # Random number between -1 and 1

EulerBias = 2 * np.pi / 180 * fsensor
EulerBias = EulerBias * (2 * np.random.rand() - 1)  # Random number between -1 and 1


def Control(BfieldM,pqrM):
    #print('BfieldM',BfieldM)
    current = (1000000*np.cross(pqrM, BfieldM))/(5)
    return current

def Sensor(BB_temp, pqr_temp, ptp_temp):
    #for idx in np.arange(3):
    BB_temp = BB_temp + MagFieldBias + MagFieldNoise
    pqr_temp = pqr_temp + AngFieldBias + AngFieldNoise
    ptp_temp = ptp_temp + EulerBias + EulerNoise
    return BB_temp, pqr_temp, ptp_temp

def Satellite(t, state):
    global Bx, By, Bz, BB_f, InvI, lastSensorUpdate, BfieldMeasured, pqrMeasured, ptpMeasured
    vel = state[3:6]

    #rotational kinematics
    q0123 = state[6:10]
    p = state[10]
    q = state[11]
    r = state[12]
    pqr = state[10:13]
    ptp = Quaternions2EulerAngles_single(q0123)
    PQRMAT = np.array([[0, - p, - q, - r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]])
    q0123dot = 0.5*np.matmul(PQRMAT, q0123)
    xyz = state[0:3]
    x = state[0]
    y = state[1]
    z = state[2]
    rho = LA.norm(xyz)
    rhat = xyz/rho
    phiE = 0
    thetaE = np.arccos(z / rho)
    psiE = math.atan2(y, x)
    latitude = 90 - thetaE * 180 / np.pi
    longitude = psiE * 180 / np.pi
    rhokm = rho / 1000
    _, _, _, BN, BE, BD, _ = pyIGRF.igrf_value(latitude, longitude, rhokm, 2020)
    BNED = np.array([[BN], [BE], [-BD]])
    BI = np.matmul(TIB(phiE, thetaE+np.pi, psiE), BNED)
    Bx = BI[0]
    By = BI[1]
    Bz = BI[2]
    BB_f = np.matmul(TIBquat(q0123).T, BI)
    BB_f = BB_f * 1e-9
    #print('BB_f', BB_f)

    #if (t >= lastSensorUpdate):
    #    lastSensorUpdate = lastSensorUpdate + nextSensorUpdate
    #    BfieldMeasured, pqrMeasured, ptpMeasured = Sensor(np.reshape(BB_f,(3,)), np.reshape(pqr,(3,)), np.reshape(ptp,(3,)))

    current = Control(np.reshape(BB_f,(3,)), pqr)
    #if np.sum(np.abs(current)) > maxCurrent / 1000:
    #    current = (current / np.sum(np.abs(current))) * maxCurrent / 1000
    muB = current * 5
    #print('current',current)
    LMN_magtorquers = np.cross(muB, np.reshape(BB_f,(3,)))

    #LMN_magtorquers = np.array([0,0,0])
    #print('LMN_magtorquers', LMN_magtorquers.shape)
    H = np.matmul(Is, pqr)
    #Rotational dynamics
    pqrdot = np.matmul(invI, (LMN_magtorquers - np.cross(pqr, H)))
    #print(pqrdot)
    #print('cross',np.cross(pqr, H).shape)
    F_grav = -(mu*ms/(rho**2))*rhat
    F = F_grav
    acc = F/ms
    #print('pqrdot',pqrdot.shape)
    dstatedt = np.concatenate((vel, acc, q0123dot, pqrdot))
    #print('d',dstatedt.shape)
    return dstatedt

#Need time window
period = 2*np.pi/(np.sqrt(mu))*semi_major**(3/2)
print('period',period)
number_of_orbits = 25
tfinal = period*number_of_orbits
timestep = 1
tout = np.arange(tfinal)
""""
sol = solve_ivp(Satellite, [0, tfinal], initial_state_f,t_eval = tout,method = 'DOP853')
print('sol',sol.t.shape)
states = np.array(sol.y).T
time = np.array(sol.t, dtype = int)
print(time)
print(states.shape)
"""

tout = np.arange(0,tfinal,timestep,dtype = int)
time = tout
print('tout',tout)
state = initial_state_f
lastPrint = 0
next = 1
stateout = np.zeros((len(tout),13))
print('stateout',stateout.shape)
for idx in range(len(tout)):
    #print(idx)
    if idx > lastPrint:
        print(['Time = ',idx,' out of ',tfinal])
        lastPrint = lastPrint + next
    k1 = Satellite(idx, state)
    k2 = Satellite(idx + timestep / 2, state + k1 * timestep / 2)
    k3 = Satellite(idx + timestep / 2, state + k2 * timestep / 2)
    k4 = Satellite(idx + timestep, state + k3 * timestep)
    k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    state = state + k * timestep
    stateout[idx] = state
    Bx_list.append(Bx)
    By_list.append(By)
    Bz_list.append(Bz)
    BBx_list.append(BB_f[0])
    BBy_list.append(BB_f[1])
    BBz_list.append(BB_f[2])
print('stateout',stateout.shape)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
# plotting
#ax.set_title('Orbit')
x = stateout[:,0]/1000
y = stateout[:,1]/1000
z = stateout[:,2]/1000

pF = stateout[:,10]
qF = stateout[:,11]
rF = stateout[:,12]

#ax.plot3D(x, y, z, 'green', linewidth=3)
# Create a grid of points on the unit sphere
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

# Parametric equations for a sphere
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

#ax.plot_surface(x*R/1000, y*R/1000, z*R/1000, cmap=plt.cm.YlGnBu_r)

print(len(Bx_list))
print(len(By_list))
print(len(Bz_list))
"""
ax = fig.add_subplot(111)
ax.plot(time, Bx_list, label = 'x')
ax.plot(time, By_list, label = 'y')
ax.plot(time, Bz_list, label = 'z')
ax.legend()
plt.show()
"""

fig, (ax1, ax2,ax3) = plt.subplots(3)

ax1.plot(time, BBx_list, label = 'x')
ax1.plot(time, BBy_list, label = 'y')
ax1.plot(time, BBz_list, label = 'z')
#ax1.plot(time, BB_meas_x_list,"--", label = 'x')
#ax1.plot(time, BB_meas_y_list,"--", label = 'y')
#ax1.plot(time, BB_meas_z_list,"--", label = 'z')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Mag Field (T)')
ax1.legend()



ax2.plot(time, pF, label = 'p')
ax2.plot(time, qF, label = 'q')
ax2.plot(time, rF, label = 'r')
#ax2.plot(time, p_meas_list, '--',label = 'p_meas')
#ax2.plot(time, q_meas_list, '--', label = 'q_meas')
#ax2.plot(time, r_meas_list, '--', label = 'r_meas')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.legend()

q1 = stateout[:,7]
q2 = stateout[:,8]
q3 = stateout[:,9]
"""
ax3.plot(time, q1, label = 'q1')
ax3.plot(time, q2, label = 'q2')
ax3.plot(time, q3, label = 'q3')
ax3.legend()
"""
print('stateout_q',stateout[:,6:10].shape)
ptp = Quaternions2EulerAngles(stateout[:,6:10])
print('ptp',ptp.shape)

ax3.plot(time, ptp[:,0]*180/np.pi, label = 'q1')
ax3.plot(time, ptp[:,1]*180/np.pi, label = 'q2')
ax3.plot(time, ptp[:,2]*180/np.pi, label = 'q3')
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Euler Angles (deg)')
ax3.legend()

fig, (ax1) = plt.subplots(1)
ax1.plot(time, pF, label = 'p')
ax1.plot(time, qF, label = 'q')
ax1.plot(time, rF, label = 'r')
#ax2.plot(time, p_meas_list, '--',label = 'p_meas')
#ax2.plot(time, q_meas_list, '--', label = 'q_meas')
#ax2.plot(time, r_meas_list, '--', label = 'r_meas')
ax1.legend()
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Angular Velocity (rad/s)')
plt.show()

