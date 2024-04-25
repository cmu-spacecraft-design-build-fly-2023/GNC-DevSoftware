import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_quaternion(ax, quaternion):
    # Convert quaternion to rotation matrix
    q = np.array([float(x) for x in quaternion.strip().split()])
    q = q / np.linalg.norm(q)  # Normalize quaternion
    q0, q1, q2, q3 = q
    R = np.array([
        [1-2*q2**2-2*q3**2, 2*(q1*q2-q3*q0), 2*(q1*q3+q2*q0)],
        [2*(q1*q2+q3*q0), 1-2*q1**2-2*q3**2, 2*(q2*q3-q1*q0)],
        [2*(q1*q3-q2*q0), 2*(q2*q3+q1*q0), 1-2*q1**2-2*q2**2]
    ])
    
    # Define the arrows (x, y, z) with labels (i, j, k)
    arrows = np.array([
        [[0, 0, 0, 1, 0, 0]],  # Arrow for x-axis
        [[0, 0, 0, 0, 1, 0]],  # Arrow for y-axis
        [[0, 0, 0, 0, 0, 1]]   # Arrow for z-axis
    ])
    labels = ['i', 'j', 'k']
    colors = ['r', 'g', 'b']
    for i, (arrow, label) in enumerate(zip(arrows, labels)):
        ax.quiver(*arrow[0], length=0.1, color=colors[i], normalize=True)
        ax.text(arrow[0][3], arrow[0][4], arrow[0][5], label, color=colors[i], fontsize=10, ha='left', va='bottom')
    
    ax.set_axis_off()  # Turn off axes

# Example of plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_quaternion(ax, "1 0 0 0")  # Example quaternion
plt.show()
