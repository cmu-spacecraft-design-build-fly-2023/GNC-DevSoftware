import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

# Function to normalize a quaternion
def normalize(q):
    norm = LA.norm(q)
    if norm == 0:
        return q
    return q / norm

# Example quaternion
q = np.array([0.5, 0.5, 0.5, 0.5])  # Example quaternion (a, b, c, d)

# Normalize quaternion
q_normalized = normalize(q)

# Extract components
a = q_normalized[0]
b, c, d = q_normalized[1:]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot vector part
ax.quiver(0, 0, 0, b, c, d, color='b', label='Quaternion Vector Part')

# Plot reference axes
ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.3)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.3)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.3)

# Set plot limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set title
plt.title('Quaternion Visualization')

# Show plot
plt.show()
