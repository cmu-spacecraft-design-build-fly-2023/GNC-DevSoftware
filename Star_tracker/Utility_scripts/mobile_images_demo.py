import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Function to extract quaternion components from attitude data
def extract_quaternion(attitude):
    lines = attitude.strip().split('\n')
    q0 = float(lines[0].split()[-1])
    q1 = float(lines[1].split()[-1])
    q2 = float(lines[2].split()[-1])
    q3 = float(lines[3].split()[-1])
    
    # Normalize quaternion to have unit norm
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0 /= norm
    q1 /= norm
    q2 /= norm
    q3 /= norm
    
    return q0, q1, q2, q3
def plot_quaternion(ax, attitude):
    if not attitude.strip():  # Check if attitude data is empty
        return
    
    lines = attitude.strip().split('\n')
    q0 = float(lines[0].split()[-1])
    q1 = float(lines[1].split()[-1])
    q2 = float(lines[2].split()[-1])
    q3 = float(lines[3].split()[-1])
    
    # Normalize quaternion to have unit norm
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0 /= norm
    q1 /= norm
    q2 /= norm
    q3 /= norm
    
    # Define the arrow directions based on the quaternion components
    arrows = np.array([
        [[0, 0, 0, q0, 0, 0]],  # i
        [[0, 0, 0, 0, q1, 0]],  # j
        [[0, 0, 0, 0, 0, q2]]   # k
    ])
    
    # Define colors for the arrows
    colors = ['r', 'g', 'b']
    
    # Plot each arrow
    for i, (arrow, color) in enumerate(zip(arrows, colors)):
        ax.quiver(*arrow[0], length=0.1, color=color, normalize=True)

# Function to plot the quaternion arrows
def plot_quaternion_t(ax, q0, q1, q2, q3):
    # Define the arrow directions based on the quaternion components
    arrows = np.array([
        [[0, 0, 0, q0, 0, 0]],  # i
        [[0, 0, 0, 0, q1, 0]],  # j
        [[0, 0, 0, 0, 0, q2]]   # k
    ])
    
    # Define colors for the arrows
    colors = ['r', 'g', 'b']
    
    # Plot each arrow
    for i, (arrow, color) in enumerate(zip(arrows, colors)):
        ax.quiver(*arrow[0], length=0.1, color=color, normalize=True)

# Function to run the command for each image and read the attitude data
def process_image(image_path):
    command = f"./lost pipeline \
        --png {image_path} \
        --focal-length 5.7 \
        --pixel-size 1.85 \
        --centroid-algo cog \
        --centroid-mag-filter 5 \
        --database my-database.dat \
        --star-id-algo py \
        --angular-tolerance 0.05 \
        --false-stars 1000 \
        --max-mismatch-prob 0.0001 \
        --attitude-algo dqm \
        --print-attitude attitude.txt \
        --plot-output annotated-{os.path.basename(image_path)}"
    os.system(command)
    
    with open("attitude.txt", "r") as file:
        all_lines = file.readlines()
        last_four_lines = ''.join(all_lines[-4:])
        # Clear the attitude.txt file
    with open("attitude.txt", "w"):
        pass
    return last_four_lines

# Path to the folder containing images
current_directory = os.getcwd()
image_folder = os.path.join(current_directory, "Pictures_from_Ashley")

# Preparing the figure for plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))  # Adjust nrows and ncols based on your needs
axs = axs.flatten()  # Flatten the array to make it easier to iterate
current_ax = 0

# Iterate through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png") and current_ax < len(axs):
        image_path = os.path.join(image_folder, filename)
        attitude = process_image(image_path)
        print(f"Attitude for {image_path}: {attitude}")
        
        # Load the image using PIL and plot it
        img = Image.open(image_path)
        axs[current_ax].imshow(img)
        #axs[current_ax].set_title(f"Attitude: {attitude}")
        axs[current_ax].set_title(filename)
        axs[current_ax].axis('off')  # Turn off axis
        
        # Create a subplot for quaternion plot
        ax_quaternion = axs[current_ax].inset_axes([0.7, 0.1, 0.2, 0.2], projection='3d')  # Adjust coordinates here
        ax_quaternion.patch.set_alpha(0.01)  # Set transparent background
        ax_quaternion.axis('off')
        
        # Extract quaternion components and plot arrows
        #q0, q1, q2, q3 = extract_quaternion(attitude)
        plot_quaternion(ax_quaternion, attitude)
        
        # Set aspect ratio to 'equal' for uniform scaling
        ax_quaternion.set_box_aspect([1, 1, 1])
        
        # Move to the next subplot
        current_ax += 1

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
