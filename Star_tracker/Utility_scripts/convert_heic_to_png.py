import os

# Function to convert HEIC images to PNG
def convert_heic_to_png(output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the current directory
    current_directory = os.getcwd()

    # Iterate through each file in the current directory
    for filename in os.listdir(current_directory):
        if filename.endswith(".HEIC"):
            heic_path = os.path.join(current_directory, filename)
            # Generate the output PNG file path
            png_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            # Run the conversion command
            os.system(f"heif-convert {heic_path} {png_path}")

# Specify the output folder for PNG images (a directory within the current folder)
output_folder = "output_png"

# Convert HEIC images to PNG
convert_heic_to_png(output_folder)
