from PIL import Image
import os

# Function to convert JPEG to PNG
def convert_jpg_to_png(jpg_file, output_folder):
    try:
        # Open the JPEG image
        with Image.open(jpg_file) as img:
            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # Generate the output PNG file path
            png_file = os.path.join(output_folder, os.path.splitext(os.path.basename(jpg_file))[0] + ".png")
            # Convert and save as PNG
            img.save(png_file, "PNG")
            print(f"{jpg_file} converted to PNG successfully.")
    except Exception as e:
        print(f"Error converting {jpg_file} to PNG: {e}")

# Function to convert all JPEGs in a folder
def convert_all_jpgs_to_png(input_folder, output_folder):
    # List all files in the input folder
    files = os.listdir(input_folder)
    # Filter JPEG files
    jpg_files = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    
    if not jpg_files:
        print("No JPEG files found in the input folder.")
        return
    
    # Convert each JPEG file to PNG
    for jpg_file in jpg_files:
        jpg_path = os.path.join(input_folder, jpg_file)
        convert_jpg_to_png(jpg_path, output_folder)

# Example usage
current_directory = os.getcwd()
input_folder = os.path.join(current_directory, "Demo")
convert_all_jpgs_to_png(input_folder, input_folder)
