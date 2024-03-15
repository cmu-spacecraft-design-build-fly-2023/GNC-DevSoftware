#Loading an image of the night sky, making the stars brighter and the sky darker

from PIL import ImageEnhance, Image
# Load the image
image_path = "IMG_8825.jpg"  # Replace "night_sky.jpg" with your image file path
image = Image.open(image_path)
# Define a function to enhance the brightness of white pixels
def enhance_white_pixels(pixel_value, enhancement_factor=1.5, threshold=220):
    # If the pixel value is above the threshold (close to white)
    if pixel_value >= threshold:
        # Enhance brightness
        return min(255, int(pixel_value * enhancement_factor))
    else:
        return pixel_value

# Apply the enhancement function to each pixel
enhanced_image = image.point(enhance_white_pixels)
# Define a function to darken the background
def darken_background(pixel_value, darken_factor=0.1, threshold=220):
    # If the pixel value is below the threshold (not close to white)
    if pixel_value < threshold:
        # Darken the pixel value
        return max(0, int(pixel_value * darken_factor))
    else:
        return pixel_value

# Apply the darkening function to each pixel
final_image = enhanced_image.point(darken_background)

# Save or display the modified image
output_path = "enhanced_night_sky.jpg"
final_image.save(output_path)
final_image.show()