import cv2

# Load your image
image_path = '/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/test_data_dir/rgb/0000020-rgb.png'  # Replace with your image path
image = cv2.imread(image_path)

# Bounding box coordinates
# Replace these with your actual coordinates
x, y, width, height = 118, 144, 64, 80  # Example coordinates

# Draw a rectangle around the specified region
# The color (255, 0, 0) is blue in BGR format, and 2 is the thickness
cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

# Save the image
save_path = 'path_to_save_image.png'  # Replace with your desired save path
cv2.imwrite(save_path, image)
