import numpy as np
import cv2

# Create a blank transparent image (300x300) with 4 channels (BGRA)
transparent_image = np.zeros((300, 300, 4), dtype=np.uint8)

# Generate a random transparent color (BGRA)
random_color = np.random.randint(0, 256, size=(4,), dtype=np.uint8)
random_color[3] = np.random.randint(50, 200)  # Set random transparency

# Draw a rectangle with the transparent color
cv2.rectangle(transparent_image, (50, 50), (250, 250), random_color.tolist(), -1)

# Create a white background (300x300) with 3 channels (BGR)
background = np.full((300, 300, 3), (255, 255, 255), dtype=np.uint8)

# Blend the transparent rectangle onto the background using alpha channel
alpha = transparent_image[:, :, 3] / 255.0  # Normalize alpha to range [0,1]
for c in range(3):  # Blend each color channel
    background[:, :, c] = (1 - alpha) * background[:, :, c] + alpha * transparent_image[:, :, c]

# Show the image using OpenCV
cv2.imshow("Transparent Color Visualization", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
