import cv2
import numpy as np

# Global variables
polygon_points = []  # Stores the polygon points
masking = False  # Flag to check if masking is active
paused = False  # Flag to pause video when 'M' is pressed
mask = None  # Mask image
zone_data = []  # Stores polygons with associated zone names
zone_counter = 0  # Counter to assign zone names like A, B, C, etc.


# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global polygon_points, masking

    if masking:  # Only allow drawing when masking is active
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add points
            polygon_points.append((x, y))


# Function to return the polygon mask as a NumPy array
def get_mask():
    if len(polygon_points) > 2:
        closed_polygon = polygon_points + [polygon_points[0]]  # Auto-close polygon
        return np.array(closed_polygon, dtype=np.int32)
    return np.array([])  # Return an empty array if no mask is drawn


# Load video and resize to 1920x1080
video_path = r"E:\RoboCam\RoboCam\assets\dataset\video-dataset\market-square.mp4"
cap = cv2.VideoCapture(video_path)

# Get original dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize scale to fit 1920x1080
scale_x = 1920 / frame_width
scale_y = 1080 / frame_height

# Create a window
cv2.namedWindow("Polygon Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Polygon Mask", 1920, 1080)
cv2.setMouseCallback("Polygon Mask", draw_polygon)

# Placeholder for mask
mask = None

while True:
    if not paused:  # Read frames only when not paused
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends

        # Resize the frame to 1920x1080
        frame = cv2.resize(frame, (1920, 1080))

        # Initialize the mask once with the correct size
        if mask is None:
            mask = np.zeros_like(frame, dtype=np.uint8)

    temp_frame = frame.copy()  # Keep original frame unchanged

    # Draw polygon outline while masking is active
    if masking and len(polygon_points) > 1:
        cv2.polylines(temp_frame, [np.array(polygon_points, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    # When 'M' is pressed again, complete the polygon and mark it
    if not masking and len(polygon_points) > 2:
        polygon_points.append(polygon_points[0])  # Auto-connect first and last points
        cv2.fillPoly(mask, [np.array(polygon_points, np.int32)], (0, 0, 255))  # Red fill
        cv2.polylines(mask, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Assign the current polygon a zone name (e.g., Zone A, Zone B, etc.)
        zone_name = f"Zone {chr(65 + zone_counter)}"  # Generate zone name (Zone A, Zone B, ...)
        zone_data.append({"zone_name": zone_name, "polygon": polygon_points})  # Store the zone and polygon

        # Display the zone name on the frame
        text_position = (polygon_points[0][0] + 10, polygon_points[0][1] - 10)
        cv2.putText(frame, zone_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Reset for the next polygon
        polygon_points.clear()
        zone_counter += 1  # Increment to assign the next zone name

    # Blend the mask with the frame to highlight the masked region
    final_frame = cv2.addWeighted(temp_frame, 0.7, mask, 0.3, 0)

    # Show the frame
    cv2.imshow("Polygon Mask", final_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("m"):
        paused = not paused  # Pause/unpause video
        masking = paused  # Start masking when paused

    elif key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()

# Example: Retrieve the mask with zone names after closing the program
print("Zone Data:", zone_data)
