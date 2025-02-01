import cv2
import numpy as np
import random  # Import random for random color generation

# Global variables
polygon_zones = []  # Stores all drawn polygon zones
current_polygon = []  # Stores the current polygon points
masking = False  # Flag to check if masking is active
paused = False  # Flag to pause video when 'M' is pressed
mask = None  # Mask image
zone_counter = 0  # Zone counter to label each zone uniquely

# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global current_polygon, masking

    if masking:  # Only allow drawing when masking is active
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add points
            current_polygon.append((x, y))

# Function to return all polygon masks as a NumPy array with additional information
def get_masks():
    return [
        {"id": zone["id"], "points": np.array(zone["points"], dtype=np.int32), "color": zone["color"]}
        for zone in polygon_zones
    ]

# Load video and resize to 1920x1080
video_path = r"E:\RoboCam\RoboCam\assets\dataset\video-dataset\market-square.mp4"
cap = cv2.VideoCapture(video_path)

# Create a window
cv2.namedWindow("Polygon Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Polygon Mask", 1920, 1080)
cv2.setMouseCallback("Polygon Mask", draw_polygon)

# Placeholder for mask
mask = None

# Function to generate a random color excluding white
def random_color():
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))  # Generate a random RGB color
        if color != (255, 255, 255):  # Exclude white color
            return color

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

    # Draw the current polygon outline while masking is active
    if masking and len(current_polygon) > 1:
        cv2.polylines(temp_frame, [np.array(current_polygon, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    # When 'M' is pressed again, complete the polygon and mark it as a zone
    if not masking and len(current_polygon) > 2:
        current_polygon.append(current_polygon[0])  # Auto-connect first and last points

        # Assign a random color for the zone
        color = random_color()

        # Add the zone to the list with a unique ID
        polygon_zones.append({
            "id": f"Zone {chr(65 + zone_counter)}",  # Convert 0 -> A, 1 -> B, etc.
            "points": current_polygon.copy(),
            "color": color
        })
        zone_counter += 1  # Increment global `zone_counter`

        # Draw the zone on the mask with transparency
        cv2.fillPoly(mask, [np.array(current_polygon, np.int32)], color)
        cv2.polylines(mask, [np.array(current_polygon, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

        current_polygon.clear()  # Clear points after marking

    # Draw all zones with their respective colors
    for zone in polygon_zones:
        cv2.fillPoly(temp_frame, [np.array(zone["points"], np.int32)], zone["color"])
        cv2.polylines(temp_frame, [np.array(zone["points"], np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
        text_position = tuple(zone["points"][0])  # Place label at first point
        cv2.putText(temp_frame, zone["id"], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Blend the mask with the frame to highlight the foreground with transparency (mask + alpha)
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

# Example: Retrieve and print all polygon masks after closing
polygon_masks = get_masks()
for zone in polygon_masks:
    print(f"Zone ID: {zone['id']}, Color: {zone['color']}, Coordinates: {zone['points']}")
