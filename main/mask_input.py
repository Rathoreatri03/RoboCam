import cv2
import numpy as np

# Variable Assigning
poly_zones = []  # all zone data store
current_poly_points = []  # store point of the zone
masking = False
paused = False
zone_counter = 0  # for zone A, B, C


# Mouse Function
def draw_polygon(event, x, y, flags, param):
    global current_poly_points, masking
    if masking:  # Only allow drawing when masking is active
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add points
            current_poly_points.append((x, y))


# Function to return the polygon mask as a NumPy array
def get_masks():
    return [
        {"id": zone["id"], "points": np.array(zone["points"], dtype=np.int32)}
        for zone in poly_zones
    ]


# Load video and resize to 1920x1080
video_path = r"E:\RoboCam\RoboCam\assets\dataset\video-dataset\market-square.mp4"
cap = cv2.VideoCapture(video_path)

# Get original dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window
cv2.namedWindow("Polygon Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Polygon Mask", 1920, 1080)
cv2.setMouseCallback("Polygon Mask", draw_polygon)

# Placeholder for mask
mask = None

while True:
    if not paused:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (1980, 1080))
        if mask is None:
            mask = np.zeros_like(frame, dtype=np.uint8)

    temp_frame = frame.copy()
    overlay = np.zeros_like(frame, dtype=np.uint8)  # Create transparent overlay

    if masking and len(current_poly_points) > 1:
        cv2.polylines(temp_frame, [np.array(current_poly_points, np.int32)],
                      isClosed=False, color=(0, 255, 0), thickness=2)

    if not masking and len(current_poly_points) > 2:
        current_poly_points.append(current_poly_points[0])  # Auto-connect first and last points

        # Add the zone to the list with a unique ID
        poly_zones.append({
            "id": f"Zone {chr(65 + zone_counter)}",  # Convert 0 -> A, 1 -> B, etc.
            "points": current_poly_points.copy(),
        })
        zone_counter += 1

        # Draw on the mask with transparency
        cv2.fillPoly(overlay, [np.array(current_poly_points, np.int32)], (0, 0, 255))
        cv2.polylines(overlay, [np.array(current_poly_points, np.int32)],
                      isClosed=True, color=(0, 255, 0), thickness=2)

        current_poly_points.clear()

    # Draw all zones with their respective colors on the overlay
    for zone in poly_zones:
        cv2.fillPoly(overlay, [np.array(zone["points"], np.int32)], (0, 0, 255))
        cv2.polylines(overlay, [np.array(zone["points"], np.int32)],
                      isClosed=True, color=(0, 255, 0), thickness=2)
        text_position = tuple(zone["points"][0])
        cv2.putText(overlay, zone["id"], text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Blend the overlay with the original frame
    alpha = 0.3  # Transparency factor
    final_frame = cv2.addWeighted(temp_frame, 1, overlay, alpha, 0)

    cv2.imshow("Polygon Mask", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):
        paused = not paused  # Pause/unpause video
        masking = paused  # Start masking when paused
    elif key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()

poly_mask = get_masks()
for zone in poly_mask:
    print(f"Zone ID: {zone['id']}, Coordinates: {zone['points']}")