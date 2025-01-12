import cv2
import numpy as np

# Load video file
video_path = "cropped_output.avi"
cap = cv2.VideoCapture(video_path)

# Preprocessing helper functions

def preprocess_frame(frame):
    # Step 1: Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_frame = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

    return enhanced_frame

# Edge detection and Hough Transform

def detect_crosswalk_edges(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply morphological operations to refine edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Detect edges using Canny
    edges = cv2.Canny(morphed, 50, 150)  # Adjusted thresholds

    return edges

def detect_rectangles(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to isolate white regions (potential crosswalks)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices and is large enough
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 50 and h > 20:  # Minimum size threshold
                rectangles.append((x, y, w, h))

    return rectangles

def group_rectangles_and_detect_crosswalk(rectangles, frame):
    crosswalk_detected = False

    # Group rectangles based on proximity
    grouped_rectangles = []
    for rect in rectangles:
        x, y, w, h = rect
        added = False
        for group in grouped_rectangles:
            for gx, gy, gw, gh in group:
                if abs(x - gx) < 50 and abs(y - gy) < 50:  # Group rectangles within proximity
                    group.append(rect)
                    added = True
                    break
            if added:
                break
        if not added:
            grouped_rectangles.append([rect])

    # Highlight groups with enough rectangles
    for group in grouped_rectangles:
        if len(group) >= 3:  # Crosswalks typically have multiple parallel rectangles
            crosswalk_detected = True
            # Calculate the bounding box of the grouped rectangles
            x_coords = [rect[0] for rect in group] + [rect[0] + rect[2] for rect in group]
            y_coords = [rect[1] for rect in group] + [rect[1] + rect[3] for rect in group]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw bounding box in red
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    if crosswalk_detected:
        cv2.putText(frame, "Crosswalk Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Step 2: Detect rectangles (potential crosswalk lines)
    rectangles = detect_rectangles(processed_frame)

    # Step 3: Group rectangles and detect crosswalk
    final_frame = group_rectangles_and_detect_crosswalk(rectangles, frame)

    # Display final result
    cv2.imshow("Crosswalk Detection", final_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
