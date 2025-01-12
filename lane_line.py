import cv2
import numpy as np
from collections import deque

# Global deque for smoothing
left_fit_history = deque(maxlen=10)
right_fit_history = deque(maxlen=10)

def smooth_fit(fit_history, new_fit):
    """Smooth the fit by averaging over the fit history."""
    if new_fit is not None and len(new_fit) in [2, 3]:
        fit_history.append(new_fit)
    if len(fit_history) > 0:
        return np.mean(np.array(fit_history), axis=0)
    return None

def apply_color_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)  # Adjusted threshold for white
    white_upper = np.array([255, 50, 255], dtype=np.uint8)
    yellow_lower = np.array([15, 90, 100], dtype=np.uint8)  # Adjusted threshold for yellow
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def apply_roi_mask(frame):
    height, width = frame.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (width // 2 - 100, int(height * 0.6)),
        (width // 2 + 100, int(height * 0.6)),
        (width, height)
    ]], dtype=np.int32)
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, roi_vertices, (255,) * frame.shape[2])
    return cv2.bitwise_and(frame, mask)

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    return edges

def separate_lines_by_slope_and_position(lines, frame_width):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if 0.5 < abs(slope) < 2:  # Ignore steep or flat lines
                if slope < 0 and x1 < frame_width / 2 and x2 < frame_width / 2:
                    left_lines.append(line)
                elif slope > 0 and x1 > frame_width / 2 and x2 > frame_width / 2:
                    right_lines.append(line)
    return left_lines, right_lines

def fit_line_to_points(lines):
    if len(lines) == 0:
        return None
    x_coords = []
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    if len(x_coords) < 2 or len(y_coords) < 2:  # Ensure sufficient points for fitting
        return None
    try:
        poly = np.polyfit(y_coords, x_coords, 1)  # Linear fit for stability
        return poly
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def calculate_direction(left_fit, right_fit):
    """Determine the direction based on lane curvature."""
    if left_fit is not None and right_fit is not None:
        # Compare the slopes at the bottom of the image to determine direction
        left_slope = left_fit[0]
        right_slope = right_fit[0]
        if left_slope > 0 and right_slope > 0:
            return "Right"
        elif left_slope < 0 and right_slope < 0:
            return "Left"
        else:
            return "Straight"
    return "Unknown"

def annotate_lane_gap(frame, left_fit, right_fit):
    height, width = frame.shape[:2]
    annotated_frame = np.copy(frame)

    if left_fit is not None and right_fit is not None:
        y_bottom = height
        y_top = int(height * 0.6)
        left_x_bottom = int(np.polyval(left_fit, y_bottom))
        left_x_top = int(np.polyval(left_fit, y_top))
        right_x_bottom = int(np.polyval(right_fit, y_bottom))
        right_x_top = int(np.polyval(right_fit, y_top))

        points = np.array([
            [left_x_bottom, y_bottom],
            [left_x_top, y_top],
            [right_x_top, y_top],
            [right_x_bottom, y_bottom]
        ], dtype=np.int32)

        overlay = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.5, 0)

    # Add direction annotation
    direction = calculate_direction(left_fit, right_fit)
    cv2.putText(annotated_frame, f"Direction: {direction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated_frame

def process_frame_for_gap_visualization(frame):
    """Process each frame to visualize lanes with stabilized fits."""
    enhanced_frame = enhance_contrast(frame)
    filtered_frame = apply_color_threshold(enhanced_frame)
    roi_frame = apply_roi_mask(filtered_frame)
    blurred_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)

    edges = detect_edges(blurred_frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=100)

    frame_width = frame.shape[1]
    left_lines, right_lines = separate_lines_by_slope_and_position(lines, frame_width)

    left_fit_raw = fit_line_to_points(left_lines)
    right_fit_raw = fit_line_to_points(right_lines)

    # Smooth the fits
    left_fit = smooth_fit(left_fit_history, left_fit_raw)
    right_fit = smooth_fit(right_fit_history, right_fit_raw)

    return annotate_lane_gap(frame, left_fit, right_fit)

def play_video_with_lane_gap(video_path):
    """Play video with stabilized lane detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame_for_gap_visualization(frame)
        cv2.imshow("Lane Gap Visualization", processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Play the video with gap visualization
play_video_with_lane_gap("cropped_output.avi")
