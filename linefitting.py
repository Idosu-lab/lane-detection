import cv2
import numpy as np
from collections import deque

# Global deque for smoothing
left_fit_history = deque(maxlen=10)
right_fit_history = deque(maxlen=10)

def smooth_fit(fit_history, new_fit):
    if new_fit is not None and len(new_fit) in [2, 3]:
        fit_history.append(new_fit)
    if len(fit_history) > 0:
        return np.mean(np.array(fit_history), axis=0)
    return None

def apply_color_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([255, 50, 255], dtype=np.uint8)
    yellow_lower = np.array([15, 90, 100], dtype=np.uint8)
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
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None
    try:
        poly = np.polyfit(y_coords, x_coords, 1)
        return poly
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def draw_lane_lines(frame, left_fit, right_fit):
    height, width = frame.shape[:2]
    line_image = np.zeros_like(frame)
    if left_fit is not None:
        y1 = height
        y2 = int(height * 0.6)
        x1 = int(np.polyval(left_fit, y1))
        x2 = int(np.polyval(left_fit, y2))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    if right_fit is not None:
        y1 = height
        y2 = int(height * 0.6)
        x1 = int(np.polyval(right_fit, y1))
        x2 = int(np.polyval(right_fit, y2))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

def detect_crosswalk(frame):
    """Detect crosswalk patterns, focusing on the lower part of the frame."""
    height, _ = frame.shape[:2]
    bottom_part = frame[int(height * 0.6):, :]

    hsv = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([255, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, white_lower, white_upper)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crosswalk_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Threshold to filter noise
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 5:  # Crosswalk lines are typically rectangular
                cv2.rectangle(bottom_part, (x, y), (x + w, y + h), (0, 0, 255), 2)
                crosswalk_detected = True

    if crosswalk_detected:
        cv2.putText(frame, "Crosswalk Detected", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def process_frame(frame):
    """Process each frame to visualize lanes and detect crosswalks."""
    enhanced_frame = enhance_contrast(frame)
    filtered_frame = apply_color_threshold(enhanced_frame)
    roi_frame = apply_roi_mask(filtered_frame)
    blurred_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)

    edges = detect_edges(blurred_frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=100)

    frame_width = frame.shape[1]
    left_lines, right_lines = separate_lines_by_slope_and_position(lines, frame_width)

    left_fit = fit_line_to_points(left_lines)
    right_fit = fit_line_to_points(right_lines)

    lane_frame = draw_lane_lines(frame, left_fit, right_fit)
    final_frame = detect_crosswalk(lane_frame)

    return final_frame

def play_video_with_lane_and_crosswalk(video_path):
    """Play video with both lane and crosswalk detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Lane and Crosswalk Detection", processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Play the video with combined lane and crosswalk detection
play_video_with_lane_and_crosswalk("cropped_output.avi")
