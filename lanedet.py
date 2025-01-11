import numpy as np
import cv2

def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    rows, cols = image.shape[:2]
    vertices = np.array([[
        [cols * 0.1, rows * 0.95],
        [cols * 0.4, rows * 0.6],
        [cols * 0.6, rows * 0.6],
        [cols * 0.9, rows * 0.95]
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

def hough_transform(image):
    return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=300)

def average_slope_intercept(lines):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    return (int((y1 - intercept) / slope), int(y1)), (int((y2 - intercept) / slope), int(y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    return [pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)]

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)
    if hough is not None:
        lanes = lane_lines(frame, hough)
        return draw_lane_lines(frame, lanes)
    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = frame_processor(frame)
        out.write(processed_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Play the output video
    cap_output = cv2.VideoCapture(output_path)
    while cap_output.isOpened():
        ret, frame = cap_output.read()
        if not ret:
            break
        cv2.imshow('Output Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    cap_output.release()
    cv2.destroyAllWindows()

process_video('miamifirstbasic.avi', 'output.avi')
