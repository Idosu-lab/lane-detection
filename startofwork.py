import cv2

# Load video file
video_path = "miamifirstbasic.avi"
output_path = "cropped_output.avi"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define cropping parameters (adjust based on screenshots)
crop_top = int(0.2 * frame_height)  # Crop upper 40% of the frame

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height - crop_top))

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame
    cropped_frame = frame[crop_top:, :]

    # Write cropped frame to output
    out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Cropped video saved to {output_path}")
import cv2
import numpy as np

# Step 1: Color Thresholding
def apply_color_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 180], dtype=np.uint8)
    white_upper = np.array([255, 80, 255], dtype=np.uint8)
    yellow_lower = np.array([15, 70, 100], dtype=np.uint8)
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

# Step 2: Contrast Enhancement
def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Step 3: Region of Interest Masking
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

# Step 4: Line Detection
def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=50)
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

# Step 5: Frame Processing
def process_frame(frame):
    enhanced_frame = enhance_contrast(frame)
    filtered_frame = apply_color_threshold(enhanced_frame)
    roi_frame = apply_roi_mask(filtered_frame)
    blurred_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)
    return detect_lines(blurred_frame)

# Step 6: Process Video
def play_processed_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Processed Video", processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Play the processed video
play_processed_video("cropped_output.avi")
