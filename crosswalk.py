import cv2
import numpy as np

# Function to preprocess the image to detect yellow and white lanes
def preprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    white_mask = cv2.inRange(gblur, 200, 255)
    lower_yellow = np.array([18, 94, 140])  # Adjusted for better yellow detection
    upper_yellow = np.array([48, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return mask

# Updated fitCurve function with robust error handling
def fitCurve(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding window parameters
    nwindows = 10
    margin = 100
    minpix = 50
    window_height = img.shape[0] // nwindows

    # Identify lane pixels
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]

        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        if len(good_left_indices) > minpix:
            leftx_current = np.int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = np.int(np.mean(x[good_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices) if left_lane_indices else []
    right_lane_indices = np.concatenate(right_lane_indices) if right_lane_indices else []

    if len(left_lane_indices) == 0 or len(right_lane_indices) == 0:
        return None, None  # Return None if no lanes are detected

    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 and len(lefty) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 and len(righty) > 0 else None

    return left_fit, right_fit

# Main loop
video = cv2.VideoCapture("miamifirstbasic.avi")
out = cv2.VideoWriter('curve_lane_detection.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))
print("Generating video output...\n")

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break

    try:
        processed_img = preprocessing(frame)
        height, width = processed_img.shape
        polygon = [(int(width * 0.15), int(height * 0.94)),
                   (int(width * 0.45), int(height * 0.62)),
                   (int(width * 0.58), int(height * 0.62)),
                   (int(0.95 * width), int(0.94 * height))]
        
        source_points = np.float32([[int(width * 0.49), int(height * 0.62)],
                                    [int(width * 0.58), int(height * 0.62)],
                                    [int(width * 0.15), int(height * 0.94)],
                                    [int(0.95 * width), int(0.94 * height)]])
        
        destination_points = np.float32([[0, 0], [400, 0], [0, 960], [400, 960]])
        warped_img_size = (400, 960)

        warped_img = warp(processed_img, source_points, destination_points, warped_img_size)
        kernel = np.ones((11, 11), np.uint8)
        opening = cv2.morphologyEx(warped_img, cv2.MORPH_CLOSE, kernel)
        left_fit, right_fit = fitCurve(opening)

        if left_fit is None or right_fit is None:
            print("No lanes detected in this frame.")
            continue

        pts_left, pts_right = findPoints((960, 400), left_fit, right_fit)
        fill_curves = fillCurves((960, 400), pts_left, pts_right)
        unwarped_fill_curves = unwarp(fill_curves, source_points, destination_points, (width, height))
        result = cv2.addWeighted(frame, 1, unwarped_fill_curves, 1, 0)

        out.write(result)

    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

video.release()
out.release()
print("Video processing completed. Output saved as 'curve_lane_detection.avi'.")
