import cv2
import numpy as np

def hist(img):
    """Compute a histogram of the binary image's bottom half to locate lane lines."""
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """Class to detect and process lane lines in video frames."""
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        """Detect lane lines in a binary warped image."""
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """Find all nonzero pixels within a specific sliding window."""
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        """Extract key features from the binary image for lane detection."""
        self.img = img
        self.window_height = int(img.shape[0] // self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find pixels belonging to lane lines."""
        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = int(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = int(np.mean(good_right_x))

        return leftx, lefty, rightx, righty

    def fit_poly(self, img):
        """Fit polynomials to detected lane line pixels and draw lane area."""
        leftx, lefty, rightx, righty = self.find_lane_pixels(img)

        # Fit polynomials only if sufficient points are detected
        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        if self.left_fit is None or self.right_fit is None:
            # Return the original image if no lanes are detected
            return np.dstack((img, img, img))

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        ploty = np.linspace(miny, maxy, img.shape[0])

        # Initialize lane area and fallback for undetected lanes
        lane_area = np.zeros_like(np.dstack((img, img, img)))
        left_fitx, right_fitx = None, None

        try:
            left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        except:
            print("Skipping frame: Lanes not detected.")

        if left_fitx is not None and right_fitx is not None:
            for i in range(len(ploty) - 1):
                pts_left = (int(left_fitx[i]), int(ploty[i]))
                pts_right = (int(right_fitx[i]), int(ploty[i]))
                pts_left_next = (int(left_fitx[i + 1]), int(ploty[i + 1]))
                pts_right_next = (int(right_fitx[i + 1]), int(ploty[i + 1]))
                cv2.fillPoly(lane_area, [np.array([pts_left, pts_left_next, pts_right_next, pts_right])], (0, 255, 0))

        # Ensure lane_area matches the original frame dimensions and channels
        if lane_area.shape[:2] != img.shape[:2]:
            lane_area = cv2.resize(lane_area, (img.shape[1], img.shape[0]))

        # Convert binary lane_area to 3 channels (if not already)
        if len(lane_area.shape) == 2 or lane_area.shape[2] == 1:
            lane_area = np.dstack((lane_area, lane_area, lane_area))

        return lane_area


def process_video(input_path, output_path):
    """Process a video file to detect and visualize lanes."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    lane_lines = LaneLines()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        lanes = lane_lines.forward(binary)
        output_frame = cv2.addWeighted(frame, 1, lanes, 0.5, 0)

        out.write(output_frame)
        cv2.imshow('Lane Detection', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and Output Paths
input_video = './miamifirstbasic.avi'
output_video = './output_lanes.avi'

process_video(input_video, output_video)
