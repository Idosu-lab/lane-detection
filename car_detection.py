import cv2
import numpy as np

########################################
#           GLOBAL PARAMETERS          #
########################################

# --- Detection Thresholds ---
MIN_CONTOUR_AREA  = 1500
MAX_CONTOUR_AREA  = 200000

# Aspect ratio (width/height) constraints
MIN_ASPECT_RATIO  = 0.5
MAX_ASPECT_RATIO  = 4.0

# Extent = (contour area) / (bounding box area)
MIN_EXTENT        = 0.6
MAX_EXTENT        = 0.95

# Solidity = (contour area) / (convex hull area)
MIN_SOLIDITY      = 0.5

# For optional color-based skipping (to remove bright green regions):
USE_COLOR_CHECK   = False  # set True if you want to skip obviously green shapes
GREEN_HUE_LOWER   = 35     # typical green hue range in HSV
GREEN_HUE_UPPER   = 85

# Approx. real-world height of the car portion used for distance calculation
REAL_WORLD_VEHICLE_HEIGHT = 1.5  # meters
FOCAL_LENGTH              = 700.0

# Distance thresholds (in meters)
DANGER_DISTANCE  = 8.0
WARNING_DISTANCE = 15.0

# How much of the bottom frame to keep. E.g., 0.6 => only bottom 60%
ROI_BOTTOM_FRACTION = 0.4

# Morphological kernel size for opening/closing (larger merges more blobs)
MORPH_KERNEL_SIZE = (9, 9)
MORPH_ITERS       = 2

# Centroid matching threshold (pixels). If new detection centroid is within
# this distance of a tracked object’s centroid, we consider them the same object.
MAX_MATCH_DISTANCE = 80

# Multi-frame tracking logic
# - Must see the object for at least CONFIRMATION_FRAMES consecutively to confirm
# - Remove if it disappears for DISAPPEAR_FRAMES
CONFIRMATION_FRAMES = 4
DISAPPEAR_FRAMES    = 10

########################################
#   Background Subtractor (MOG2)       #
########################################
bgs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

########################################
#           HELPER FUNCTIONS           #
########################################

def apply_bottom_roi(frame, fraction):
    """
    Keep only the bottom 'fraction' of the frame. Mask out the rest.
    """
    h, w = frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int((1 - fraction) * h)

    roi_vertices = np.array([[
        (0, h),
        (0, y_start),
        (w, y_start),
        (w, h)
    ]], dtype=np.int32)

    cv2.fillPoly(roi_mask, roi_vertices, 255)
    return cv2.bitwise_and(frame, frame, mask=roi_mask)

def estimate_distance(bbox_height_px):
    """
    Simple pinhole camera model to estimate distance in meters.
    """
    if bbox_height_px <= 0:
        return 9999.0
    return (FOCAL_LENGTH * REAL_WORLD_VEHICLE_HEIGHT) / float(bbox_height_px)

def get_centroid(box):
    """
    For a bounding box [x, y, w, h], return (cX, cY).
    """
    x, y, w, h = box
    cX = x + w // 2
    cY = y + h // 2
    return (cX, cY)

def detect_moving_boxes(frame):
    """
    Detect moving objects in 'frame' using background subtraction
    + morphological ops + shape filtering.

    Returns:
      - boxes:      list of [x, y, w, h] for final passing cars
      - fgmask:     foreground mask (for debugging display)
      - shape_info: list of dicts with shape/size metrics for each passing car,
                    in the same order as 'boxes'.
                    Example of each dict:
                      {
                         "contour_area": ...,
                         "bbox_area": ...,
                         "aspect_ratio": ...,
                         "extent": ...,
                         "solidity": ...
                      }
    """
    # 1) Optionally keep only bottom part of frame
    roi_frame = apply_bottom_roi(frame, ROI_BOTTOM_FRACTION)

    # 2) Convert to grayscale & subtract background
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    fgmask = bgs.apply(gray)

    # 3) Morphological operations (opening/closing) to merge partial blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERS)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERS)

    # 4) Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Because we used ROI, we need to offset bounding boxes by how much we cut from the top
    h_roi, w_roi = roi_frame.shape[:2]
    y_offset = frame.shape[0] - h_roi

    boxes = []
    shape_info = []  # store shape stats for each passing detection

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area < MIN_CONTOUR_AREA or contour_area > MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        y_global = y + y_offset

        # (A) Aspect Ratio
        aspect_ratio = (w / float(h)) if h > 0 else 9999
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue

        # (B) Extent = area / bbox_area
        bbox_area = w * h
        extent = float(contour_area) / float(bbox_area + 1e-6)
        if not (MIN_EXTENT <= extent <= MAX_EXTENT):
            continue

        # (C) Solidity = contour_area / convexHull_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(contour_area) / float(hull_area + 1e-6)
        if solidity < MIN_SOLIDITY:
            continue

        # (D) Optional color check for bright green shapes
        if USE_COLOR_CHECK:
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            y1 = max(y, 0)
            y2 = min(y + h, hsv.shape[0])
            x1 = max(x, 0)
            x2 = min(x + w, hsv.shape[1])
            sub_hsv = hsv[y1:y2, x1:x2]
            mean_hue = np.mean(sub_hsv[..., 0])  # average hue in this region
            if GREEN_HUE_LOWER <= mean_hue <= GREEN_HUE_UPPER:
                # Possibly a green foliage region
                continue

        boxes.append([x, y_global, w, h])
        shape_info.append({
            "contour_area": contour_area,
            "bbox_area":    bbox_area,
            "aspect_ratio": aspect_ratio,
            "extent":       extent,
            "solidity":     solidity
        })

    return boxes, fgmask, shape_info

########################################
#         CENTROID-BASED TRACKER       #
########################################

class TrackedObject:
    """
    Represents one tracked object/vehicle using a bounding box and centroid.
    """
    def __init__(self, box, object_id, frame_num):
        self.id = object_id
        self.box = box              # [x, y, w, h]
        self.centroid = get_centroid(box)
        self.confirmed = False
        self.frames_visible = 1     # total frames object has been seen
        self.frames_consecutive = 1 # consecutive frames object has been seen
        self.last_frame = frame_num

        if self.frames_consecutive >= CONFIRMATION_FRAMES:
            self.confirmed = True

    def update(self, box, frame_num):
        """
        Update bounding box & centroid, increment counters.
        """
        self.box = box
        self.centroid = get_centroid(box)
        self.frames_visible += 1
        if frame_num == self.last_frame + 1:
            # consecutive
            self.frames_consecutive += 1
        else:
            # gap => reset consecutive count
            self.frames_consecutive = 1
        self.last_frame = frame_num

        if self.frames_consecutive >= CONFIRMATION_FRAMES:
            self.confirmed = True

    def time_since_update(self, current_frame):
        """
        How many frames since we last saw this object?
        """
        return current_frame - self.last_frame


class CentroidTracker:
    """
    Matches new detections to existing tracked objects by centroid distance.
    If no match is found within a threshold, we create a new object.
    Remove objects not seen for DISAPPEAR_FRAMES frames.
    """
    def __init__(self):
        self.objects = {}   # object_id -> TrackedObject
        self.next_id = 1

    def update_objects(self, detected_boxes, frame_num):
        """
        Match each detected box to the best existing object within a distance threshold.
        """
        if not self.objects:
            # No existing objects => each box is a new object
            for b in detected_boxes:
                self.objects[self.next_id] = TrackedObject(b, self.next_id, frame_num)
                self.next_id += 1
            return

        new_centroids = [get_centroid(b) for b in detected_boxes]
        used_detections = set()

        # Try to match each existing object to a detection
        obj_ids = list(self.objects.keys())
        for oid in obj_ids:
            obj = self.objects[oid]
            best_match_idx = -1
            min_dist = float('inf')

            for i, c in enumerate(new_centroids):
                if i in used_detections:
                    continue
                dist = np.hypot(c[0] - obj.centroid[0], c[1] - obj.centroid[1])
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx >= 0 and min_dist <= MAX_MATCH_DISTANCE:
                matched_box = detected_boxes[best_match_idx]
                obj.update(matched_box, frame_num)
                used_detections.add(best_match_idx)

        # For any unmatched boxes, create a new object
        for i, b in enumerate(detected_boxes):
            if i not in used_detections:
                self.objects[self.next_id] = TrackedObject(b, self.next_id, frame_num)
                self.next_id += 1

    def remove_stale_objects(self, frame_num):
        """
        Remove objects that haven't been seen for DISAPPEAR_FRAMES frames.
        """
        to_remove = []
        for oid, obj in self.objects.items():
            if obj.time_since_update(frame_num) > DISAPPEAR_FRAMES:
                to_remove.append(oid)
        for oid in to_remove:
            del self.objects[oid]


########################################
#     PRINTING SHAPE STATS ON DEMAND   #
########################################

def print_car_stats(shape_info_list):
    """
    Print shape/size metrics for each detected (and confirmed) car in this frame.
    shape_info_list: a list of dicts with the keys:
       'contour_area', 'bbox_area', 'aspect_ratio', 'extent', 'solidity'
    """
    if not shape_info_list:
        print("[No cars detected this frame or no data to show.]")
        return

    print("\n===== Car Detection Stats (Press 'p') =====")
    for i, info in enumerate(shape_info_list):
        print(f"Car {i+1}:")
        print(f" - Contour area   = {info['contour_area']:.1f}  "
              f"(Min {MIN_CONTOUR_AREA}, Max {MAX_CONTOUR_AREA})")
        print(f" - BBox area      = {info['bbox_area']:.1f}")
        print(f" - Aspect ratio   = {info['aspect_ratio']:.2f}  "
              f"(Min {MIN_ASPECT_RATIO}, Max {MAX_ASPECT_RATIO})")
        print(f" - Extent         = {info['extent']:.2f}    "
              f"(Min {MIN_EXTENT}, Max {MAX_EXTENT})")
        print(f" - Solidity       = {info['solidity']:.2f}   "
              f"(Min {MIN_SOLIDITY})")
        if USE_COLOR_CHECK:
            print(" - Color check    = [Skipping bright green shapes]")
    print("===========================================\n")


########################################
#         MAIN PLAY FUNCTION           #
########################################

def play_proximity_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    tracker = CentroidTracker()
    frame_num = 0

    # We'll store the shape_info from detect_moving_boxes() each frame.
    # This helps us print shape stats on demand (press 'p').
    current_shape_info = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Optionally resize if too large
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # 1) Detect bounding boxes + shape info
        boxes, fgmask, shape_info_list = detect_moving_boxes(frame)
        current_shape_info = shape_info_list  # store for potential printing

        # 2) Update tracker with new detections
        tracker.update_objects(boxes, frame_num)

        # 3) Remove stale objects
        tracker.remove_stale_objects(frame_num)

        # 4) Draw confirmed objects with color-coded distance
        confirmed_count = 0
        distances = []

        for oid, obj in tracker.objects.items():
            if not obj.confirmed:
                # not yet confirmed => skip drawing
                continue
            confirmed_count += 1

            x, y, w, h = obj.box
            dist = estimate_distance(h)
            distances.append(dist)

            if dist < DANGER_DISTANCE:
                color = (0, 0, 255)    # red
            elif dist < WARNING_DISTANCE:
                color = (0, 255, 255) # yellow
            else:
                color = (0, 255, 0)   # green

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{oid} {dist:.1f}m",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count how many are in "danger" range
        num_danger = sum(1 for d in distances if d < DANGER_DISTANCE)

        # 5) Display overlays
        status_text = f"Frame:{frame_num}  Cars:{confirmed_count}  Danger:{num_danger}"
        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

        display_frame = cv2.resize(frame, (900, 550))
        debug_mask    = cv2.resize(fgmask, (450, 275))

        cv2.imshow("Vehicle Detection + Centroid Tracking", display_frame)
        cv2.imshow("Foreground Mask", debug_mask)

        # Press 'p' to print shape info or 'q' to quit
        key = cv2.waitKey(30) & 0xFF
        if key == ord('p'):
            print_car_stats(current_shape_info)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

########################################
#   SAMPLE MAIN ENTRY POINT            #
########################################

if __name__ == "__main__":
    video_path = "miamifirstbasic2.avi"
    play_proximity_detection(video_path)
