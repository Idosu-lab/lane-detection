import cv2
import numpy as np

"""
~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETER DOCUMENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~

MIN_ROI_WHITE_COVERAGE (float):
    - The minimum fraction (0.0 to 1.0) of white pixels in the bottom half of
      the frame required to even consider detecting a crosswalk.
    - If the bottom half has less white than this, we skip detection.

MAX_ROI_WHITE_COVERAGE (float):
    - The maximum fraction (0.0 to 1.0) of white pixels in the bottom half.
    - If the bottom half has more white than this, we skip detection.

MIN_TRANSITIONS_PER_ROW (int):
    - The minimum number of white↔black transitions in a single row
      for that row to be considered “stripy.”

CONSECUTIVE_NEEDED (int):
    - The number of consecutive “stripy” rows required to form a candidate
      crosswalk region.

WHITE_COVERAGE_RATIO (float):
    - Within the candidate bounding box, the fraction of white pixels
      required to confirm it's truly a white-striped region.

MIN_BOX_HEIGHT (int):
    - The minimum vertical height (in pixels) of the candidate region.

MIN_COLUMN_STRIPES (int):
    - Minimum columns that must be “stripy” in the bounding box.

MIN_TRANSITIONS_PER_COLUMN (int):
    - Transitions needed in each stripy column.

MAX_COLUMN_STRIPES (int):
    - Maximum columns allowed to be stripy (beyond this, discard the region).

-------------------------------------------------------------------------------
"""

MIN_ROI_WHITE_COVERAGE = 0.04
MAX_ROI_WHITE_COVERAGE = 0.1

MIN_TRANSITIONS_PER_ROW = 15
CONSECUTIVE_NEEDED = 40
WHITE_COVERAGE_RATIO = 0.2
MIN_BOX_HEIGHT = 60

MIN_COLUMN_STRIPES = 15
MIN_TRANSITIONS_PER_COLUMN = 5
MAX_COLUMN_STRIPES = 900

def apply_bottom_half_roi(frame):
    """
    Keep only the bottom half of the frame.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    roi_vertices = np.array([[
        (0, h),
        (0, h // 2),
        (w, h // 2),
        (w, h)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def white_mask(frame_bgr):
    """
    Convert to HSV, threshold for 'white' color, then morph open/close.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def count_row_transitions(row):
    """
    Count white<->black transitions in a 1D array of [0,255].
    """
    row_bool = (row > 127).astype(np.uint8)
    diff = np.abs(np.diff(row_bool))
    transitions = diff.sum()
    return transitions

def count_column_transitions(col):
    """
    Similar logic for a vertical slice (column).
    """
    col_bool = (col > 127).astype(np.uint8)
    diff = np.abs(np.diff(col_bool))
    transitions = diff.sum()
    return transitions

def validate_strip_region(roi_mask, row_start, row_end, row_transition_counts):
    """
    1) Height check
    2) White coverage check
    3) Column-based repeated stripe check

    Returns (valid, region_height, coverage, stripy_columns).
    If invalid, 'valid' is False and the other returns might be partial data.
    """
    region_height = row_end - row_start + 1
    if region_height < MIN_BOX_HEIGHT:
        return (False, region_height, None, None)

    h, w = roi_mask.shape[:2]
    region = roi_mask[row_start:row_end+1, 0:w]

    total_pixels = region.size
    white_pixels = cv2.countNonZero(region)
    coverage = float(white_pixels) / (total_pixels + 1e-6)

    if coverage < WHITE_COVERAGE_RATIO:
        return (False, region_height, coverage, None)

    stripy_columns = 0
    for col_idx in range(w):
        transitions = count_column_transitions(region[:, col_idx])
        if transitions >= MIN_TRANSITIONS_PER_COLUMN:
            stripy_columns += 1

    # Must be >= MIN_COLUMN_STRIPES and <= MAX_COLUMN_STRIPES
    if stripy_columns < MIN_COLUMN_STRIPES or stripy_columns > MAX_COLUMN_STRIPES:
        return (False, region_height, coverage, stripy_columns)

    return (True, region_height, coverage, stripy_columns)

def detect_crosswalk_stripes(frame):
    """
    Returns:
      - output_frame (with bounding box if found)
      - found (bool): Did we find a crosswalk?
      - mask (white mask)
      - stats (dict) containing all relevant measurements from *this frame*
        so we can print them on-demand (press 'p').
    """
    # ROI + White Mask
    roi = apply_bottom_half_roi(frame)
    mask = white_mask(roi)

    # Coverage in the ROI
    total_pixels = mask.size
    white_pixels = cv2.countNonZero(mask)
    roi_coverage = float(white_pixels) / (total_pixels + 1e-6)

    h, w = mask.shape[:2]
    row_transition_counts = np.zeros(h, dtype=np.int32)

    for row_idx in range(h):
        row_transition_counts[row_idx] = count_row_transitions(mask[row_idx, :])

    crosswalk_found = False
    start_row = -1
    end_row = -1

    final_length_of_run = 0
    region_height = None
    region_coverage = None
    stripy_columns = None

    # ROW check
    if MIN_ROI_WHITE_COVERAGE <= roi_coverage <= MAX_ROI_WHITE_COVERAGE:
        # Only do row-based if coverage is in range
        for i in range(h-1, -1, -1):
            if row_transition_counts[i] >= MIN_TRANSITIONS_PER_ROW:
                if end_row == -1:
                    end_row = i
                    start_row = i
                else:
                    start_row = i
            else:
                if end_row != -1:
                    length_of_run = end_row - start_row + 1
                    if length_of_run >= CONSECUTIVE_NEEDED:
                        valid, r_height, r_coverage, s_cols = validate_strip_region(
                            mask, start_row, end_row, row_transition_counts
                        )
                        if valid:
                            crosswalk_found = True
                            region_height = r_height
                            region_coverage = r_coverage
                            stripy_columns = s_cols
                            final_length_of_run = length_of_run
                            break
                start_row = -1
                end_row = -1

        # Check final run if not found
        if not crosswalk_found and end_row != -1:
            length_of_run = end_row - start_row + 1
            if length_of_run >= CONSECUTIVE_NEEDED:
                valid, r_height, r_coverage, s_cols = validate_strip_region(
                    mask, start_row, end_row, row_transition_counts
                )
                if valid:
                    crosswalk_found = True
                    region_height = r_height
                    region_coverage = r_coverage
                    stripy_columns = s_cols
                    final_length_of_run = length_of_run

    # Draw bounding box if found
    output = frame.copy()
    if crosswalk_found and start_row >= 0 and end_row >= 0:
        y_offset = frame.shape[0] - h
        box_top = start_row + y_offset
        box_bottom = end_row + y_offset

        # Highlight the rectangle with a transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (0, box_top), (w-1, box_bottom), (255, 0, 0), -1)  # Filled rectangle (yellow)
        alpha = 0.4  # Transparency factor
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        # Draw the bounding box border on top of the overlay
        cv2.rectangle(output, (0, box_top), (w-1, box_bottom), (0, 0, 255), 3)

    # ~~~~ STATS FOR THIS FRAME ~~~~
    stats = {
        "found": crosswalk_found,
        "roi_coverage": roi_coverage,
        "min_row_t": None,
        "max_row_t": None,
        "avg_row_t": None,
        "length_of_run": final_length_of_run,
        "region_coverage": region_coverage,
        "region_height": region_height,
        "stripy_columns": stripy_columns
    }

    # If we identified a region, let's fill in row transitions stats
    if start_row >= 0 and end_row >= 0:
        region_rows = row_transition_counts[start_row:end_row+1]
        if region_rows.size > 0:
            stats["min_row_t"] = np.min(region_rows)
            stats["max_row_t"] = np.max(region_rows)
            stats["avg_row_t"] = float(np.mean(region_rows))

    return output, crosswalk_found, mask, stats

def print_crosswalk_stats(stats):
    """
    Print the same info as if we just detected a crosswalk,
    using the stats dictionary from detect_crosswalk_stripes.
    """
    found = stats["found"]

    print("\n===== FRAME STATS =====")
    if found:
        print("[CROSSWALK DETECTED] Actual Values:")
    else:
        print("[NO CROSSWALK DETECTED] Current Frame Values:")

    print(f" - ROI coverage         = {stats['roi_coverage']:.4f} "
          f"(must be between {MIN_ROI_WHITE_COVERAGE} and {MAX_ROI_WHITE_COVERAGE})")

    # If we don't have region stats, print placeholders
    if stats["min_row_t"] is None:
        print(" - Row transitions in region: no region found yet.")
    else:
        min_row_t = stats["min_row_t"]
        max_row_t = stats["max_row_t"]
        avg_row_t = stats["avg_row_t"]
        print(f" - Row transitions in region: min={min_row_t}, max={max_row_t}, avg={avg_row_t:.2f} "
              f"(each row >= {MIN_TRANSITIONS_PER_ROW} to be stripy)")

    length_of_run = stats["length_of_run"]
    if length_of_run == 0:
        print(f" - Length of run        = 0 (no consecutive stripy run found; needed >= {CONSECUTIVE_NEEDED})")
    else:
        print(f" - Length of run        = {length_of_run} (needed >= {CONSECUTIVE_NEEDED})")

    region_coverage = stats["region_coverage"]
    region_height = stats["region_height"]
    stripy_columns = stats["stripy_columns"]

    if region_coverage is None:
        print(f" - BBox coverage        = None (region not validated)")
    else:
        print(f" - BBox coverage        = {region_coverage:.4f} (needed >= {WHITE_COVERAGE_RATIO})")

    if region_height is None:
        print(f" - BBox height          = None (region not validated; needed >= {MIN_BOX_HEIGHT} px)")
    else:
        print(f" - BBox height          = {region_height} px (needed >= {MIN_BOX_HEIGHT} px)")

    if stripy_columns is None:
        print(f" - Stripy columns       = None (region not validated; needed >= {MIN_COLUMN_STRIPES} and <= {MAX_COLUMN_STRIPES})")
    else:
        print(f" - Stripy columns       = {stripy_columns} (needed >= {MIN_COLUMN_STRIPES} && <= {MAX_COLUMN_STRIPES}; col >= {MIN_TRANSITIONS_PER_COLUMN} transitions)")

    print("--------------------------------------------------------\n")

def play_crosswalk_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if needed
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Now detect, capturing the stats
        result_frame, found, debug_mask, stats = detect_crosswalk_stripes(frame)

        status = "Crosswalk Detected" if found else "No Crosswalk"
        color = (0,255,0) if found else (0,0,255)
        cv2.putText(result_frame, status, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        display_result = cv2.resize(result_frame, (800, 500)) 
        display_mask = cv2.resize(debug_mask, (800, 500))

        cv2.imshow("Crosswalk Detection", display_result)
        cv2.imshow("White Mask", display_mask)

        # Print stats on-demand or if we found a crosswalk
        key = cv2.waitKey(30) & 0xFF
        if key == ord('p') or found:
            print_crosswalk_stats(stats)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "miamifirstbasic.avi"
    play_crosswalk_detection(video_path)