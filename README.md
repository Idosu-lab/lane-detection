# 🚗 ADAS Computer Vision Pipeline: Lane, Vehicle & Crosswalk Detection

This repository contains a modular Computer Vision pipeline developed in Python and OpenCV. Designed as a foundational prototype for Advanced Driver-Assistance Systems (ADAS), the project processes dashcam footage to extract and track critical traffic elements in real-time without relying on high-overhead deep learning inference. 

---

## 🎯 Core Features & Visual Demonstrations

### 1. Robust Lane Tracking (Day & Night Conditions)
Extracts and tracks lane boundaries using an optimized spatial pipeline. The system maintains stability across varying lighting conditions using a deque-based history buffer to smooth polynomial fits across frames.
* **Techniques Used:** CLAHE contrast enhancement, HSL/HSV color masking, Canny Edge Detection, Hough Transform, and 2nd-degree sliding-window polynomial fitting.

**Daytime Detection:**
<br>
<img src="Screen Shot 2026-05-12 at 17.14.21 PM.jpg" alt="Daytime Lane Detection" width="800"/>

**Nighttime Detection:**
<br>
<img src="Screen Shot 2026-05-12 at 17.14.45 PM.jpg" alt="Nighttime Lane Detection" width="800"/>

### 2. Vehicle Detection & Proximity Estimation
Identifies moving vehicles and estimates their real-world distance to simulate forward collision warnings.
* **Detection & Tracking:** Utilizes MOG2 Background Subtraction combined with morphological operations (opening/closing) to isolate moving foreground objects. A custom Centroid Tracking algorithm assigns unique IDs to maintain object permanence.
* **Distance Estimation:** Employs a pinhole camera model, utilizing a fixed focal length and real-world vehicle height approximations to calculate distance in meters. Displays threat levels dynamically (Warning vs. Danger).

**Vehicle Proximity Tracking:**
<br>
<img src="Screen Shot 2026-05-12 at 17.14.00 PM.jpg" alt="Vehicle Tracking" width="800"/>

### 3. Crosswalk Recognition
Detects pedestrian zebra crossings by analyzing specific geometric and high-contrast patterns within a targeted Region of Interest (ROI).
* **Algorithm:** Scans for white↔black pixel transitions to identify "stripy" regions. Validates bounding boxes based on minimum white pixel coverage, consecutive row/column transitions, and aspect ratio constraints.

**Crosswalk Identification:**
<br>
<img src="Screen Shot 2026-05-12 at 17.15.17 PM.jpg" alt="Crosswalk Detection" width="800"/>

---

## 🛠 Technical Architecture

The codebase is highly modular, allowing independent testing and execution of each core feature.

| Module | Core Functionality | Algorithms / Methods |
| :--- | :--- | :--- |
| `lane_detection.py` | Advanced Lane Tracking | Sliding Window, Histogram Peaks, `np.polyfit` |
| `car_detection.py` | Vehicle Tracking & Proximity | MOG2, Centroid Tracking, Pinhole Camera Model |
| `crosswalk.py` | Zebra Crossing Detection | Binary Transition Counting, Aspect Ratio Validation |
| `linefitting.py` / `lane_line.py`| Curve Fitting & Stabilization | Frame History Deque, Moving Average Smoothing |
| `filters.py` | Preprocessing Pipeline | CLAHE, Color Thresholding, Gaussian Blur |

---

## 🚀 Setup & Execution

### Prerequisites
* Python 3.x
* OpenCV (`cv2`)
* NumPy (`numpy`)

### Running the Modules
Ensure your video files and scripts are placed in the root directory. You can test each system independently:

```bash
# Test Vehicle Detection and Distance Tracking
python car_detection.py

# Test Crosswalk Recognition
python crosswalk.py

# Test Complete Lane Detection Pipeline
python lane_detection.py

Author: Ido S. 
        Charlie.an

B.Sc. Computer Science | Specializing in Machine Learning & Computer Vision
