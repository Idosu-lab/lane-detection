```markdown
# 🚗 ADAS Computer Vision Pipeline: Lane, Vehicle & Crosswalk Detection

This repository contains a modular Computer Vision pipeline developed in Python and OpenCV. Designed as a foundational prototype for Advanced Driver-Assistance Systems (ADAS), the project processes dashcam footage to extract and track critical traffic elements in real-time without relying on high-overhead deep learning inference. 

## Project Goal

- Detect and highlight lane lines in real-time across varying lighting conditions
- Identify moving vehicles and estimate proximity to simulate forward collision warnings
- Detect pedestrian zebra crossings using geometric and high-contrast pattern analysis
- Establish a modular foundation for future integration with machine learning classifiers

## Project Structure

```text
lane-detection/
├── data/
│   ├── miamifirstbasic.avi               # Source dashcam video
│   └── miamifirstbasic2.avi              # Source dashcam video
├── figures/
│   ├── lane_day.jpg                      # Daytime lane detection example
│   ├── lane_night.jpg                    # Nighttime lane detection example
│   ├── car_tracking.jpg                  # Vehicle proximity tracking example
│   └── crosswalk_detection.jpg           # Crosswalk recognition example
├── src/
│   ├── lane_detection.py                 # Main script for lane line detection
│   ├── lane_line.py                      # Utility for tracking lane segments
│   ├── linefitting.py                    # Curve and polynomial fitting
│   ├── crosswalk.py                      # Independent crosswalk detection module
│   ├── car_detection.py                  # Vehicle tracking and distance estimation
│   ├── filters.py                        # Image preprocessing functions
│   ├── lanedet.py                        # Alternate lane detection implementation
│   └── startofwork.py                    # Early setup and ROI cropping
├── README.md
└── requirements.txt

```

## Core Features & Visual Demonstrations

### 1. Robust Lane Tracking (Day & Night Conditions)

Extracts and tracks lane boundaries using an optimized spatial pipeline. The system maintains stability across varying lighting conditions using a deque-based history buffer to smooth polynomial fits across frames.

* **Techniques Used:** CLAHE contrast enhancement, HSL/HSV color masking, Canny Edge Detection, Hough Transform, and 2nd-degree sliding-window polynomial fitting.

**Daytime Detection:**

**Nighttime Detection:**

### 2. Vehicle Detection & Proximity Estimation

Identifies moving vehicles and estimates their real-world distance to simulate forward collision warnings.

* **Detection & Tracking:** Utilizes MOG2 Background Subtraction combined with morphological operations (opening/closing) to isolate moving foreground objects. A custom Centroid Tracking algorithm assigns unique IDs to maintain object permanence.
* **Distance Estimation:** Employs a pinhole camera model, utilizing a fixed focal length and real-world vehicle height approximations to calculate distance in meters. Displays threat levels dynamically (Warning vs. Danger).

**Vehicle Proximity Tracking:**

### 3. Crosswalk Recognition

Detects pedestrian zebra crossings by analyzing specific geometric and high-contrast patterns within a targeted Region of Interest (ROI).

* **Algorithm:** Scans for white↔black pixel transitions to identify "stripy" regions. Validates bounding boxes based on minimum white pixel coverage, consecutive row/column transitions, and aspect ratio constraints.

**Crosswalk Identification:**

## How to Run

### Setup

```bash
pip install opencv-python numpy

```

Ensure your video files are placed in the root directory or `data/` folder (update paths in the scripts accordingly). You can test each system independently:

### Run Vehicle Detection and Distance Tracking

```bash
python src/car_detection.py

```

### Run Crosswalk Recognition

```bash
python src/crosswalk.py

```

### Run Complete Lane Detection Pipeline

```bash
python src/lane_detection.py

```

## Author

* Ido S.
* Charlie. A.N

```

```