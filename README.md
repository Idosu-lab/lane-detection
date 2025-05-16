# lane-detection
# 🚗 Lane & Object Detection with OpenCV

This project demonstrates computer vision techniques for **lane detection**, **crosswalk detection**, and basic **vehicle detection** using Python and OpenCV. It processes dashcam-style videos to identify relevant traffic elements, forming the foundation for advanced driver-assistance systems (ADAS).

---

## 🎯 Features

- 🛣️ **Lane Detection**: Detects and highlights lane lines in real-time.
- 🚶 **Crosswalk Detection**: Identifies zebra crossings by finding grouped white rectangles in specific zones.
- 🚗 **Basic Car Detection**: Implements experimental car detection from frontal dashcam footage.
- 🧪 Modular Codebase: Each task is isolated in its own file for clarity and flexibility.

---
| File               | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `lanedet.py`       | Main script for lane line detection      |
| `lane_line.py`     | Utility for tracking lane segments       |
| `linefitting.py`   | Curve and polynomial fitting experiments |
| `crosswalk.py`     | Independent crosswalk detection module   |
| `car_detection.py` | Initial attempt at detecting vehicles    |
| `filters.py`       | Image preprocessing functions            |
| `startofwork.py`   | Early setup and experimentation          |

🛠 Technologies

Python 3.x
OpenCV
NumPy
👤 Author

Ido.s
Computer Science Student | Passionate about Computer Vision
GitHub Profile
