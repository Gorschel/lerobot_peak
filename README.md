# GenICam / IDS peak camera integration for HuggingFace/LeRobot

This repository provides a modular GenICam/IDS peak backend for the LeRobot robotics framework.
It includes an OpenCV‑style video capture wrapper, a LeRobot camera driver, and a configuration
class to enable seamless use of USB3 and GigE vision cameras via the IDS peak SDK.

## Features
- OpenCV‑style API
- Asynchronous acquisition
- GenICam node access
- Acquisition management
- LeRobot integration
- Color & rotation utilities

## Repository Structure
```
genicam_video_capture.py
camera_genicam.py
configuration_genicam.py
```

## Installation
1. Install IDS peak SDK
2. Install dependencies:
```
pip install opencv-python numpy
```
3. Add modules to your LeRobot project

## Basic Usage
```python
from camera_genicam import GenICamCamera
from configuration_genicam import GenICamCameraConfig

config = GenICamCameraConfig(index_or_serial=0, fps=30, width=1920, height=1200)
cam = GenICamCamera(config)
cam.connect()
frame = cam.read()
cam.disconnect()
```

## Asynchronous Acquisition
```python
cam.connect()
frame = cam.async_read(timeout_ms=200)
cam.disconnect()
```

## Adjusting Camera Parameters
```python
from genicam_video_capture import GenICamVideoCapture
import cv2
cap = GenICamVideoCapture(0)
cap.set(cv2.CAP_PROP_GAIN, 5.0)
```

## Troubleshooting
- Ensure no other application is using the camera
- Increase async timeout if needed
