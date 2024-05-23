
# Smart Surveillance System with YOLO v5

This repository contains the code for a Smart Surveillance System developed using YOLO (You Only Look Once) version 7. The system is designed to monitor surveillance camera feeds in real-time and perform various security tasks, including:

Detecting unauthorized persons entering restricted areas.
Identifying abandoned objects.
Recognizing specific actions such as fighting or vandalism.
Triggering an alarm sound and saving photos of detected incidents.
Additionally, the system utilizes Firebase for real-time face recognition, enhancing security measures.


## Features

- **Object Detection**: Utilizes YOLO v5 for real-time object detection in surveillance camera feeds.
- **Unauthorized Person Detection**: Detects and alerts when unauthorized persons enter restricted areas.
- **Abandoned Object Detection**: Identifies and notifies about abandoned objects in the surveillance area.
- **Action Recognition**: Recognizes specific actions such as fighting or vandalism and triggers appropriate responses.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- NumPy
- Pygame
- YOLO v5 (pre-trained weights and configuration files)
- Face-recognition
- dlib
- Firebase

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nafeessidd1/Smart-Surveillance-System.git
