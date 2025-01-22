# Homemade-FocusTracker with Facial Recognition

This project is a work monitoring system that uses facial recognition and object detection to track the presence and activity of a specific person.  
The system can detect when you're in focus mode, on a break, or using your mobile phone, providing real-time metrics on focus time, break time, and phone usage.

## Features

- **Personalized Facial Recognition:** Initial calibration to automatically recognize Pedro (if you work sideways, move your head during calibration).  
- **Object Detection:** Uses the YOLO model to identify specific objects such as bottles and glasses.  
- **State Management:**
  - **Focus Time:** Total time Pedro is focused on work.
  - **Pause Time:** Total time Pedro is on a break (not focused or using the phone).
  - **Phone Time:** Total time Pedro uses the mobile phone.
  - **Start of Focus Time:** Exact time Pedro started focusing.  
- **Visual Interface:** Real-time display of counters for Focus Time, Pause Time, Phone Time, and the start time of Focus Time directly on the video window.  
- **Body Tracking:** Integration with MediaPipe Pose to maintain Pedro’s detection even when his face turns.  

## Requirements

- **Operating System:** Windows, macOS, or Linux.  
- **Programming Language:** Python 3.10  
- **Dependencies:**
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - `face_recognition`
  - `dlib` (required by `face_recognition`)  

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/work-monitoring.git
   cd work-monitoring

2.	**YOLO Model**
	•	Create a yolo folder.
	•	Download the yolov3.weights, yolov3.cfg, and coco.names files from YOLO website.
