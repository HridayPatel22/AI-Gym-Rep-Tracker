# AI-Gym-Rep-Tracker

## Project Description
The AI Gym Rep Tracker is an application that utilizes AI-powered pose estimation to enhance your workout experience. By leveraging MediaPipe and Python, this project detects and analyzes various exercise poses in real-time from your webcam feed. The tool provides immediate feedback on exercise performance by calculating joint angles to determine the stage you are in your exercise and tracking repetitions, which are then rendered on the screen using OpenCV. 

## Problem 
Many gym-goers, including myself, know that tracking exercise performance and maintaining proper form can be difficult, especially during complex movements such as bicep curls or deadlifts. Manual counting of repetitions during your sets is prone to error, and ensuring that you have the correct exercise form while tracking how many reps you have left can be a pain. As a result, this can often lead to inefficient workouts and stress over missing out on the optimal amount of reps per exercise. This tool solves the need for automatated rep counting with high accuracy as well as allowing users to monitor progress in real-time. 

## Features 
The AI Gym Rep Tracker solves these issues by: 
- Pose Detection: Uses MediaPipe to detect and analyze exercise poses from a live webcam feed.
- Rep Counting: Automatically counts repetitions based on detected poses, reducing manual tracking errors during workouts.
- Stage Tracking: Monitors and displays the current stage in your exercise (e.g, "up" or "down"), ensuring correct exercise execution.
- Real-Time Visualization: Renders joint angles and repetition counts directly on the video feed using OpenCV for immediate feedback.

## Tech Stack 
- Programming Language: Python
- Libraries:
  - Open CV:  For video capture and visualization
  - MediaPipe: For pose detection and landmark analysis
  - NumPy: For angle calculations (np.array and np.arctan2 functions) and coordinate conversion (scale coordinates for annotations on video feed)
- IDE: Visual Studio Code

## Installation 
1. Clone the Repository:
- git clone https://github.com/HridayPatel22/AI-Gym-Rep-Tracker.git
3. Navigate to the Project Directory:
- cd AI-Gym-Rep-Tracker
4. Install dependencies:
- pip install -r requirements.txt

## Usage 
1. Run the Script:
- python main.py
2. Interact with the App
- The app with open a video feed from your webcam, showing real-time pose detection. Begin by moving your arms as if you were performing a bicep curl, and you will see the reps start to automatically accumulate as you begin to lift the weights along with your current postition during the lift. 
3. Customization
- Adjust the visual settings and colors in the constants right at the top to fit your preferences or in the def display_reps_and_stage function to change the box dimensions and spacing of the text.

## Code Structure 
This project is implemented in a single file 'main.py' which includes: 
  - Imports and Constants: Includes necessary libraries ('cv2, 'mediapipe', 'numpy') and defines necessary constants for annotations and display.
- Functions:
  - 'initialize_video_capture()': Sets up the webcam for video capture.
  - 'process_frame(pose, frame)': Processes video frames to detect pose landmarks.
  - 'draw_landmarks(image, results)': Draws pose landmarks on the image.
  - 'calculate_angle(a, b, c)': Calculates the angle between the three points.
  - 'get_landmarks(results)': Extracts landmarks from pose detection results.
  - 'annotate_angle_on_image(image, angle, position)': Adds angle annotations to the image.
  - 'display_reps_and_stage(image, counter, stage)': Displays the rep counter and exercise stage on the image.
  - 'main()': Runs the video capture and processing loop. 

## Contact 
For questions or additional information, please reach out to me: 
- Email: hpatel85@icloud.com
- LinkedIn: https://www.linkedin.com/in/hriday-patel-299195249/