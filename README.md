# Gesture Arm Pro – AI Hand-Controlled Robotic Arm Simulator

## Overview

Gesture Arm Pro is a computer-vision-based robotic arm simulator that allows users to control a virtual robotic manipulator using hand gestures detected through a webcam.

The system uses MediaPipe hand tracking and OpenCV for real-time gesture recognition, while a custom robotic arm simulation provides smooth motion, gripper control, object pickup, and autonomous task execution.

This project was developed as part of a Robotics & Automation Internship.

---

## Features

### Manual Gesture Control

* 1 Finger → Rotate Base Left
* 2 Fingers → Rotate Base Right
* 3 Fingers → Move Arm Up
* 4 Fingers → Move Arm Down
* Fist → Close Gripper
* Open Palm → Open Gripper

### Computer Vision

* Real-time hand detection using MediaPipe
* Finger counting and gesture recognition
* Webcam-based interaction

### Robotic Arm Simulation

* Multi-joint robotic arm
* Smooth animated movement
* Gripper mechanism
* Real-time angle display

### Pick-and-Place System

* Object detection and interaction
* Box pickup functionality
* Box transport and release
* Visual feedback during manipulation

### Autonomous Mode

* Automated pick-and-place workflow
* State-machine-based task execution
* Automatic object selection
* Return-to-home functionality

### User Interface

* Modern robotics-inspired dashboard
* Live status indicators
* Gesture display panel
* Task completion counter
* Interactive workspace visualization

---

## Technologies Used

* Python 3.10
* OpenCV
* MediaPipe
* NumPy
* Pygame
* Computer Vision
* Robotics Simulation

---

## Project Structure

```text
Gesture-Controlled-Arm/
│
├── gesture_arm_control.py
├── robot_arm_sim.py
├── hand_tracking.py
├── finger_counter.py
├── .gitignore
└── assets/
    └── screenshots/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/adithyamadhupal-dev/gesture-controlled-robotic-arm.git
cd gesture-controlled-robotic-arm
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install opencv-python mediapipe numpy pygame
```

### Run Project

```bash
python gesture_arm_control.py
```

---

## How It Works

1. Webcam captures the user's hand.
2. MediaPipe detects hand landmarks.
3. Finger counting logic identifies gestures.
4. Gestures are translated into robotic arm commands.
5. The robotic arm responds in real time.
6. Objects can be picked, transported, and released using the gripper.
7. Autonomous mode executes predefined pick-and-place tasks.

---


## Future Improvements

* Advanced Inverse Kinematics
* Improved Autonomous Navigation
* Multiple Object Handling
* 3D Robotic Arm Visualization
* ROS Integration
* Industrial Robot Simulation

---

## Learning Outcomes

* Computer Vision
* Human-Robot Interaction
* Gesture Recognition
* Robotics Simulation
* State Machines
* Python Application Development
* Autonomous Control Systems

---

## Author

Adithya Madhupal

GitHub:
https://github.com/adithyamadhupal-dev

---

## License

This project is intended for educational and internship demonstration purposes.
