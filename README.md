# Development of an Autonomous Robot
## ENPM701 - Autonomous Robotics

## Overview
This project involves the design and implementation of an autonomous mobile manipulator robot that integrates both mobility and manipulation to handle complex industrial tasks. The aim was to develop a robot capable of navigating and performing tasks such as locating and transporting construction blocks in a simulated environment. The project was conducted between January 2024 and May 2024.

## Features
- **Integrated Design**: The robot combines a standard mobile platform with a manipulator arm to perform diverse tasks.
- **Autonomous Navigation**: Utilized landmark-based coordinates for path planning and obstacle avoidance to enable efficient navigation.
- **Object Interaction**: Capable of detecting, differentiating, and interacting with objects while handling obstacles in dynamic environments.
- **Task Optimization**: Focused on optimizing task planning and execution speed to ensure efficient task completion.

## Project Structure
- **Robot Construction**: The autonomous robot was constructed using:
  - A standard mobile platform with motors, wheels, and motor drivers.
  - A Raspberry Pi equipped with a camera for vision-based tasks.
  - Sonar range sensors for distance measurement and obstacle detection.
  - A servo gripper assembly to enable object manipulation.
- **Control Systems**: Developed using control theory and kinematic models:
  - Control Algorithms: Designed based on kinematic models to achieve precise movements of the robot and manipulator.
  - Sensors and Actuators: Integrated the sonar sensors, camera, motors, and gripper to create a fully functional mechatronic system.
- **Navigation and Object Handling**: The robot navigated autonomously using:
  - Landmark Coordinates: Implemented for path planning, allowing the robot to reach the desired locations while avoiding obstacles.
  - Obstacle Avoidance: Enabled the robot to handle different situations during navigation, ensuring safe interaction in the environment.

## Methodology
- Hardware Integration: Combined a mobile platform, manipulation arm, and a range of sensors to construct the robot. Key components included the Raspberry Pi, motors, sonar sensors, and the servo gripper.
- Control Algorithm Design: The control algorithms were developed based on kinematic and dynamic models to achieve accurate positioning and task execution. The goal was to enable the robot to effectively combine mobility and manipulation.
- Path Planning and Optimization: Path planning was achieved using a landmark-based system to reach specific points in the environment, ensuring efficient movement while avoiding obstacles.
- Task Planning: The robot's behavior was designed to optimize task execution, focusing on increasing speed and efficiency when locating and moving blocks.
- Object Detection and Interaction: The Raspberry Pi camera was used for object detection, combined with sonar sensors for precise distance measurements. Image processing techniques were applied to detect, differentiate, and interact with various objects.

## Key Challenges
- Part Recognition: Developed and fine-tuned the recognition system for accurately identifying parts using vision and sonar data.
- Sensor Calibration: Ensured sensors such as sonar and the camera were calibrated for accurate and consistent readings, crucial for navigation and object interaction.
- Environmental Adaptation: Designed robust control and detection algorithms to handle diverse environmental conditions, ensuring the robot performed reliably regardless of changes in lighting or obstacles.

## Dependencies and Libraries Used
- Raspberry Pi OS: Used for controlling the hardware components of the robot.
- OpenCV: For image processing and object detection.
- Python: Main programming language used for algorithm development and hardware integration.
- RPi.GPIO: Library used for interfacing with the hardware connected to the Raspberry Pi, including motors, sensors, and servos.
- SciPy and NumPy: Used for mathematical computations required in control algorithm development.

## Key Results
- Successful Task Execution: The robot demonstrated the ability to autonomously locate and transport construction blocks in a simulated industrial environment.
- Optimized Task Planning: The robot was able to efficiently plan its path and execute its tasks, minimizing the time required for each task.
- Effective Object Interaction: Integrated sensors and the manipulator allowed the robot to reliably interact with and manipulate objects, avoiding obstacles and adjusting to varying conditions.

## Robot
![](https://github.com/vishnumandala/Development-of-an-Autonomous-Robot/blob/main/1.png)  
![](https://github.com/vishnumandala/Development-of-an-Autonomous-Robot/blob/main/2.png)

## Arena
![](https://github.com/vishnumandala/Development-of-an-Autonomous-Robot/blob/main/Arena%203D.png)

## Progress Video
The video documenting the process and the final result can be viewed at: https://youtu.be/_uw1MORqhX0?feature=shared
