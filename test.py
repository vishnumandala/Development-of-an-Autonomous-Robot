import RPi.GPIO as gpio
import time
import numpy as np
import cv2
import re
import serial
from picamera import PiCamera
from picamera.array import PiRGBArray
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import os

smtp_user = 'kingofminister@gmail.com'
smtp_pass = 'jnsl njnh vdwa kvtk'
smtp_server = 'smtp.gmail.com'
smtp_port = 587

# Email sender and recipient
email_from = smtp_user
email_to = ['kingofminister@gmail.com']
# # email_to = ['ENPM809TS19@gmail.com','kingofminister@gmail.com']

# Constants
wheel_circumference = 2 * np.pi * 0.0325  # Circumference of the wheel in meters
cm_to_meters = 0.01
counts_per_revolution = 960  # Assuming 960 counts per revolution
PWM_FREQUENCY = 50  # Adjust this to find optimal operation
LOW_DUTY_CYCLE = 25 # Low duty cycle to reduce speed near target orientation
FINE_TUNE_THRESHOLD = 15 # Threshold for fine-tuning orientation
PIXEL_TO_DEGREE_RATIO = 0.061
CAMERA_CENTER_X = 320
KNOWN_DISTANCE = 0.5  # Known distance from the object in meters
KNOWN_WIDTH = 0.038  # Known width of the object in meters 
PIXEL_WIDTH = 38  # Width of the object in pixels

gpio.setmode(gpio.BOARD)

# Setup GPIO for ultrasonic sensor
TRIG = 16
ECHO = 18
gpio.setup(TRIG, gpio.OUT)
gpio.setup(ECHO, gpio.IN)

# Setup GPIO for servo
SERVO_PIN = 36  
gpio.setup(SERVO_PIN, gpio.OUT)
servo_pwm = gpio.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo_pwm.start(3)

# HSV color bounds
# color_bounds = {
#     'red': (np.array([159, 115, 48]), np.array([255, 255, 255])),
#     'green': (np.array([28, 55, 71]), np.array([94, 255, 255])),
#     'blue': (np.array([100, 86, 54]), np.array([160, 255, 255]))
# }

color_bounds = {
    'red': (np.array([170, 79, 75]), np.array([255, 255, 255])),
    'green': (np.array([30, 79, 75]), np.array([100, 255, 255])),
    'blue': (np.array([57, 79, 75]), np.array([122, 255, 255]))
}
color_order = ['red', 'green', 'blue']  # Order to pick up blocks

# Motor and encoder pin definitions
motor_pins = [31, 33, 35, 37]
encoder_pins = [(7, gpio.PUD_UP), (12, gpio.PUD_UP)]
serial_port = '/dev/ttyUSB0'

# Camera calibration parameters
focal_length = (PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH
distance_to_object = None  # Initialize distance to object

task_completed = False  # Flag to indicate task completion
count = 0
degrees_rotated = 0
initial_search_direction = 'left'  # Initially search to the left
distance_to_go = 42
left_wall = 182
right_wall = 273

# Initialize camera and GPIO
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32 
raw_capture = PiRGBArray(camera, size=(640, 480))
ser = serial.Serial(serial_port, 9600)

frames_folder = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)
    
def init():
    gpio.setmode(gpio.BOARD)
    for pin in motor_pins:
        gpio.setup(pin, gpio.OUT)
    for pin, pud in encoder_pins:
        gpio.setup(pin, gpio.IN, pull_up_down=pud)

def capture_image(filename):
    camera.capture(filename)
    print(f"Captured image {filename}")

def gameover():
    for pin in motor_pins:
        gpio.output(pin, False)

def setup_pwm(pin, base_value):
    pwm = gpio.PWM(pin, PWM_FREQUENCY)
    pwm.start(base_value)
    gpio.output(pin, True)
    return pwm

def move(base_val, motor1_pin, motor2_pin):
    gameover()
    pwm1 = setup_pwm(motor1_pin, base_val)
    pwm2 = setup_pwm(motor2_pin, base_val)
    return pwm1, pwm2

def send_email(image_path, target_color, block_number):
    subject = f'ENPM701 - Grand Challenge | Trial 1 - {os.path.basename(image_path)}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} - [Block {block_number}: {target_color.upper()}] - Vishnu Mandala | UID: 119452608'
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = email_from
    msg['To'] = ', '.join(email_to)

    # Attach text body
    body = MIMEText(f"Image recorded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    msg.attach(body)

    # Attach image
    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
        msg.attach(img)

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(email_from, email_to, msg.as_string())
        print('Email delivered!')

def measure_distance():
    gpio.output(TRIG, False)
    time.sleep(0.01)

    gpio.output(TRIG, True)
    time.sleep(0.00001)
    gpio.output(TRIG, False)

    while gpio.input(ECHO) == 0:
        pulse_start = time.time()
    while gpio.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    return distance

def control_servo(open=True):
    if open:
        servo_pwm.ChangeDutyCycle(7)  # Adjust for the open position of your gripper
    else:
        servo_pwm.ChangeDutyCycle(3)  # Adjust for the closed position of your gripper
    time.sleep(0.5)  # Allow some time for the servo to move

def forward(tf):
    gpio.output(31, True)
    gpio.output(33, False)
    gpio.output(35, False)
    gpio.output(37, True)
    time.sleep(tf)
    gameover()
    # gpio.cleanup()

def drive_motors(direction, target_orientation=None, angle_degrees=None, distance_cm=None, base_val=80):
    init()
    counterBR, counterFL = np.uint64(0), np.uint64(0)
    buttonBR, buttonFL = int(0), int(0)
    fine_tune_mode = False
    distance = 0
    base_val = base_val if distance_cm else 30
    if angle_degrees:
        distance = (angle_degrees / 360) * wheel_circumference
    elif distance_cm:
        distance = distance_cm * cm_to_meters
        
    revolutions_needed = distance / wheel_circumference
    counts_needed = revolutions_needed * counts_per_revolution

    direction_config = {
        'forward': (base_val, 31, 37),
        'reverse': (base_val, 33, 35),
        'pivotleft': (base_val, 33, 37), 
        'pivotright': (base_val, 31, 35)
    }

    base_val, motor1_pin, motor2_pin = direction_config[direction]
    pwm1, pwm2 = move(base_val, motor1_pin, motor2_pin)
        
    while True:
        if not fine_tune_mode:
            # adjust_motors_based_on_pid(pwm1, pwm2, target_orientation)
            if int(gpio.input(12)) != buttonBR:
                buttonBR = int(gpio.input(12))
                counterBR += 1
            if int(gpio.input(7)) != buttonFL:
                buttonFL = int(gpio.input(7))
                counterFL += 1

        if distance_cm:  # Check if the move is linear and needs distance tracking
            if counterBR >= counts_needed and counterFL >= counts_needed:
                time.sleep(0.1)
                break  # Stop motors when target distance is reached

        if angle_degrees:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    current_orientation = float(numbers[0])
                    orientation_error = (target_orientation - current_orientation + 360) % 360
                    if orientation_error > 180:
                        orientation_error -= 360  # Normalize error to -180 to 180
                    print(f"Current orientation: {current_orientation}, Orientation error: {orientation_error}")
                    if abs(orientation_error) <= FINE_TUNE_THRESHOLD:
                        fine_tune_mode = True  # Activate fine-tuning mode
                        pwm1.ChangeDutyCycle(LOW_DUTY_CYCLE)
                        pwm2.ChangeDutyCycle(LOW_DUTY_CYCLE)
                    elif abs(orientation_error) > FINE_TUNE_THRESHOLD:
                        fine_tune_mode = False  # Deactivate fine-tuning mode
                    if abs(orientation_error) <= 1:  # Threshold of 2 degrees
                        break  # Stop adjusting when within 2 degrees of target

    pwm1.stop()
    pwm2.stop()
    gameover()
    # gpio.cleanup()

def process_frame(image, target_color):
    top_crop = int(image.shape[0] * 0.3)

    # Crop the top 30% of the frame
    cropped_image = image[top_crop:, :, :]
    image_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, color_bounds[target_color][0], color_bounds[target_color][1])
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ########################## Final Method #########################################
    if len(contours) > 1:
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        second_largest_contour = contours[1]

        if cv2.contourArea(largest_contour) > 25:
            # Check if the second largest is similar in size (e.g., within 30% of the largest)
            if cv2.contourArea(second_largest_contour) / cv2.contourArea(largest_contour) > 0.7:
                # Calculate centers
                M1 = cv2.moments(largest_contour)
                M2 = cv2.moments(second_largest_contour)
                if M1['m00'] > 0 and M2['m00'] > 0:
                    cX1 = int(M1['m10'] / M1['m00'])
                    cX2 = int(M2['m10'] / M2['m00'])
                    # Choose the contour closest to the center of the camera frame
                    if abs(cX1 - CAMERA_CENTER_X) < abs(cX2 - CAMERA_CENTER_X):
                        selected_contour = largest_contour
                    else:
                        selected_contour = second_largest_contour
            else:
                selected_contour = largest_contour
        else:
            return None, None
    elif len(contours) == 1 and cv2.contourArea(contours[0]) > 25:
        selected_contour = contours[0]
    else:
        return None, None

    # Calculate centroid of the selected contour
    M = cv2.moments(selected_contour)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"]) + top_crop 
        cv2.drawContours(image, [selected_contour], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)

        rect = cv2.boundingRect(selected_contour)
        x, y, w, h = rect
        y += top_crop
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate displacement
        displacement = cX - CAMERA_CENTER_X
        angle_to_rotate = displacement * PIXEL_TO_DEGREE_RATIO
        return angle_to_rotate, w
    return None, None

def execute_measurement(angle_to_rotate, width, distance, base_val=80):
    global focal_length, KNOWN_WIDTH, count
    if abs(angle_to_rotate) <= 1.5:  # Check if the block is within ±1 degrees of center
        print("Block is within the 3-degree threshold from center. No rotation needed.")
        print(f"Object width in pixels: {width}, Distance to object: {distance} cm")
        drive_motors('forward', distance_cm=distance, base_val=base_val)
    else:
        initial_orientation = read_imu_orientation()
        print("Angle orientation:", angle_to_rotate)
        if angle_to_rotate > 0:
            direction = 'pivotright'
        else:
            direction = 'pivotleft'
        target_orientation = (initial_orientation + angle_to_rotate) % 360 if direction.startswith('pivotright') else (initial_orientation + angle_to_rotate + 360) % 360 
        print("Target orientation:", target_orientation)
        drive_motors(direction = direction, target_orientation = target_orientation, angle_degrees = angle_to_rotate)

def read_imu_orientation():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if numbers:
            return float(numbers[0])
    return 0

def image_distance_measurement(width):
    return (KNOWN_WIDTH * focal_length * 100) / width

def reverse_and_position():
    global initial_search_direction, degrees_rotated, distance_to_go, left_wall, right_wall
    # Step 1: Reverse 20 cm
    drive_motors('reverse', distance_cm=20)
    time.sleep(0.5)

    # Step 2: Rotate to exactly 180 degrees
    correct_orientation_to(left_wall)
    time.sleep(0.5)

    # Step 3: Move forward based on ultrasonic reading minus 40 cm
    while True:
        first = measure_distance()
        if first > 250:
            forward(0.1)
        else:
            break

    forward_distance = measure_distance() - distance_to_go
    print(f"Moving forward by {forward_distance} cm")
    if forward_distance > 0:
        drive_motors('forward', distance_cm=forward_distance)
    time.sleep(0.5)

    # Step 4: Rotate 90 degrees to the right
    correct_orientation_to(right_wall)
    time.sleep(0.5)

    # Step 5: Move forward based on ultrasonic reading minus 40 cm
    while True:
        first = measure_distance()
        if first > 250:
            forward(0.1)
        else:
            break

    forward_distance = measure_distance() - distance_to_go
    print(f"Moving forward by {forward_distance} cm")
    if forward_distance > 0:
        drive_motors('forward', distance_cm=forward_distance)
    time.sleep(0.5)

    # Step 6: Drop the block
    control_servo(open=True)
    print("Block dropped successfully")
    
    # Step 7: Reverse 25 cm
    print("Reversing 25 cm")
    drive_motors('reverse', distance_cm=25)
    time.sleep(0.5)
       
    # Step 8: Rotate 45 degrees to the right
    correct_orientation_to(335)
    time.sleep(0.5)
    
    #Step 9: Close gripper
    control_servo(open=False)
    time.sleep(0.5)
    
    initial_search_direction = 'right'
    degrees_rotated = 0  # Reset the rotation count after setting the search direction
    print("Orientation reset to 0 degrees and search direction set to right. Ready to search.")

def correct_orientation_to(target_orientation):
    while True:
        current_orientation = read_imu_orientation()
        orientation_error = (target_orientation - current_orientation + 360) % 360
        if orientation_error > 180:
            orientation_error -= 360  # Normalize error to -180 to 180

        # Stop adjusting if within a tight tolerance of 3 degrees
        if abs(orientation_error) < 3:
            print(f"Orientation adjusted to within 3 degrees of {target_orientation} degrees.")
            break

        # Determine rotation direction based on the error
        direction = 'pivotright' if orientation_error > 0 else 'pivotleft'
        print(f"Adjusting orientation by {abs(orientation_error)} degrees towards {direction}.")
        drive_motors(direction, target_orientation=target_orientation, angle_degrees=abs(orientation_error))
        time.sleep(0.5)  # Short delay to allow for sensor refresh

def execution():
    global distance_to_object, focal_length, KNOWN_WIDTH, task_completed, initial_search_direction, degrees_rotated, count, distance_to_go, left_wall, right_wall

    # Allow the camera to warm up
    time.sleep(1.5)
    control_servo(open=False)
    initial_orientation = 0
    if ser.in_waiting > 0:
        count += 1
        line = ser.readline().decode('utf-8').strip()
        if count > 20:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if numbers:
                initial_orientation = float(numbers[0])

    search_attempts = 0  # Track the number of search attempts
    max_search_attempts = 3  # Maximum number of search attempts
    first = True  # Flag to indicate the first search attempt

    # Initialize variables for block tracking
    blocks_picked_up = 0
    current_color_index = 0
    target_color = color_order[current_color_index]

    try:
        camera.start_preview()
        frame_count = 0
        while True:
            print(f'Searching for {target_color} block')
            
            # Capture frames from the camera
            for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                try:
                    angle_to_rotate, w = process_frame(frame.array, target_color)
                    if angle_to_rotate is None:
                        if degrees_rotated >= (360 if initial_search_direction == 'left' else 115):
                            print("Rotation limit reached. No block found. Stopping search.")
                            if search_attempts < max_search_attempts and initial_search_direction == 'left':
                                print("Performing forward movement for another search attempt.")
                                drive_motors('forward', distance_cm=10)  # Move forward 10cm
                                initial_orientation = read_imu_orientation()
                                degrees_rotated = 0
                                search_attempts += 1
                                continue
                            elif search_attempts < max_search_attempts and initial_search_direction == 'right':
                                print("Performing another search attempt.")
                                drive_motors('pivotleft', target_orientation=0, angle_degrees=90)  # Rotate left
                                time.sleep(0.5)
                                drive_motors('forward', distance_cm=13)  # Move forward 10cm
                                time.sleep(0.5)
                                drive_motors('pivotleft', target_orientation=335, angle_degrees=25)  # Rotate left
                                time.sleep(0.5)
                                initial_orientation = read_imu_orientation()
                                degrees_rotated = 0
                                search_attempts += 1
                                continue

                        # Rotate 5 degrees right if block is not found in the current frame
                        rotation_increment = -10 if initial_search_direction == 'left' else 10 

                        target_orientation = (initial_orientation + rotation_increment) % 360
                        direction = 'pivotleft' if rotation_increment < 0 else 'pivotright'
                        print(f"No block detected. Rotating {direction} to {target_orientation} degrees.")
                        drive_motors(direction, target_orientation=target_orientation, angle_degrees=abs(rotation_increment))
                        degrees_rotated += abs(rotation_increment)
                        initial_orientation = target_orientation

                    else:
                        if first and abs(angle_to_rotate) < FINE_TUNE_THRESHOLD:
                            drive_motors('forward', distance_cm=10, base_val=25)
                            first = False
                            continue
                        distance_to_object = image_distance_measurement(w)  # Distance calculation
                        if distance_to_object >= 280:
                            print("Detected block too far away. Ignoring and searching for another block.")
                            continue
                        elif 50 <= distance_to_object < 280:
                            execute_measurement(angle_to_rotate, w, distance_to_object * 0.8)
                            time.sleep(0.5)
                        elif 30 <= distance_to_object < 50:
                            execute_measurement(angle_to_rotate, w, distance_to_object * 0.3, base_val=35)
                            time.sleep(0.5)                            
                        elif 21 <= distance_to_object < 30:
                            execute_measurement(angle_to_rotate, w, distance_to_object * 0.3, base_val=25)
                            time.sleep(0.5)
                        else:
                            if abs(angle_to_rotate) > 2.5:
                                print("Block detected close but not centered. Reversing and reorienting.")
                                drive_motors('reverse', distance_cm=17, base_val=30)
                                time.sleep(0.5)
                                continue

                            print("Close enough to grab the block")
                            control_servo(open=True)
                            drive_motors('forward', distance_cm=12, base_val=25)
                            control_servo(open=False)
                            image_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                            capture_image(image_filename)
                            send_email(image_filename, target_color, blocks_picked_up + 1)
                            print("Block retrieved. Coming Back...")
                            reverse_and_position()
                            
                            blocks_picked_up += 1
                            current_color_index = (current_color_index + 1) % len(color_order)
                            target_color = color_order[current_color_index]
                            print(f"Block picked up. Total blocks: {blocks_picked_up}")
                            
                            if blocks_picked_up > 5:
                                left_wall = 177
                                right_wall = 267
                            elif blocks_picked_up > 3:
                                distance_to_go = 35
                            
                            # if blocks_picked_up == 9:
                            #     print("All blocks picked up. Exiting the program.")
                            #     return
                            search_attempts = 0  # Reset search attempts after picking up a block
                            task_completed = True # Set the flag to indicate task completion
                            break
                        
                    # Save the frame to the folder
                    frame_filename = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame.array)
                    frame_count += 1  # Increment frame count
                    
                finally:
                    cv2.imshow("Frame", frame.array)
                    raw_capture.truncate(0)
                    raw_capture.seek(0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if task_completed:
                task_completed = False
                degrees_rotated = 0
                initial_orientation = read_imu_orientation()
                continue
    finally:
        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()
        
try:
    execution()
except Exception as e:
    print("An error occurred:", e)
finally:
    servo_pwm.stop()  # Stop PWM
    gpio.cleanup()  
    print("Program ended and cleaned up successfully.")