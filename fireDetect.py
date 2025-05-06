import cv2
import numpy as np
from playsound import playsound
import threading

# Function to play sound in a separate thread
def play_alarm_sound():
    playsound('alert.mp3')

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for fire-like colors in HSV
    lower_fire = np.array([18, 50, 50])
    upper_fire = np.array([35, 255, 255])

    # Create a mask for fire color
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Count the number of fire-like pixels
    fire_pixels = cv2.countNonZero(mask)

    # If enough fire-like pixels detected, trigger alarm
    if fire_pixels > 15000:
        cv2.putText(frame, "Fire Detected!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Start alarm sound in a separate thread so it doesn't block video feed
        threading.Thread(target=play_alarm_sound, daemon=True).start()

    # Show the mask and the original frame
    cv2.imshow("Fire Mask", mask)
    cv2.imshow("Fire Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
