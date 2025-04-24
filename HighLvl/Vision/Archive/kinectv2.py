import cv2
import numpy as np
import subprocess

protonect_path = "/home/aryan/libfreenect2/bin/Protonect"  # Update if needed
proc = subprocess.Popen([protonect_path, "cpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

stdout, stderr = proc.communicate()
print(stdout.decode(), stderr.decode())


# Wait for Protonect to initialize, then capture frames
while True:
    # Read frame from OpenCV
    ret, frame = cv2.VideoCapture(0).read()  
    if not ret:
        break
    
    # Convert frame to grayscale (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the RGB and Depth frames
    cv2.imshow("Kinect RGB", frame)
    cv2.imshow("Kinect Depth", gray)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
proc.terminate()
