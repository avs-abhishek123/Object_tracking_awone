import cv2
import numpy as np

"""
The xfeatures2d module is not available in OpenCV 4.0 and above. 
Instead, you can use the SIFT_create() and drawKeypoints() methods from the cv2 module 
to detect and draw SIFT keypoints.
"""

cap = cv2.VideoCapture(0)

# Create a SIFT object detector
sift = cv2.SIFT_create()

# Initialize the previous frame and keypoints
prev_frame = None
prev_kp = None

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If this is not the first frame, detect SIFT keypoints and calculate optical flow
    if prev_frame is not None:
        # Detect SIFT keypoints in the previous frame
        prev_kp = sift.detect(prev_frame, None)

        # Calculate optical flow using the SIFT keypoints
        curr_kp, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_kp, None)

        # Filter out keypoints with a low optical flow error and draw them on the current frame
        good_kp = curr_kp[status == 1]
        for kp in good_kp:
            x, y = kp.ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Update the previous frame
    prev_frame = gray

    # Show the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
