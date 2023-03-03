import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Create a FAST object detector
fast = cv2.FastFeatureDetector_create()

# Initialize the previous frame and keypoints
prev_frame = None
prev_kp = None

# Define the region of interest
roi = (200, 200, 200, 200)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If this is not the first frame, detect keypoints and calculate optical flow
    if prev_frame is not None:
        # Detect keypoints in the previous frame
        prev_kp = fast.detect(prev_frame, None)

        # Extract the coordinates of the keypoints and convert them to a numpy array
        prev_pts = np.float32([kp.pt for kp in prev_kp]).reshape(-1, 1, 2)

        # Define a mask to exclude points outside the region of interest
        mask = np.zeros_like(gray)
        x, y, w, h = roi
        mask[y:y+h, x:x+w] = 255

        # Calculate optical flow using the Lucas-Kanade method
        # curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_pts, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, mask=mask)
        # The mask argument is not a valid keyword argument for the cv2.calcOpticalFlowPyrLK function in OpenCV 4.7.0.

        curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_pts, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)

        # Filter out keypoints with a low optical flow error and draw them on the current frame
        good_kp = [kp for kp, s in zip(prev_kp, status) if s == 1 and mask[int(kp.pt[1]), int(kp.pt[0])] == 255]
        for kp in good_kp:
            x, y = kp.pt
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Update the previous frame
    prev_frame = gray

    # Show the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
