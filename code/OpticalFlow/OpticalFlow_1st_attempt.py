import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture(0)

# Define the initial frame and region of interest
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
mask = np.zeros_like(prev_frame)

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the new frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using the Lucas-Kanade method
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select good points
    good_prev_pts = prev_pts[status == 1]
    good_next_pts = next_pts[status == 1]

    # Draw the tracking lines
    for i, (prev_pt, next_pt) in enumerate(zip(good_prev_pts, good_next_pts)):
        x, y = prev_pt.ravel()
        u, v = next_pt.ravel()
        mask = cv2.line(mask, (x, y), (u, v), (0, 255, 0), 2)
        frame = cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)

    # Overlay the tracking lines on the frame
    img = cv2.add(frame, mask)

    # Show the output
    cv2.imshow('Object Tracking', img)

    # Update the previous frame, points, and mask
    prev_gray = gray.copy()
    prev_pts = good_next_pts.reshape(-1, 1, 2)
    mask = np.zeros_like(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
