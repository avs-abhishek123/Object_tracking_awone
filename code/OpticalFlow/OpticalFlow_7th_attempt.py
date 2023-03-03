# FAST feature detection and Farneback optical flow

import cv2
import numpy as np

# Initialize FAST feature detector
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# Initialize Farneback optical flow parameters
fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Initialize video capture object
cap = cv2.VideoCapture('videos/card.mp4')

# Get first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = fast.detect(prev_gray, None)

# Initialize mask for visualizing tracked points
mask = np.zeros_like(prev_frame)

# Loop over all frames
while True:
    # Read current frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback algorithm
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prevPts, None, **fb_params)
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, **fb_params)

    # Farneback Parameters
    fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Lucas Kanade Parameters
    lk_params = dict(winSize=(200,200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback algorithm
        if prev_gray is not None:
            curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, pyr_scale=fb_params['pyr_scale'], levels=fb_params['levels'], winsize=fb_params['winsize'], iterations=fb_params['iterations'], poly_n=fb_params['poly_n'], poly_sigma=fb_params['poly_sigma'], flags=fb_params['flags'])
            curr_good_pts = curr_pts[status==1]
            prev_good_pts = prev_pts[status==1]


    # Select good points
    good_pts = curr_pts[status == 1]
    good_prev_pts = prev_pts[status == 1]
    
    # Draw lines between tracked points
    for i, (new, old) in enumerate(zip(good_pts, good_prev_pts)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    
    # Combine mask and frame to visualize tracked points
    img = cv2.add(frame, mask)
    
    # Show output
    cv2.imshow('Object Tracking', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    # Update previous frame and points
    prev_gray = gray.copy()
    prev_pts = good_pts.reshape(-1, 1, 2)

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
