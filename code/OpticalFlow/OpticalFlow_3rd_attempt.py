import cv2
import numpy as np
from cv2 import xfeatures2d

# Define the video capture object
cap = cv2.VideoCapture(0)

# Initialize the SIFT feature detector
sift = xfeatures2d.SIFT_create()

# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# Initialize the mask and previous frame
mask = None
prev_frame = None

while True:
    # Capture a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the new frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Compute SIFT keypoints and descriptors for the previous and current frames
        prev_kp, prev_desc = sift.detectAndCompute(prev_frame, None)
        kp, desc = sift.detectAndCompute(gray, None)

        # Match the keypoints using a FLANN-based matcher
        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(prev_desc, desc, k=2)

        # Select good matches using a ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Estimate the homography matrix using RANSAC
        if len(good_matches) > 10:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None:
                mask = np.zeros_like(frame)

            # Warp the mask and previous frame to the current frame using the homography matrix
            prev_warp = cv2.warpPerspective(prev_frame, M, (gray.shape[1], gray.shape[0]))
            mask_warp = cv2.warpPerspective(mask, M, (gray.shape[1], gray.shape[0]))

            # Compute optical flow using the Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_warp, cv2.COLOR_BGR2GRAY),
                                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Get the center and size
            h, w = prev_frame.shape[:2]
            center = np.array([w/2, h/2])
            size = np.array([w, h])

            # Use Kalman filter to update the center and size
            state = kalman.predict()
            predicted_center = state[0], state[1]
            predicted_size = state[2], state[3]

            if mask_warp is not None:
                mask_warp = cv2.bitwise_and(mask_warp, mask_warp, mask=prev_warp)

            # Update the Kalman filter measurement
            if np.sum(mask_warp) > 10:
                measurement = np.array([np.mean(np.nonzero(mask_warp)[::-1], axis=1)], np.float32)
                kalman.correct(measurement)

            # Draw a rectangle around the object
            tl, br = tuple((predicted_center - predicted_size / 2).astype(int)), \
                     tuple((predicted_center + predicted_size / 2).astype(int))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

            # Draw the optical flow
            for y in range(0, h, 10):
                for x in range(0, w, 10):
                    fx, fy = flow[y, x]
                    if mask_warp[y, x] == 255:
                        cv2.line(frame, (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1)

        # Update the previous frame and mask
        prev_frame = gray
        mask = np.zeros_like(frame)

    # Show the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
