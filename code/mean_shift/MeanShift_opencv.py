import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define initial region of interest (ROI)
ret, frame = cap.read()
x, y, w, h = 200, 200, 100, 100
track_window = (x, y, w, h)

# Set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set up the termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # Read a new frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply MeanShift algorithm to get the new location of the object
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw the object on the frame
        x, y, w, h = track_window
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Object Tracking', img)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

