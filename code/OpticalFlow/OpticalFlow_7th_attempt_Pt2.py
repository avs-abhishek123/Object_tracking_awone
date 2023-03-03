import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Farneback optical flow
fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)

# Random colors for visualizing tracks
color = np.random.randint(0, 255, (100, 3))

# Get the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get the corners for the first frame
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)

while True:
    # Get the next frame
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, **fb_params)
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, pyr_scale=fb_params['pyr_scale'], levels=fb_params['levels'], winsize=fb_params['winsize'], iterations=fb_params['iterations'], poly_n=fb_params['poly_n'], poly_sigma=fb_params['poly_sigma'], flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, fb_params['pyr_scale'], fb_params['levels'], fb_params['winsize'], fb_params['iterations'], fb_params['poly_n'], fb_params['poly_sigma'], cv2.OPTFLOW_USE_INITIAL_FLOW)
    # curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, **fb_params, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, **fb_params, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

    
    good_new = curr_pts[status == 1]
    good_old = prev_pts[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    # Display the resulting frame
    cv2.imshow('frame', img)

    # Update the previous points and previous frame
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""
Error
  File "c:\Users\MSI\Desktop\Awone\code\OpticalFlow\ss.py", line 40, in <module>
    curr_pts, status, error = cv2.calcOpticalFlowFarneback(prev_gray, gray, prev_pts, None, **fb_params, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'calcOpticalFlowFarneback'      
> Overload resolution failed:
>  - calcOpticalFlowFarneback() takes at most 10 arguments (11 given)
>  - calcOpticalFlowFarneback() takes at most 10 arguments (11 given)

"""