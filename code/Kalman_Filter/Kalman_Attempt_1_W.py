import cv2
import numpy as np

# initialize video capture and Kalman filter
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('code/Kalman_Filter/haarcascade_frontalface_default.xml')

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# initialize variables for tracking
measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)
while True:
    # read a frame from video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect object using some method, such as Haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        measurement = np.array([[x+w/2], [y+h/2]], np.float32)
    
    # predict object location using Kalman filter
    prediction = kalman.predict()
    
    # correct object location using measurement and Kalman filter
    kalman.correct(measurement)
    
    # draw rectangle around object location
    cv2.rectangle(frame, (int(prediction[0]-w/2), int(prediction[1]-h/2)), (int(prediction[0]+w/2), int(prediction[1]+h/2)), (0, 255, 0), 2)
    
    # display frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
