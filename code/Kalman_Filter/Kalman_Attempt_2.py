import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('videos/card.mp4')

# Create the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('code/Kalman_Filter/haarcascade_frontalface_default.xml')

# Initialize Kalman filter parameters
dt = 1/30.0
A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Q = np.array([[0.01, 0, 0.01, 0],
              [0, 0.01, 0, 0],
              [0, 0, 0.01, 0],
              [0, 0, 0, 0.01]])
R = np.array([[10, 0],
              [0, 10]])
x = np.array([0, 0, 0, 0])
P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Initialize the Kalman filter
def kalman_filter(z):
    # Predict
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)

    return x, P

# Initialize the first frame
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) > 0:
    x, y, w, h = faces[0]
    cx, cy = x + w/2, y + h/2
    z = np.array([[cx], [cy]])
    x, P = kalman_filter(z)
else:
    x = np.array([0, 0, 0, 0])
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the Kalman filter
    x, P = kalman_filter(z)

    # Predict the position of the face
    pred_z = np.dot(H, x)

    # Update the position of the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   
