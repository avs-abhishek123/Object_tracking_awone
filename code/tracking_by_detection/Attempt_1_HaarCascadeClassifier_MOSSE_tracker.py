# Haar Cascade Classifier and the MOSSE tracker

import cv2

# Initialize Haar Cascade Classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Get initial frame and detect faces
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# If face detected, select the first one to track
if len(faces) > 0:
    x, y, w, h = faces[0]
    bbox = (x, y, w, h)
    ok = tracker.init(frame, bbox)

# Loop through video frames and track the detected object
while True:
    # Read a new frame
    ret, frame = cap.read()
    
    # Track the object
    ok, bbox = tracker.update(frame)
    
    # If tracking successful, draw the bounding box around the object
    if ok:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


"""
It seems that the cv2 module doesn't have the TrackerMOSSE_create() method. 
It might be because of the OpenCV version I are using. 
The cv2.TrackerMOSSE_create() method was added in OpenCV 3.4.2. 
I can try upgrading our OpenCV version and then run the code again.
"""