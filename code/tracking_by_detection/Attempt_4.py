import cv2
"""
different tracking algorithm such as cv2.TrackerKCF_create()
"""
# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier('code/tracking_by_detection/haarcascade_frontalface_default.xml')

# Create a KCF tracker object
tracker = cv2.TrackerKCF_create()

# Open the video file
video = cv2.VideoCapture('sample.mp4')

# Read the first frame
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    exit()

# Detect the face in the first frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) == 0:
    print('Cannot detect face in the first frame')
    exit()

# Initialize the tracker
x, y, w, h = faces[0]
bbox = (x, y, w, h)
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Draw the bounding box
    if ok:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Tracking failure detected', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Release the resources
video.release()
cv2.destroyAllWindows()
