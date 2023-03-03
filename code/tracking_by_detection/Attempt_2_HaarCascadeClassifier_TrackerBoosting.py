import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('code/tracking_by_detection/haarcascade_frontalface_default.xml')

# Create tracker object
tracker = cv2.TrackerBoosting_create()

# Read video
cap = cv2.VideoCapture(0)

# Read first frame
success, frame = cap.read()

# Detect faces in the first frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Initialize tracker with first frame and bounding box of first face
if len(faces) > 0:
    x, y, w, h = faces[0]
    bbox = (x, y, w, h)
    tracker.init(frame, bbox)

# Loop through frames
while True:
    # Read a new frame
    success, frame = cap.read()

    if not success:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    # If tracking was successful, draw the bounding box
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Tracking', frame)

    # Exit if user presses Esc
    if cv2.waitKey(1) == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
