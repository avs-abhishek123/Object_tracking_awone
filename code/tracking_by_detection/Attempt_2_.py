import cv2

"""
In this code, the cv2.selectROI() method is used to define the initial region of interest. 
The cv2.TrackerKCF_create() method is used to initialize the tracker, and then the tracker.
update() method is called to update the tracker with each new frame. 
The resulting bounding box is drawn on the frame using the cv2.rectangle() method.
"""

# Load the video
video = cv2.VideoCapture("videos/card.mp4")

# Read the first frame
ret, frame = video.read()

# Define the region of interest (ROI)
roi = cv2.selectROI(frame, False)

# Initialize the tracker
tracker = cv2.TrackerKCF_create()

# Initialize the tracker with the ROI and the first frame
tracker.init(frame, roi)

# Loop through the video frames
while True:
    # Read a new frame
    ret, frame = video.read()

    # If there are no more frames, break the loop
    if not ret:
        break

    # Update the tracker with the new frame
    success, roi = tracker.update(frame)

    # Draw the bounding box around the tracked object
    if success:
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()
