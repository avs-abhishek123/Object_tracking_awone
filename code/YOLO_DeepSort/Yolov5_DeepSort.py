import cv2
import torch
import numpy as np
from deep_sort import DeepSort
from yolov5 import YOLOv5

# Initialize the YOLOv5 model and DeepSORT tracker
yolo = YOLOv5(weights='yolov5s.pt')
deepsort = DeepSort()

# Initialize the video capture object
cap = cv2.VideoCapture('test_video.mp4')

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get the video frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the output video writer object
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

while True:
    # Capture the frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLOv5 on the frame to get the detections
    detections = yolo(frame)

    # Extract the bounding boxes and class ids from the detections
    boxes = []
    class_ids = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_id = detection
        if cls_id == 0:  # Only track cars
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            class_ids.append(cls_id)

    # Pass the detections to DeepSORT tracker
    outputs = deepsort.update(np.array(boxes), np.array(class_ids), frame)

    # Draw the tracked objects on the frame
    for output in outputs:
        bbox = output[0:4]
        identity = output[4]
        color = (255, 0, 0)  # Blue color for car tracking
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, str(identity), (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write the output frame to the output video
    out.write(frame)

    # Display the output frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
