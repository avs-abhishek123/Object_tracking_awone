import cv2
import numpy as np

# Define the dimensions of the frame
resolution = (3840,2160)

# Define the dimensions of the rectangle
rect_width = 500
rect_height = 200
rect_top = (resolution[1] - rect_height) // 2
rect_bottom = rect_top + rect_height

# Define the background color (white)
bg_color = (255, 255, 255)

# Define the length of the video in frames
video_length = 1000

# Calculate the speed of the rectangle to cover the whole width of the frame in `video_length` frames
rect_dx = resolution[0] / video_length

# Create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("vid/2_1.mp4", fourcc, 30.0, resolution, isColor=True)

# Generate the frames of the video
for i in range(video_length):
    # Create a new frame with the background color
    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    frame[:, :] = bg_color

    # Calculate the position of the rectangle at the current frame
    rect_left = int(i * rect_dx)
    rect_right = int(rect_left + rect_width)

    # Draw the green rectangle at the current position
    cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (0, 255, 0), -1)

    # Write the frame to the video
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()
