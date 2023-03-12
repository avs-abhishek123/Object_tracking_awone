import numpy as np
import cv2

# Define video parameters
resolution = (1920, 1080)
fps = 30
duration = 10
num_frames = fps * duration

# Generate green frames
frames = []
for i in range(num_frames):
    frame = np.zeros(resolution + (3,), dtype=np.uint8)
    frame[:, :, 1] = 255
    frames.append(frame)

# Encode frames into video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("random_green_card_video.mp4", fourcc, fps, resolution)
for frame in frames:
    video_writer.write(frame)
video_writer.release()

# Remove frames from memory
del frames
