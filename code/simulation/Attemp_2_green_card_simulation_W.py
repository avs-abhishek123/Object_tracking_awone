import numpy as np
from moviepy.editor import ImageSequenceClip

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

# Write frames to video
clip = ImageSequenceClip(frames, fps=fps)
clip.write_videofile("random_green_card_video.mp4", fps=fps)

# Remove frames from memory
del frames
