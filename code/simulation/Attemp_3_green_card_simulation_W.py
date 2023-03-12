import numpy as np
from moviepy.editor import ImageSequenceClip

# Define video parameters
resolution = (1920, 1080)
fps = 30
duration = 10
num_frames = fps * duration

# Define rectangle parameters
rect_width = 600
rect_height = 400
rect_top = (resolution[1] - rect_height) // 2
rect_bottom = rect_top + rect_height
rect_left = (resolution[0] - rect_width) // 2
rect_right = rect_left + rect_width

# Generate frames
frames = []
for i in range(num_frames):
    # Create white background
    frame = np.zeros(resolution + (3,), dtype=np.uint8)
    frame[:] = (255, 255, 255)
    
    # Add green rectangle
    frame[rect_top:rect_bottom, rect_left:rect_right, :] = [0, 255, 0]
    
    # Append frame to list
    frames.append(frame)

# Write frames to video
clip = ImageSequenceClip(frames, fps=fps)
clip.write_videofile("random_green_card_video_rect.mp4", fps=fps)

# Remove frames from memory
del frames
