import numpy as np
from moviepy.editor import ImageSequenceClip

# Define video parameters
resolution = (2160,3840)
fps = 30
duration = 20
num_frames = fps * duration

# Define rectangle parameters
rect_width = 600
rect_height = 400
rect_left = (resolution[0] - rect_width) // 2
# rect_top = (resolution[1] - rect_height) // 2
rect_top = (resolution[1]//2 - rect_height) // 2

rect_dx = 10  # Change in x position per frame
rect_dy = 0  # Change in y position per frame

# Generate frames
frames = []
for i in range(num_frames):
    # Create white background
    frame = np.zeros(resolution + (3,), dtype=np.uint8)
    frame[:] = (255, 255, 255)
    
    # Calculate rectangle position for this frame
    rect_right = rect_left + rect_width
    rect_bottom = rect_top + rect_height
    if rect_left + rect_dx < 0 or rect_right + rect_dx > resolution[0]:
        rect_dx = -rect_dx
    if rect_top + rect_dy < 0 or rect_bottom + rect_dy > resolution[1]:
        rect_dy = -rect_dy
    rect_left += rect_dx
    rect_top += rect_dy
    
    # Add green rectangle
    frame[rect_top:rect_bottom, rect_left:rect_right, :] = [0, 255, 0]
    
    # Append frame to list
    frames.append(frame)

# Write frames to video
clip = ImageSequenceClip(frames, fps=fps)
clip.write_videofile("test_2.mp4", fps=fps)

# Remove frames from memory
del frames
