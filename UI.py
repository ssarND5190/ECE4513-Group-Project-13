import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

# Dot initial position (center of img_tex_show)
dot_radius = 8
dot1_color = (255, 144, 55)
dot1_pos = [255,255]
dragging1 = False

def mouse_callback(event, x, y, flags, param):
    global dot1_pos, dragging1
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the dot
        if (x - dot1_pos[0]) ** 2 + (y - dot1_pos[1]) ** 2 <= dot_radius ** 2:
            dragging1 = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging1:
            # Only allow moving inside the tex image area
            if x >= 0 and x < 480 and y >= 0 and y < 480:
                dot1_pos = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging1 = False

def display_images(img1, img2):
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    combined_image[:img1.shape[0], :img1.shape[1]] = img1
    combined_image[:img2.shape[0], img1.shape[1]:] = img2
    # Draw the dot on the combined image
    cv2.circle(combined_image, (dot1_pos[0], dot1_pos[1]), dot_radius, dot1_color, -1)
    cv2.imshow('window', combined_image)