import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

# Dot initial position (center of img_tex_show)
dots_pos = []  # List to store positions of dots
dot_radius = 8
dot_color = [(255, 144, 55), (0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Different colors for different dots

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dots_pos) < 4:
            dots_pos.append([x, y])
        elif len(dots_pos) == 4:  # If already reached max dots, do not add more
            print("You have already marked the maximum number of points.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Clear all points when right-clicked
        dots_pos.clear()

def display_images(img1, img2, img3):
    height = max(img1.shape[0], img2.shape[0], img3.shape[0])
    width = img1.shape[1] + img2.shape[1] + img3.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    combined_image[:img1.shape[0], :img1.shape[1]] = img1
    combined_image[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
    combined_image[:img3.shape[0], img1.shape[1] + img2.shape[1]:] = img3
    # Draw the dot on the combined image
    for i, pos in enumerate(dots_pos):
        cv2.circle(combined_image, (pos[0], pos[1]), dot_radius, dot_color[i % len(dot_color)], -1)
    cv2.imshow('window', combined_image)