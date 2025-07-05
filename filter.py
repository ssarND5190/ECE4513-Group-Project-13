import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter1(img, radius):
    output = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_gray = np.zeros_like(gray)
    blured = cv2.GaussianBlur(gray, (radius*2+1, radius*2+1), 0)
    positive = np.zeros_like(gray)
    negative = np.zeros_like(gray)
    average = 0.0
    total_pixels = img.shape[0] * img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            average += img[i, j][0] / total_pixels
            if gray[i, j] > blured[i, j]:
                positive[i, j] = gray[i, j] - blured[i, j]
            else:
                negative[i, j] = blured[i, j] - gray[i, j]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = average+positive[i, j]-negative[i, j] 
            if pixel > 0 and pixel < 255:
                output_gray[i, j] = pixel
            else:
                output_gray[i, j] = 255
    output= cv2.cvtColor(output_gray, cv2.COLOR_GRAY2BGR)
    print(average)
    #process Image here
    return output

def filter2(img, radius):
    output = np.zeros_like(img)
    blured = cv2.medianBlur(img, radius*2+1, 0)
    positive = np.zeros_like(img)
    negative = np.zeros_like(img)
    average = [0.0, 0.0, 0.0]
    total_pixels = img.shape[0] * img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                average[k] += img[i, j][k] / total_pixels
                if img[i, j][k] > blured[i, j][k]:
                    positive[i, j][k] = img[i, j][k] - blured[i, j][k]
                else:
                    negative[i, j][k] = blured[i, j][k] - img[i, j][k]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                pixel = average[k] + positive[i, j][k] - negative[i, j][k]
                if pixel > 0 and pixel < 255:
                    output[i, j][k] = pixel
                else:
                    output[i, j][k] = 255
    print(average)
    #process Image here
    return output

def filter3(img, radius):
    output = np.zeros_like(img)
    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.ximgproc.guidedFilter(guide, img, 8, 50, dDepth=-1)
    positive = np.zeros_like(img)
    negative = np.zeros_like(img)
    average = [0.0, 0.0, 0.0]
    total_pixels = img.shape[0] * img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                average[k] += img[i, j][k] / total_pixels
                if img[i, j][k] > blured[i, j][k]:
                    positive[i, j][k] = img[i, j][k] - blured[i, j][k]
                else:
                    negative[i, j][k] = blured[i, j][k] - img[i, j][k]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                pixel = average[k] + positive[i, j][k] - negative[i, j][k]
                if pixel > 0 and pixel < 255:
                    output[i, j][k] = pixel
                else:
                    output[i, j][k] = 255
    print(average)
    #process Image here
    return blured

# Optimized version of filter4 using vectorized operations and efficient filtering
def filter4(img, gs_radius):
    median_radius=20
    output = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (gs_radius*2+1, gs_radius*2+1), 0)
    # Use a fast median filter with a mask based on gaussian similarity
    blured = np.zeros_like(gray)
    kernel_size = median_radius * 2 + 1
    padded_gray = np.pad(gray, median_radius, mode='reflect')
    padded_gaussian = np.pad(gaussian, median_radius, mode='reflect')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            g_center = padded_gaussian[i+median_radius, j+median_radius]
            window_g = padded_gaussian[i:i+kernel_size, j:j+kernel_size]
            window = padded_gray[i:i+kernel_size, j:j+kernel_size]
            mask = np.abs(window_g - g_center) < 2
            if np.any(mask):
                blured[i, j] = np.median(window[mask])
            else:
                blured[i, j] = gray[i, j]
    print("DONE")
    positive = np.zeros_like(gray)
    negative = np.zeros_like(gray)
    average = 0.0
    total_pixels = gray.shape[0] * gray.shape[1]
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            average += gray[i, j] / total_pixels
            if gray[i, j] > blured[i, j]:
                positive[i, j] = gray[i, j] - blured[i, j]
            else:
                negative[i, j] = blured[i, j] - gray[i, j]
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            pixel = average + positive[i, j] - negative[i, j]
            if pixel > 0 and pixel < 255:
                blured[i, j] = pixel
            else:
                blured[i, j] = 255
    output = cv2.cvtColor(blured, cv2.COLOR_GRAY2BGR)
    return output
