import cv2
import numpy as np
import matplotlib.pyplot as plt

def getNormal(img):
    output = np.zeros_like(img)
    output[:]= [255, 128, 128]  # Set a default normal map color
    #process Normal here
    return output