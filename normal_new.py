import cv2
import numpy as np
import matplotlib.pyplot as plt

def getnormal(img, rotation, len):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.average(gray)
    hight1 = np.copy(gray)
    hight2 = np.zeros_like(gray).astype(np.int32)
    for x in range(img.shape[1]):
        hight2[0,x]=gray[0, x] - average
    for y in range(1, img.shape[0]):
        for x in range(img.shape[1]):
            hight2[y,x]=hight2[y-1, x]+ gray[y, x] - average
    for x in range(img.shape[1]):
        min2 = hight2[:,x].min()
        hight2[:,x]=hight2[:,x]-min2
        max2 = hight2[:,x].max()
        if max2 > 0:
            hight2[:,x] = hight2[:,x] * 255.0 / max2
    hight2 = hight2.astype(np.uint8)
    hight2_blurY = cv2.GaussianBlur(hight2,(1,75),0)
    hight2_blurX = cv2.GaussianBlur(hight2,(95,1),0)
    positive = np.zeros_like(hight2)
    negative = np.zeros_like(hight2)
    output_gray = np.zeros_like(hight2)
    for i in range(hight2.shape[0]):
        for j in range(hight2.shape[1]):
            if hight2[i, j] > hight2_blurY[i, j]:
                positive[i, j] = hight2[i, j] - hight2_blurY[i, j]
            else:
                negative[i, j] = hight2_blurY[i, j] - hight2[i, j]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = int(hight2_blurX[i, j]) + int(positive[i, j]) - int(negative[i, j])
            pixel = np.clip(pixel, 0, 255).astype(np.uint8)
            output_gray[i, j] = pixel

    output = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2BGR)
    return output

