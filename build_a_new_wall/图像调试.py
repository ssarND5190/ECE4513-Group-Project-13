import cv2
import numpy as np
def click_and_show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y, x]
        hsv_val = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked at ({x}, {y}) - BGR: {bgr}, HSV: {hsv_val}")

img = cv2.imread('wlll.png')
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_and_show_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
