import cv2
import numpy as np
import matplotlib.pyplot as plt
import UI

cv2.namedWindow('window')
# Read Source Image
img_src = cv2.imread('img.png')
src_x = img_src.shape[1]
src_y = img_src.shape[0]
src_zoom = 480.0 / src_x
print(src_x,src_zoom,src_y)
img_src_show = cv2.resize(img_src, (int(src_x * src_zoom), int(src_y * src_zoom)), interpolation=cv2.INTER_CUBIC)
# Texture Image
tex_x = 512
tex_y = 512
img_tex = np.zeros((tex_x,tex_y,3),np.uint8)
tex_zoom = 480.0 / tex_x
img_tex_show = cv2.resize(img_tex, (int(tex_x * tex_zoom), int(tex_y * tex_zoom)), interpolation=cv2.INTER_CUBIC)

canProcess = False

cv2.setMouseCallback('window', UI.mouse_callback)

cv2.createTrackbar('bar1','window',0,255,UI.nothing)
cv2.createTrackbar('bar2','window',0,255,UI.nothing)
cv2.createTrackbar('bar3','window',0,255,UI.nothing)

def processTex():
    r = cv2.getTrackbarPos('bar1','window')
    g = cv2.getTrackbarPos('bar2','window')
    b = cv2.getTrackbarPos('bar3','window')
    img_tex[:] = [b,g,r]
    pass

while True:
    if canProcess:
        processTex()
        print(UI.dot1_pos)
        canProcess = False
    img_tex_show = cv2.resize(img_tex, (int(tex_x * tex_zoom), int(tex_y * tex_zoom)), interpolation=cv2.INTER_CUBIC)
    UI.display_images(img_src_show, img_tex_show)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # key [Esc]
        break
    elif key == 32:  # key [Space]
        canProcess = True

cv2.destroyAllWindows()

