import cv2
import numpy as np
import matplotlib.pyplot as plt
import UI
import normal
import filter

cv2.namedWindow('window')
# Source Image
img_src = cv2.imread('./img.png')
src_x = img_src.shape[1]
src_y = img_src.shape[0]
src_zoom = 360.0 / src_x
print(src_x,src_zoom,src_y)
img_src_show = cv2.resize(img_src, (int(src_x * src_zoom), int(src_y * src_zoom)), interpolation=cv2.INTER_CUBIC)
# Texture Image
tex_x = 512
tex_y = 512
img_tex = np.zeros((tex_x,tex_y,3),np.uint8)
tex_zoom = 360.0 / tex_x
img_tex_show = cv2.resize(img_tex, (int(tex_x * tex_zoom), int(tex_y * tex_zoom)), interpolation=cv2.INTER_CUBIC)
# Normal Map
normal_x = 512
normal_y = 512
img_normal = np.zeros((normal_x,normal_y,3),np.uint8)
normal_zoom = 360.0 / normal_x
img_normal_show = cv2.resize(img_normal, (int(normal_x * normal_zoom), int(normal_y * normal_zoom)), interpolation=cv2.INTER_CUBIC)

canProcess = False

cv2.setMouseCallback('window', UI.mouse_callback)

# These bars are not used currently
cv2.createTrackbar('bar1','window',0,255,UI.nothing)
cv2.createTrackbar('bar2','window',0,255,UI.nothing)
cv2.createTrackbar('bar3','window',0,255,UI.nothing)

def processImage():
    global img_src, img_tex, img_normal
    # These values are not used currently
    value1 = cv2.getTrackbarPos('bar1','window')
    value2 = cv2.getTrackbarPos('bar2','window')
    value3 = cv2.getTrackbarPos('bar3','window')
    # Do perspective Transform Here
    img_tex = filter.filter1(img_src)
    # 理论上normal这里传入的参数是img_tex，但是texureFilter尚未完工，所以暂时使用img_src
    img_normal = normal.getNormal(img_src)
    pass

while True:
    if canProcess:
        processImage()
        print(UI.dot1_pos)
        canProcess = False
    img_tex_show = cv2.resize(img_tex, (int(tex_x * tex_zoom), int(tex_y * tex_zoom)), interpolation=cv2.INTER_CUBIC)
    img_normal_show = cv2.resize(img_normal, (int(normal_x * normal_zoom), int(normal_y * normal_zoom)), interpolation=cv2.INTER_CUBIC)
    UI.display_images(img_src_show, img_tex_show, img_normal_show)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # key [Esc]
        break
    elif key == 32:  # key [Space]
        canProcess = True

cv2.destroyAllWindows()

