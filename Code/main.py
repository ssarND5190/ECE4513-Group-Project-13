import cv2
import numpy as np
import matplotlib.pyplot as plt
import UI
import normal
import filter
import normal_LGI

cv2.namedWindow('ECE4513-TexProcess')
# Source Image
img_src = cv2.imread('./img.png')
src_x = img_src.shape[1]
src_y = img_src.shape[0]
src_zoom = 360.0 / src_x
print("img info:",src_x,src_zoom,src_y) #调试信息
print("Please select four points outlining the area you want to process.\nThey will be transformed into a square.\n")
print("----Press the Space Bar to Start Process----")
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

cv2.setMouseCallback('ECE4513-TexProcess', UI.mouse_callback)

# These bars are not used currently
cv2.createTrackbar('Radius','ECE4513-TexProcess',0,255,UI.nothing)
cv2.createTrackbar('L(*0.01)','ECE4513-TexProcess',0,100,UI.nothing)
cv2.createTrackbar('angle','ECE4513-TexProcess',0,359,UI.nothing)

def processImage():
    global img_src, img_tex, img_normal, tex_zoom
    value1 = cv2.getTrackbarPos('Radius','ECE4513-TexProcess')

    if len(UI.dots_pos) == 4:

        print("----Processing image. Please Wait----")

        # 将UI.dots_pos中的点从展示图像坐标反向映射回原图坐标
        pts_src = np.array([[int(pos[0] / src_zoom), int(pos[1] / src_zoom)] for pos in UI.dots_pos], dtype=np.float32)
        center = np.mean(pts_src, axis=0)
        angles = np.arctan2(pts_src[:, 1] - center[1], pts_src[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        pts_src = pts_src[sorted_indices] # 计算每个点相对于重心的角度，并据此排序，这样不拘泥于输入顺序

        print("----Start: Perspective transform")

        # 定义目标区域为512x512的矩形，原图中去做变换，精度更高
        pts_dst = np.array([
            [0, 0],
            [512, 0],
            [512, 512],
            [0, 512]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # 应用透视变换到原始图像上，输出大小设置为512x512
        warped = cv2.warpPerspective(img_src, M, (512, 512))

        print("----Finished: Perspective transform")

        # 把视角转换后的纹理用filer进行光照处理
        img_tex = filter.filterSim(warped, value1)


        # 内部纹理处理时，我们用的是512像素，输出前要压缩一下，这里计算压缩率
        tex_zoom = 360.0 / 512

        # 法线生成
        img_normal = normal_LGI.getNormal(img_tex, UI.sun_pos[1], UI.sun_pos[0]/100)
    else:
        print("Please select exactly four points on the source image.")


while True:
    if cv2.getWindowProperty('ECE4513-TexProcess', cv2.WND_PROP_VISIBLE) < 1:
        break
    if canProcess:
        processImage()
        print("Selected points:")
        for i, pos in enumerate(UI.dots_pos):
            print(f"  Point {i+1}: {pos}")  #打印点的坐标
        cv2.imwrite("output_tex.png",img_tex)
        cv2.imwrite("output_nm.png",img_normal)
        canProcess = False
    value2 = cv2.getTrackbarPos('L(*0.01)','ECE4513-TexProcess')
    value3 = cv2.getTrackbarPos('angle','ECE4513-TexProcess')
    UI.sun_pos[0]=value2
    UI.sun_pos[1]=value3* np.pi / 180
    img_tex_show = cv2.resize(img_tex, (int(tex_x * tex_zoom), int(tex_y * tex_zoom)), interpolation=cv2.INTER_CUBIC)
    img_normal_show = cv2.resize(img_normal, (int(normal_x * normal_zoom), int(normal_y * normal_zoom)), interpolation=cv2.INTER_CUBIC)
    UI.display_images(img_src_show, img_tex_show, img_normal_show)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # [Esc] key
        break
    elif key == 32:
        if len(UI.dots_pos) == 4:  # [Space] key 按下，且选择了四个点
            canProcess = True
        else:
            print("Please select exactly four points on the source image.")
    

cv2.destroyAllWindows()