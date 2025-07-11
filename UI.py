import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

# Dot initial position (center of img_tex_show)
dots_pos = []  # List to store positions of dots
dot_radius = 5
dot_color = [(255, 144, 55), (0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Different colors for different dots

#光源位置 (极坐标 r, theta)
sun_pos = [0.0,0.0]

def mouse_callback(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 0 or x >= 360: #只允许在第一张图的宽度360内点击
            print("You can only select points on the left image.")
            return

        # 检查点数量
        if len(dots_pos) < 3:
            dots_pos.append([x, y])
        elif len(dots_pos) == 3:
            temp_dots = dots_pos + [[x, y]]
            # 构建一个临时点列表，检查所有4个点是否都合法（即不在其它三点构成的三角形内或三角形的边上）
            valid = True
            for i in range(4): #每个点都要查
                others = [tuple(pt) for j, pt in enumerate(temp_dots) if j != i]
                point = tuple(temp_dots[i])
                triangle = np.array(others, dtype=np.float32)

                is_inside = cv2.pointPolygonTest(triangle, point, False)
                if is_inside >= 0: #点在形内/形上
                    print(f"Error: Point {i+1} is inside the triangle formed by the other three.")
                    valid = False
                    break
            if valid:
                dots_pos.append([x, y])
            else:
                print("The selected points are invalid. You can reselect the last point or right-click to reset all points.")
        else:
            print("You have already selected four points. Press the space bar to start image processing. Right-click to reset.")
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
    cv2.line(combined_image, (180,180), (int(180+sun_pos[0]*np.cos(sun_pos[1])), int(180-sun_pos[0]*np.sin(sun_pos[1]))), (128,255,255))
    cv2.circle(combined_image, (int(180+sun_pos[0]*np.cos(sun_pos[1])), int(180-sun_pos[0]*np.sin(sun_pos[1]))), 8, (128,255,255), -1)
    cv2.imshow('window', combined_image)