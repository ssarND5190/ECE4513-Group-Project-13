import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

def detect_bricks(image_path): ######这是放弃的直线检测方案，对竖直边缘的检测很麻烦，因为竖直边沿往往不够长
    # 读取图像               
    img = cv2.imread(image_path)  ####而且就算正确找到了边缘，对于有宽度的砖缝，分割也是问题：到底哪块是砖缝，哪块是砖块？
    if img is None:
        print("无法加载图像，请检查路径")
        return None
    #img = cv2.GaussianBlur(img, (5, 5, ), 0) #先进行模糊，虽然能过滤砖缝内部细节，但也让边缘不那么清晰了
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny 边缘检测
    # 黑帽突出缝隙区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 加权增强缝隙
    enhanced = cv2.add(gray, blackhat)

    # Canny 边缘检测
    edges = cv2.Canny(enhanced, 50, 150)

    # 闭运算连接边缘
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    # -----------------------
    # 第一次：检测水平线段
    # -----------------------
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                              minLineLength=60, maxLineGap=15)

    # -----------------------
    # 第二次：检测垂直线段
    # -----------------------
    lines_v = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                              minLineLength=30, maxLineGap=10)

    # 可视化图像
    line_img = img.copy()
    
    # 水平/垂直线列表
    horizontal = []
    vertical = []

    # 处理水平线
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180
            if abs(angle) < 5 or abs(angle - 180) < 5:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                horizontal.append((min(y1, y2), max(y1, y2)))

    # 处理垂直线
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180
            if abs(angle - 90) < 5:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                vertical.append((min(x1, x2), max(x1, x2)))

    # 排序+去重
    horizontal = sorted(list(set(horizontal)), key=lambda x: x[0])
    vertical = sorted(list(set(vertical)), key=lambda x: x[0])

    # 合并相近水平线
    merged_horizontal = []
    for h in horizontal:
        if not merged_horizontal:
            merged_horizontal.append(h)
        else:
            last = merged_horizontal[-1]
            if h[0] - last[1] < 15:
                merged_horizontal[-1] = (last[0], h[1])
            else:
                merged_horizontal.append(h)

    # 合并相近垂直线
    merged_vertical = []
    for v in vertical:
        if not merged_vertical:
            merged_vertical.append(v)
        else:
            last = merged_vertical[-1]
            if v[0] - last[1] < 15:
                merged_vertical[-1] = (last[0], v[1])
            else:
                merged_vertical.append(v)

    # 提取砖块区域
    bricks = []
    brick_sizes = []

    for i in range(len(merged_horizontal) - 1):
        for j in range(len(merged_vertical) - 1):
            y1 = merged_horizontal[i][0]
            y2 = merged_horizontal[i+1][1]
            x1 = merged_vertical[j][0]
            x2 = merged_vertical[j+1][1]
            
            if y2 > y1 and x2 > x1:
                brick = img[y1:y2, x1:x2]
                bricks.append(((x1, y1, x2, y2), brick))
                brick_sizes.append((x2-x1, y2-y1))

    # 量化砖块尺寸
    quantized_sizes = []
    size_groups = {}
    tolerance = 0.1  # 10%

    for size in brick_sizes:
        matched = False
        for q_size in quantized_sizes:
            if (abs(size[0] - q_size[0]) / q_size[0] < tolerance and 
                abs(size[1] - q_size[1]) / q_size[1] < tolerance):
                matched = True
                break
        if not matched:
            quantized_sizes.append(size)

    # 分组砖块
    brick_groups = {i: [] for i in range(len(quantized_sizes))}

    for idx, ((x1, y1, x2, y2), brick) in enumerate(bricks):
        size = (x2-x1, y2-y1)
        for i, q_size in enumerate(quantized_sizes):
            if (abs(size[0] - q_size[0]) / q_size[0] < tolerance and 
                abs(size[1] - q_size[1]) / q_size[1] < tolerance):
                brick_groups[i].append(((x1, y1, x2, y2), brick))
                break

    return img, line_img, brick_groups, merged_horizontal, merged_vertical


def rearrange_bricks(img, brick_groups, horizontal_lines, vertical_lines):
    # 创建新图像
    new_img = img.copy()
    
    # 对每组砖块进行重排
    for group in brick_groups.values():
        # 提取所有砖块
        bricks = [brick for (rect, brick) in group]
        # 随机打乱顺序
        random.shuffle(bricks)
        
        # 将打乱后的砖块放回原位
        for idx, (rect, _) in enumerate(group):
            x1, y1, x2, y2 = rect
            new_img[y1:y2, x1:x2] = cv2.resize(bricks[idx], (x2 - x1, y2 - y1))

    
    # 重新绘制网格线
    for i in range(len(horizontal_lines) - 1):
        y = horizontal_lines[i][1]
        cv2.line(new_img, (0, y), (new_img.shape[1], y), (0, 0, 0), 2)
    
    for i in range(len(vertical_lines) - 1):
        x = vertical_lines[i][1]
        cv2.line(new_img, (x, 0), (x, new_img.shape[0]), (0, 0, 0), 2)
    
    return new_img

def main():
    image_path = 'wall.jpg'
    
    # 检测砖块
    original_img, line_img, brick_groups, horizontal_lines, vertical_lines = detect_bricks(image_path)
    
    if original_img is None:
        return
    
    # 重排砖块
    rearranged_img = rearrange_bricks(original_img, brick_groups, horizontal_lines, vertical_lines)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title('直线检测结果(绿色:水平,红色:垂直)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(rearranged_img, cv2.COLOR_BGR2RGB))
    plt.title('重排后的砖墙')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    cv2.imwrite('detected_lines.jpg', line_img)
    cv2.imwrite('rearranged_wall.jpg', rearranged_img)

if __name__ == "__main__":
    main()