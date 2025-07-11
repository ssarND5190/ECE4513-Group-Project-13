import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def extract_by_mortar(image_path, shuffle_top_n=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 灰色砖缝颜色范围（低饱和度），下面两行参数需要调
    lower_gray = np.array([0, 0, 0])
    upper_gray = np.array([25, 150, 240]) 
    mortar_mask = cv2.inRange(hsv, lower_gray, upper_gray) #得到的是砖缝上许多密集的点，没有连成整条砖缝

    # 平滑掩膜
    blurred = cv2.blur(mortar_mask, (7, 7))
    _, cleaned = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY) #平滑再阈值，连成整条的砖缝

    # 提取砖块
    brick_mask = cv2.bitwise_not(cleaned)
    contours, _ = cv2.findContours(brick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bricks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 15:  # 忽略碎块
            bricks.append(((x, y, x + w, y + h), img[y:y + h, x:x + w]))

    # 尺寸聚类
    tolerance = 0.15
    quantized_sizes = []
    grouped = defaultdict(list)

    for rect, b in bricks:
        size = (rect[2] - rect[0], rect[3] - rect[1])
        matched = False
        for i, q in enumerate(quantized_sizes):
            if (abs(size[0] - q[0]) / q[0] < tolerance and 
                abs(size[1] - q[1]) / q[1] < tolerance):
                grouped[i].append((rect, b))
                matched = True
                break
        if not matched:
            quantized_sizes.append(size)
            grouped[len(quantized_sizes) - 1].append((rect, b))

    # 按砖块数量降序排序
    sorted_groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 打乱并重建
    rearranged = img.copy()
    for idx, (group_id, group) in enumerate(sorted_groups):
        if shuffle_top_n is not None and idx >= shuffle_top_n:
            continue  # 跳过非前n类
        patches = [b for _, b in group]
        random.shuffle(patches)
        for i, (rect, _) in enumerate(group):
            x1, y1, x2, y2 = rect
            patch = patches[i % len(patches)]  # 循环使用以防数量不匹配
            try:
                h, w = patch.shape[:2]
                rearranged[y1:y1 + h, x1:x1 + w] = patch
            except Exception as e:
                print(f"Warning: Patch insertion failed at {rect}: {e}")

    return img, cleaned, bricks, rearranged

# 使用
image_path = 'regular.jpg'
shuffle_top_n = 1 #这里如果去掉，可以自适应，但效果可能不好（需打乱的砖块种类上限）
original_img, mortar_mask_img, bricks, shuffled_img = extract_by_mortar(
    image_path, shuffle_top_n)

# 显示结果（与原代码相同）
debug_img = original_img.copy()
for rect, _ in bricks:
    x1, y1, x2, y2 = rect
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Original Wall")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Bricks")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(shuffled_img, cv2.COLOR_BGR2RGB))
plt.title("Rearranged Wall")
plt.axis("off")

plt.tight_layout()
plt.show()