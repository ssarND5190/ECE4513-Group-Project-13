import cv2
import numpy as np
import matplotlib.pyplot as plt
import normal

#一种结合高度直接生成和沿光源法向生成的方法

def getNormal(img, rotation, len):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    average = np.average(gray)
    #由亮度确定光源方向的法线强度
    normalL = np.zeros_like(img)
    normalL[:]= [255, 128, 128]
    cosr=np.cos(rotation)
    sinr=np.sin(rotation)
    for x in range(img.shape[1]):
        for y in range(1, img.shape[0]):
            dn = int(gray[y,x]) - int(average)
            dx = 128 + dn * cosr
            dy = 128 + dn * sinr
            dx = np.clip(dx, 0, 255).astype(np.uint8)
            dy = np.clip(dy, 0, 255).astype(np.uint8)
            normalL[y,x,1]=dy
            normalL[y,x,2]=dx
    #亮度转高度的法线
    normalH = normal.getNormal(img)
    #融合，保留垂直于光源方向的法线
    for x in range(img.shape[1]):
        for y in range(1, img.shape[0]):
            g=float(normalH[y,x,1])-128.0
            r=float(normalH[y,x,2])-128.0
            dTr=sinr*sinr*r-cosr*sinr*g
            dTg=cosr*cosr*g-cosr*sinr*r
            dlr=cosr*cosr*r+cosr*sinr*g
            dlg=cosr*sinr*r+sinr*sinr*g
            lr=len*(float(normalL[y,x,1])-128.0)
            lg=len*(float(normalL[y,x,2])-128.0)
            nmg = 128+(1.0-len)*dlg + lr + dTg
            nmr = 128+(1.0-len)*dlr + lg + dTr
            normalH[y,x,1]=np.clip(nmg,0,255).astype(np.uint8)
            normalH[y,x,2]=np.clip(nmr,0,255).astype(np.uint8)
    return normalH

#一种特殊的高度推算方法，在单光源平行于平面时有较好效果，但是非常容易受干扰（如垂直于光源的条纹）
#目前光源方向暂定为正上方，还不能调节

def getnormal(img, rotation, len):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #亮度转高度图
    hight1 = np.copy(gray)
    #亮度累积高度图
    hight2 = np.zeros_like(gray).astype(np.int32)
    for x in range(img.shape[1]):
        average = np.average(gray[:,x])
        for y in range(1, img.shape[0]):
            hight2[y,x]=hight2[y-1, x]+ gray[y, x] - average
    for x in range(img.shape[1]):
        min2 = hight2[:,x].min()
        hight2[:,x]=hight2[:,x]-min2
        max2 = hight2[:,x].max()
        if max2 > 0:
            hight2[:,x] = hight2[:,x] * 255.0 / max2
    hight2 = hight2.astype(np.uint8)
    #径向模糊以平滑结果
    hight2_blurY = cv2.GaussianBlur(hight2,(1,25),0)
    hight2_blurX = cv2.GaussianBlur(hight2,(55,1),0)
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
    output_gray=cv2.equalizeHist(output_gray)
    output = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2BGR)
    return output

def rotate_image(img, angle):
    # Get image size
    (h, w) = img.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calculate the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated