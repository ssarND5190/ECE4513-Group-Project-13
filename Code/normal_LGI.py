import cv2
import numpy as np
import matplotlib.pyplot as plt
import normal

#一种结合高度直接生成和沿光源法向生成的方法

def getNormal(img, rotation, len):
    print("----Start: Normal estimation")
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
            dx = 128 - dn * cosr
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
    print("----Finished: Normal estimation")
    normalH[:,:,0] = 255
    return normalH

