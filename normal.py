import cv2
import numpy as np
import matplotlib.pyplot as plt

def High_Pass_Filter(img,radius = 100):
    # get the size
    rows, cols = img.shape
    # fft
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # High Pass Filter
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    # Apply the filter
    dft_shift_filtered = dft_shift * mask

    # IFFT
    f_ishift = np.fft.ifftshift(dft_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def generate_height_map(img):
    
    # 后面一定要上深度学习，现在效果实在是太拉垮了

    # transfer to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # map to 0 - 255
    height_map = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 转为uint8
    height_map = height_map.astype(np.uint8)
    return height_map

def generate_normal_map(img, strength = 3.0):

    if img is None:
        raise ValueError("Cannot load image: ")
    
    img = img.astype(np.float32) / 255.0

    # get gradient
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # apply strength
    sobel_x *= strength
    sobel_y *= strength

    normal_x = sobel_x
    normal_y = sobel_y
    normal_z = np.ones_like(img)
    normal_z = np.sqrt(np.clip(1.0 - normal_x**2 - normal_y**2, 0, 1))
    normals = np.stack((normal_x, normal_y, normal_z), axis=2)

    # normalize
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norms

    # map [-1,1] to [0,255]
    normals = (normals + 1.0) * 0.5 * 255.0
    normals = normals.astype(np.uint8)

    normal_map = cv2.merge((normals[:,:,2], normals[:,:,1], normals[:,:,0]))
    return normal_map

def getNormal(img):
    img = generate_normal_map(img)
    return img

# if __name__ == "__main__":
#     img = cv2.imread('/Users/sarahlu/Desktop/CVData/brick2_original.png', cv2.IMREAD_GRAYSCALE)
#     cv2.imshow("Low Pass", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     img = High_Pass_Filter(img)
#     # height_map = generate_height_map(img)
#     cv2.imshow("Low Pass", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     normal_map = generate_normal_map(img, strength=2.0)
#     cv2.imshow("normal map", normal_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

