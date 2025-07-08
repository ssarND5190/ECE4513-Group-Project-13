import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import cv2

def lambertian_error(grad_flat, I, light_dir, lambda_smooth=0.1):
    """
    Energy = data term + smoothness term
    grad_flat: [p,q,p,q,...]
    """
    h, w = I.shape
    assert grad_flat.shape[0] == h * w * 2, f"Expected shape {(h*w*2,)} but got {grad_flat.shape}"
    print(f"[DEBUG] lambertian_error: grad_flat.shape={grad_flat.shape}, expected={h*w*2}")
    grad = grad_flat.reshape((h, w, 2))
    p = grad[...,0]
    q = grad[...,1]

    # Recompute normal vector
    norm = np.sqrt(p**2 + q**2 + 1)
    n = np.stack([-p/norm, -q/norm, 1/norm], axis=2)

    # Predicted intensity
    pred_I = np.sum(n * light_dir.reshape(1,1,3), axis=2)

    # Data term: difference to observed image
    data = (pred_I - I)**2
    data_term = np.sum(data)

    # Smoothness term: encourage smooth gradients
    dp_dx = np.diff(p, axis=1)
    dp_dy = np.diff(p, axis=0)
    dq_dx = np.diff(q, axis=1)
    dq_dy = np.diff(q, axis=0)
    smooth = np.sum(dp_dx**2) + np.sum(dp_dy**2) + np.sum(dq_dx**2) + np.sum(dq_dy**2)

    return data_term + lambda_smooth * smooth

def optimize_gradient(I, light_dir, lambda_smooth=0.1, maxiter=50):
    h, w = I.shape
    x0 = np.zeros((h * w * 2,), dtype=np.float64)

    def lambertian_error_checked(grad_flat, I, light_dir, lambda_smooth=0.1):
        h, w = I.shape
        assert grad_flat.size == h * w * 2, f"grad_flat size {grad_flat.size} does not match expected {h*w*2}"
        return lambertian_error(grad_flat, I, light_dir, lambda_smooth)

    res = minimize(lambertian_error_checked, x0, args=(I, light_dir, lambda_smooth), method="CG", options={'maxiter':maxiter})
    grad = res.x.reshape((h,w,2))
    # reconstruct normal 
    # grad: shape (h, w, 2)
    p = grad[..., 0]
    q = grad[..., 1]
    norm = np.sqrt(p**2 + q**2 + 1e-8)  # 防止除以0
    n = np.stack([-p/norm, -q/norm, 1/norm], axis=2)  # shape: (h, w, 3)

    # transfer normal to normal map
    normals = (n + 1.0) * 0.5 * 255.0
    normals = normals.astype(np.uint8)
    normal_map = cv2.merge((normals[:,:,2], normals[:,:,1], normals[:,:,0]))
    return normal_map

def getNormal_sfs(img,light_dir = [0,0,1]):
    '''
    img: color image with the shape (h,w,3)
    return: normal map with the shape (h,w,3)
    '''
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32) / 255.0
    # Resize image to reduce computation for visualization and optimization
    # But it works really slow...
    scale_factor = 0.1
    # take a small size for debug
    I_small = cv2.resize(I, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    h, w = I_small.shape
    print(f"Loaded image size (resized): {h} x {w}")
    light_dir = np.array(light_dir, dtype=np.float32)
    normal = optimize_gradient(I_small, light_dir, lambda_smooth=0.1, maxiter=10)
    return normal

if __name__ == "__main__":
    img = cv2.imread('/Users/sarahlu/Desktop/CVData/sfs_ball.jpg')
    normal = getNormal_sfs(img)
    cv2.imshow("normal map", normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


