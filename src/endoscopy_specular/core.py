
import numpy as np
import cv2

def detect_specular_reflection(image_bgr: np.ndarray) -> np.ndarray:
    orig_h, orig_w = image_bgr.shape[:2]
    img_resized = cv2.resize(image_bgr, (128, 128))
    img_float = img_resized.astype(np.float32) / 255.0
    img_hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
    _, S, V = img_hsv[..., 0], img_hsv[..., 1], img_hsv[..., 2]
    S_smooth = cv2.GaussianBlur(S, (0, 0), sigmaX=2, sigmaY=2)
    
    Sxx = cv2.Sobel(S_smooth, cv2.CV_32F, dx=2, dy=0, ksize=3) / 16.0
    Syy = cv2.Sobel(S_smooth, cv2.CV_32F, dx=0, dy=2, ksize=3) / 16.0
    Sxy = cv2.Sobel(S_smooth, cv2.CV_32F, dx=1, dy=1, ksize=3) / 16.0
    trace = Sxx + Syy
    disc = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4.0 * Sxy ** 2, 0.0))
    lambda1 = (trace + disc) / 2.0
    lambda2 = (trace - disc) / 2.0
    hessian_gate = (lambda1 >= 0) & (lambda2 >= 0)
    base_mask = hessian_gate & (V > 0.35) & (S < 0.15)
    mask_rescaled = (base_mask.astype(np.uint8) * 255)
    return cv2.resize(mask_rescaled, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)