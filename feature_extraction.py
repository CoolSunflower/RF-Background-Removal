import cv2
import numpy as np

# QUEST feature extraction function
def quest_feature_extraction(image, window_size=5):
    H, W = image.shape
    # Compute image gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute elements of the structure tensor
    Ixx = gx * gx
    Ixy = gx * gy
    Iyy = gy * gy

    # Apply a Gaussian filter to average the tensor components
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)

    # Compute the eigenvalues of the structure tensor
    lambda1 = 0.5 * (Sxx + Syy + np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))
    lambda2 = 0.5 * (Sxx + Syy - np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))

    # Ensure the output shape matches H * W
    quest_features = np.concatenate([lambda1, lambda2]).reshape(H*W, 2)
    # quest_features = np.mean([lambda1, lambda2], axis=0).reshape(H * W, 1)
    print(f"Quest Features shape: {quest_features.shape}")
    return quest_features
