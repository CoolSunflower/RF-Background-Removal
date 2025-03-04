"""
TODO:
- change instead of sampling everything in the bounding box as foreground maybe only sample towards center?
- Can you provide support for adding a bunch of bounding boxes instead of just one?

- How to make the features better!!!!

"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def remove_background_with_bbox(img: np.ndarray, bboxes: list, color_space: str = 'LAB', max_samples: int = 5000, morph_kernel_size: int = 5) -> np.ndarray:
    """
    Remove background given a bounding box around the foreground object.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (H x W x 3) in BGR or RGB format (as read by cv2 or otherwise).
    bboxes : list
        List of bounding boxes around foreground objects, each as a tuple (x1, y1, x2, y2).
        - (x1, y1) is the top-left corner
        - (x2, y2) is the bottom-right corner
    color_space : str, optional
        Which color space to use for classification features. 
        One of {'RGB', 'LAB', 'HSV'}. Default 'LAB'.
    max_samples : int, optional
        Max number of foreground and background pixels to sample for training.
    morph_kernel_size : int, optional
        Size of the morphological kernel for refining the mask. Default 5.
        
    Returns
    -------
    final_mask : np.ndarray
        A binary mask (H x W) where 1 = foreground, 0 = background.
        
    Notes
    -----
    - This function does the following:
        1. Extracts foreground samples from the bounding box.
        2. Extracts background samples from outside the bounding box.
        3. Converts color space, builds training set (features + labels).
        4. Trains a Random Forest to discriminate FG vs BG.
        5. Predicts for every pixel in the image -> initial mask.
        6. Refines the mask with morphological ops.
        7. Returns the final binary mask (foreground=1, background=0).
    """
    
    # Validate & parse bounding boxes
    H, W, C = img.shape
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: out of image bounds or illogical coordinates.")
    
    # 2. Convert color space if needed
    #    We'll unify everything into a 'features' array later
    if color_space.upper() == 'LAB':
        # Convert BGR -> Lab if using OpenCV's default image read
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif color_space.upper() == 'HSV':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space.upper() == 'RGB':
        # If the input is in BGR, just convert to RGB
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("color_space must be one of {'RGB','LAB','HSV'}")

    # For convenience, let's assume `img_converted` is HxWx3
    H, W, C = img_converted.shape

    # 3. Gather Foreground (FG) samples from inside bounding box
    fg_pixels = []
    for x1, y1, x2, y2 in bboxes:
        fg_pixels.extend(img_converted[y1:y2, x1:x2].reshape(-1, C))
    fg_pixels = np.array(fg_pixels, dtype=np.float32)
    fg_labels = np.ones((fg_pixels.shape[0],), dtype=np.uint8)
    # for row in range(y1, y2):
    #     for col in range(x1, x2):
    #         # (Lab or HSV or RGB values)
    #         fg_pixels.append(img_converted[row, col, :])

    # 4. Gather Background (BG) samples from outside bounding box
    # #    We'll pick some random points outside the box to limit data size
    # bg_pixels_list = []
    # #   - Top region: rows in [0, y1)
    # for row in range(0, y1):
    #     for col in range(0, W):
    #         bg_pixels_list.append(img_converted[row, col, :])
    # #   - Bottom region: rows in [y2, H)
    # for row in range(y2, H):
    #     for col in range(0, W):
    #         bg_pixels_list.append(img_converted[row, col, :])
    # #   - Left region: columns in [0, x1), but rows in [y1, y2] (to avoid double counting corners)
    # for row in range(y1, y2):
    #     for col in range(0, x1):
    #         bg_pixels_list.append(img_converted[row, col, :])
    # #   - Right region: columns in [x2, W), rows in [y1, y2]
    # for row in range(y1, y2):
    #     for col in range(x2, W):
    #         bg_pixels_list.append(img_converted[row, col, :])
    # Gather Background (BG) samples from outside bounding boxes

    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1
    bg_pixels = img_converted[mask == 0].reshape(-1, C)
    bg_labels = np.zeros((bg_pixels.shape[0],), dtype=np.uint8)

    # bg_pixels = np.array(bg_pixels_list, dtype=np.float32)
    # bg_labels = np.zeros((bg_pixels.shape[0],), dtype=np.uint8)  # label = 0 for background

    # 5. Subsample to avoid huge training sets
    if fg_pixels.shape[0] > max_samples:
        idx_fg = np.random.choice(fg_pixels.shape[0], max_samples, replace=False)
        fg_pixels = fg_pixels[idx_fg]
        fg_labels = fg_labels[idx_fg]
    if bg_pixels.shape[0] > max_samples:
        idx_bg = np.random.choice(bg_pixels.shape[0], max_samples, replace=False)
        bg_pixels = bg_pixels[idx_bg]
        bg_labels = bg_labels[idx_bg]

    # Combine FG and BG training samples
    X = np.vstack((fg_pixels, bg_pixels))
    y = np.concatenate((fg_labels, bg_labels))

    # Shuffle to avoid any ordering bias
    X, y = shuffle(X, y, random_state=42)

    # 6. Train a simple RandomForest classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # 7. Predict foreground vs. background for every pixel
    #    We'll reshape the image into Nx3, predict, then reshape back
    flat_img = img_converted.reshape((-1, C)).astype(np.float32)
    pred = rf.predict(flat_img)
    mask = pred.reshape((H, W))  # shape: HxW, values in {0,1}

    # 8. Morphological refinement (closing -> opening, for example)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    # Closing to fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 9. Return the final refined mask (uint8, 1=FG, 0=BG)
    #    The user can apply it to the original image to remove background
    return mask


if __name__ == "__main__":
    # DEMO USAGE
    # Load an example image with OpenCV
    # (Adjust the file path as needed)
    demo_img_path = "example.jpg"
    image_bgr = cv2.imread(demo_img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {demo_img_path}")

    # Suppose we have a bounding box around the object: (x1, y1, x2, y2)
    bounding_box = [(61, 92, 349, 230)]  # example values

    # Call our function
    mask = remove_background_with_bbox(image_bgr, bounding_box)

    # Create a background-removed visualization
    # We'll set BG pixels to white for demonstration
    out_img = image_bgr.copy()
    out_img[mask == 0] = [255, 255, 255]

    # Show results in a notebook or window
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Background Removed")
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
