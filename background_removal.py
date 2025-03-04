"""
TODO:
- Add more sampling towards center!!
- The image width in streamlit app is constrained by the maximum width of the column instead of actual image width 
    (we cant use actual since will cause problems for larger images, but need to ensure if scaling that the positions are also scaled when passing to function call)
- More user options on the UI, including which features to use
- Add higher weightage to classifying points to bg class --> this will ensure point in fg which look like bg are also correctly classified!
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
import streamlit as st
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from feature_extraction import quest_feature_extraction

def remove_background_with_bbox(img: np.ndarray, bboxes: list, color_space: str = 'LAB', max_samples: int = 5000, morph_kernel_size: int = 5, features: list = [], max_hp_tuning_iter: int = 50) -> np.ndarray:
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
    features : list, optional
        List of additional features to include, such as {'lbp', 'ltp', 'quest', 'hog'}.        
    Returns
    -------
    final_mask : np.ndarray
        A binary mask (H x W) where 1 = foreground, 0 = background.
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
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif color_space.upper() == 'HSV':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space.upper() == 'RGB':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("color_space must be one of {'RGB','LAB','HSV'}")

    # For convenience, let's assume `img_converted` is HxWx3
    H, W, C = img_converted.shape

    # 3. Gather Foreground (FG) samples from inside bounding box
    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1

    # Combine FG and BG training samples
    X = img_converted.reshape(-1, C)  # Basic pixel features
    y = mask.reshape(-1)

    additional_features = []
    if 'lbp' in features:
        gray_img = rgb2gray(img_converted)
        lbp_features = local_binary_pattern(gray_img, P=8, R=1, method='uniform').reshape(-1, 1)
        print(f"LBP Features shape: {lbp_features.shape}")
        additional_features.append(lbp_features)
    if 'ltp' in features:
        # Placeholder for LTP (Local Ternary Pattern) extraction
        ltp_features = np.where(gray_img > 0.5, 1, -1).reshape(-1, 1)
        print(f"LTP Features shape: {ltp_features.shape}")
        additional_features.append(ltp_features)
    if 'quest' in features:
        # QUEST (Quantitative Evaluation of Texture) feature extraction
        # quest_features = np.mean(img_converted, axis=2).reshape(-1, 1)
        # print(f"Quest Features shape: {quest_features.shape}")
        # additional_features.append(quest_features)
        gray_img = cv2.cvtColor(img_converted, cv2.COLOR_RGB2GRAY)
        quest_features = quest_feature_extraction(gray_img)
        additional_features.append(quest_features)
    if 'hog' in features:
        gray_img = rgb2gray(img_converted)
        hog_features = hog(
            gray_img, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            block_norm='L2-Hys', 
            feature_vector=True
        )
        
        expected_size = H * W
        if len(hog_features) < expected_size:
            # Repeat features to match size
            hog_features = np.tile(hog_features, (expected_size // len(hog_features) + 1))[:expected_size]
        elif len(hog_features) > expected_size:
            # Trim features if too many
            hog_features = hog_features[:expected_size]
        
        hog_features = hog_features.reshape(-1, 1)
        print(f"HOG Features shape: {hog_features.shape}")
        additional_features.append(hog_features)

    if additional_features:
        X = np.hstack([X] + additional_features)

    # Shuffle to avoid any ordering bias
    X, y = shuffle(X, y, random_state=42)

    if max_samples < len(y[y == 0]):
        bg_idx = np.random.choice(np.where(y == 0)[0], max_samples, replace=False)
    else:
        bg_idx = np.where(y == 0)[0]
    if max_samples < len(y[y == 1]):
        fg_idx = np.random.choice(np.where(y == 1)[0], max_samples, replace=False)
    else:
        fg_idx = np.where(y == 1)[0]

    idx = np.concatenate([bg_idx, fg_idx])
    X, y = X[idx], y[idx]

    max_iter = max_hp_tuning_iter

    # 6. Train a simple RandomForest classifier
    def hyperopt_objective(params, X, y, status_widget):        
        """Objective function for Hyperopt to minimize."""
        model = RandomForestClassifier(**params, random_state=42)
        accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

        # Increment a global counter stored in session_state
        st.session_state.iteration += 1
        iteration = st.session_state.iteration
        st.session_state.best_accuracy = max(accuracy, st.session_state.best_accuracy)
        best_accuracy = st.session_state.best_accuracy

        # Update the status widget with the current iteration and accuracy
        status_widget.text(f"Iteration {iteration}/{max_iter}: Accuracy = {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
    
        return {'loss': -accuracy, 'status': STATUS_OK}

    search_space = {
        'n_estimators': hp.choice('n_estimators', list(range(50, 201))),
        'max_depth': hp.choice('max_depth', list(range(1, 51))),
        'min_samples_split': hp.choice('min_samples_split', list(range(3, 11))),
        'bootstrap': hp.choice('bootstrap', [True, False])
    }

    # Run Hyperopt optimization
    trials = Trials()
    status_text = st.empty()
    best_params = fmin(
        fn=lambda params: hyperopt_objective(params, X, y, status_text),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_iter,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Convert best params to integers where applicable
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['bootstrap'] = bool(best_params['bootstrap'])

    print("Best Hyperparameters found by Hyperopt:")
    print(best_params)

    st.success("Hyperparameter tuning completed!")
    st.write("Best Hyperparameters:", best_params)

    # Train the RandomForest model with the best hyperparameters
    rf = RandomForestClassifier(**best_params, random_state=42)
    rf.fit(X, y)

    # Print the accuracy score on the training data
    print(f"Random Forest Training Accuracy: {rf.score(X, y):.4f}")
    
    # 7. Predict foreground vs. background for every pixel
    #    We'll reshape the image into Nx3, predict, then reshape back
    # flat_img = img_converted.reshape((-1, C)).astype(np.float32)
    # if additional_features:
    #     flat_additional_features = [feat.reshape((-1, 1)) for feat in additional_features]
    #     flat_img = np.hstack([flat_img] + flat_additional_features)

    pred = rf.predict(np.hstack([img_converted.reshape(-1, C)] + additional_features))
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
    mask = remove_background_with_bbox(image_bgr, bounding_box, features=['lbp', 'ltp', 'quest', 'hog'])

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
