"""
TODO:
- The image width in streamlit app is constrained by the maximum width of the column instead of actual image width 
    (we cant use actual since will cause problems for larger images, but need to ensure if scaling that the positions are also scaled when passing to function call)
"""

import cv2
import numpy as np
from lr import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
import streamlit as st
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from feature_extraction import quest_feature_extraction

def remove_background_with_bbox(img: np.ndarray, bboxes: list, max_samples: int = 5000, morph_kernel_size: int = 5, features: list = [], max_hp_tuning_iter: int = 50) -> np.ndarray:
    H, W, C = img.shape
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: out of image bounds or illogical coordinates.")
    
    img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    H, W, C = img_converted.shape

    # Gather Foreground (FG) samples from inside bounding box
    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1

    # Combine FG and BG training samples
    X = img_converted.reshape(-1, C)
    y = mask.reshape(-1)

    additional_features = []
    if 'lbp' in features:
        gray_img = rgb2gray(img_converted)
        lbp_features = local_binary_pattern(gray_img, P=8, R=1, method='uniform').reshape(-1, 1)
        print(f"LBP Features 1 shape: {lbp_features.shape}")
        additional_features.append(lbp_features)
        lbp_features = local_binary_pattern(gray_img, P=24, R=3, method='uniform').reshape(-1, 1)
        print(f"LBP Features 2 shape: {lbp_features.shape}")
        additional_features.append(lbp_features)
        lbp_features = local_binary_pattern(gray_img, P=72, R=9, method='uniform').reshape(-1, 1)
        print(f"LBP Features 3 shape: {lbp_features.shape}")
        additional_features.append(lbp_features)
    if 'quest' in features:
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
        print(f"HOG Features 1 shape: {hog_features.shape}")
        additional_features.append(hog_features)

        hog_features = hog(
            gray_img, 
            pixels_per_cell=(16, 16), 
            cells_per_block=(4, 4), 
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
        print(f"HOG Features 2 shape: {hog_features.shape}")
        additional_features.append(hog_features)

    if additional_features:
        X = np.hstack([X] + additional_features)

    # --- Modified Sampling Strategy ---
    # For background pixels, sample uniformly.
    bg_indices_all = np.where(y == 0)[0]
    if (8*max_samples) < len(bg_indices_all):
        bg_idx = np.random.choice(bg_indices_all, 8*max_samples, replace=False)
    else:
        bg_idx = bg_indices_all

    # For foreground pixels, sample with a bias toward the center of the bounding boxes.
    fg_indices_all = np.where(y == 1)[0]
    if len(fg_indices_all) > 0:
        # Compute (row, col) coordinates for each foreground pixel.
        rows = fg_indices_all // W
        cols = fg_indices_all % W
        fg_weights = np.zeros_like(rows, dtype=float)

        # For each bounding box, assign higher weights to pixels closer to the center.
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            in_bbox = (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)
            if np.any(in_bbox):
                center_row = (y1 + y2) / 2.0
                center_col = (x1 + x2) / 2.0
                sigma = min(x2 - x1, y2 - y1) / 4.0
                distances = np.sqrt((rows[in_bbox] - center_row)**2 + (cols[in_bbox] - center_col)**2)
                weights = np.exp(- (distances**2) / (2 * sigma**2))
                fg_weights[in_bbox] = weights

        # Normalize the weights so they sum to 1.
        if fg_weights.sum() > 0:
            fg_prob = fg_weights / fg_weights.sum()
        else:
            fg_prob = np.ones_like(fg_weights) / len(fg_weights)
        
        if max_samples < len(fg_indices_all):
            fg_idx = np.random.choice(fg_indices_all, max_samples, replace=False, p=fg_prob)
        else:
            fg_idx = fg_indices_all
    else:
        fg_idx = np.array([])

    # Combine background and foreground indices.
    idx = np.concatenate([bg_idx, fg_idx])
    X, y = X[idx], y[idx]
    # ---------------------------------

    # Shuffle to avoid any ordering bias
    X, y = shuffle(X, y, random_state=42)

    max_iter = max_hp_tuning_iter

    def hyperopt_objective(params, X, y, status_widget):
        """Objective function for Hyperopt to minimize."""
        model = LogisticRegression(**params, class_weight={0: 3, 1: 1})
        model.fit(X, y)
        accuracy = model.score(X, y)
        # accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

        # Increment a global counter stored in session_state
        st.session_state.iteration += 1
        iteration = st.session_state.iteration
        st.session_state.best_accuracy = max(accuracy, st.session_state.best_accuracy)
        best_accuracy = st.session_state.best_accuracy

        # Update the status widget with the current iteration and accuracy
        status_widget.text(f"Iteration {iteration}/{max_iter}: Accuracy = {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

        return {'loss': -accuracy, 'status': STATUS_OK}

    search_space = {
        'C': hp.loguniform('C', np.log(1e-4), np.log(1e4)),
        'max_iter': hp.choice('max_iter', list(range(500, 2001))),
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

    # # Convert best params to appropriate types
    best_params['C'] = float(best_params['C'])
    best_params['max_iter'] = max(int(best_params['max_iter']), 500)

    print("Best Hyperparameters found by Hyperopt:")
    print(best_params)

    st.success("Hyperparameter tuning completed!")
    st.write("Best Hyperparameters:", best_params)

    # Train the Logistic Regression model with the best hyperparameters
    lr = LogisticRegression(**best_params, class_weight={0: 3, 1: 1})
    lr.fit(X, y)

    # Print the accuracy score on the training data
    print(f"Logistic Regression Training Accuracy: {lr.score(X, y):.4f}")
    
    pred = lr.predict(np.hstack([img_converted.reshape(-1, C)] + additional_features))
    mask = pred.reshape((H, W))  # shape: HxW, values in {0,1}
    mask = mask.astype(np.uint8)

    # 8. Morphological refinement (closing -> opening, for example)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    # Closing to fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask