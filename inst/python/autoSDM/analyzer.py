import numpy as np
import pandas as pd

def geometric_median(X, eps=1e-5, max_iter=100):
    """
    Calculates the Geometric Median of a set of points X.
    Weiszfeld's algorithm is an iterative step to find the point minimizing 
    the sum of Euclidean distances.
    """
    y = np.mean(X, axis=0) # Initial guess: arithmetic mean
    for _ in range(max_iter):
        dist = np.linalg.norm(X - y, axis=1)
        # Avoid division by zero
        dist = np.maximum(dist, 1e-10)
        
        weights = 1.0 / dist
        new_y = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        if np.linalg.norm(new_y - y) < eps:
            return new_y
        y = new_y
    return y

def analyze_embeddings(df, class_property='presence'):
    """
    Analyzes embeddings: calculates species centroid, cosine similarity, and 
    several thresholds (95% TPR, 95% TNR, Balanced) if presence/absence is available.
    """
    emb_cols = [f"A{i:02d}" for i in range(64)]
    
    # Check if all columns exist
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing embedding columns: {missing}")
    
    # Auto-detect presence column with various naming conventions
    if class_property not in df.columns:
        for candidate in ['present?', 'presence', 'Present.', 'Present', 'PRESENCE']:
            if candidate in df.columns:
                class_property = candidate
                break
    
    # Drop NAs in embeddings and class property
    cols_to_check = emb_cols
    if class_property in df.columns:
        cols_to_check = emb_cols + [class_property]
    
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    if df_clean.empty:
        raise ValueError(f"No valid data found after dropping NAs. Rows: {len(df)}")
        
    # Species Centroid (calculated from PRESENCE points only)
    if class_property in df_clean.columns and df_clean[class_property].sum() > 0:
        presence_df = df_clean[df_clean[class_property] == 1]
    else:
        presence_df = df_clean # Fallback to all data if binary distinction not possible
        
    # Using Geometric Median for robustness
    centroid = geometric_median(presence_df[emb_cols].values)
    
    # Similarities for ALL points
    similarities = np.dot(df_clean[emb_cols].values, centroid)
    df_clean['similarity'] = similarities
    
    # Threshold Calculation
    res_meta = {
        "centroid": centroid,
        "similarities": similarities,
        "clean_data": df_clean,
        "thresholds": {}
    }
    
    if class_property in df_clean.columns and (df_clean[class_property] == 0).any():
        # Tuning with Presence/Absence
        pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
        neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
        
        # 95% TPR (5th percentile of presence)
        res_meta['thresholds']['95tpr'] = float(np.percentile(pos_scores, 5))
        
        # 95% TNR (95th percentile of absence)
        res_meta['thresholds']['95tnr'] = float(np.percentile(neg_scores, 95))
        
        # Balanced (Maximize TPR + TNR)
        y_scores = df_clean['similarity'].values
        threshold_candidates = np.unique(y_scores)
        best_t = 0
        max_sum = 0
        for t in threshold_candidates:
            tpr = np.mean(pos_scores >= t)
            tnr = np.mean(neg_scores < t)
            if tpr + tnr > max_sum:
                max_sum = tpr + tnr
                best_t = t
        res_meta['thresholds']['balanced'] = float(best_t)
        
        # AUC
        auc = 0
        for p in pos_scores:
            auc += np.sum(p > neg_scores)
            auc += 0.5 * np.sum(p == neg_scores)
        auc /= (len(pos_scores) * len(neg_scores))
        res_meta['thresholds']['auc'] = float(auc)
        
    else:
        # Fallback for Presence-Only (legacy behavior)
        res_meta['thresholds']['95tpr'] = float(np.percentile(similarities, 5))
        res_meta['thresholds']['balanced'] = float(np.percentile(similarities, 50))
        res_meta['thresholds']['auc'] = 0.5

    return res_meta
