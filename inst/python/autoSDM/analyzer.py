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

def calculate_cbi(pos_scores, all_scores, window_width=0.1, n_bins=100):
    """
    Calculates the Continuous Boyce Index (CBI).
    Boyce Index is a measure of how much the model predictions differ from 
    random distribution of presence.
    """
    if len(pos_scores) == 0:
        return 0.0
    
    # Define the range of scores
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Range of evaluation points (centers of moving windows)
    eval_points = np.linspace(min_score, max_score, n_bins)
    
    f_obs = [] # Observed frequency (proportion of presences in window)
    f_exp = [] # Expected frequency (proportion of total area/background in window)
    
    half_width = (max_score - min_score) * window_width / 2
    
    for p in eval_points:
        # Window bounds
        low = p - half_width
        high = p + half_width
        
        # Count presences in window
        n_pos = np.sum((pos_scores >= low) & (pos_scores <= high))
        # Count total background/all points in window
        n_all = np.sum((all_scores >= low) & (all_scores <= high))
        
        # Observed frequency (proportion of presences)
        f_obs.append(n_pos / len(pos_scores) if len(pos_scores) > 0 else 0)
        # Expected frequency (proportion of total)
        f_exp.append(n_all / len(all_scores) if len(all_scores) > 0 else 0)
    
    f_obs = np.array(f_obs)
    f_exp = np.array(f_exp)
    
    # Avoid division by zero
    valid = f_exp > 0
    if not np.any(valid):
        return 0.0
        
    p_e = f_obs[valid] / f_exp[valid]
    
    # CBI is the Spearman rank correlation between P/E and the score (eval_points)
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(eval_points[valid], p_e)
    
    return float(correlation) if not np.isnan(correlation) else 0.0

def analyze_embeddings(df, class_property='present'):
    """
    Analyzes embeddings: calculates species centroid, similarity, and 
    validation metrics (CBI, AUC).
    Expects lowercase column name 'present' for presence/background data.
    """
    emb_cols = [f"A{i:02d}" for i in range(64)]
    
    # Check if all columns exist
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing embedding columns: {missing}")
    
    # Auto-detect presence column - try common variations
    if class_property not in df.columns:
        for candidate in ['present', 'presence', 'Present', 'Present.', 'present?']:
            if candidate in df.columns:
                class_property = candidate
                break
    
    # Drop NAs in embeddings and class property
    cols_to_check = emb_cols
    if class_property in df.columns:
        cols_to_check = emb_cols + [class_property]
    
    before_count = len(df)
    df_clean = df.dropna(subset=cols_to_check).copy()
    after_count = len(df_clean)
    
    if after_count < before_count:
        import sys
        sys.stderr.write(f"Centroid Analysis: Dropped {before_count - after_count} points due to missing embeddings or class property.\n")
    
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
    
    # Metric Calculation
    res_meta = {
        "centroid": centroid,
        "similarities": similarities,
        "clean_data": df_clean,
        "metrics": {}
    }
    
    if class_property in df_clean.columns:
        pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
        neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
        
        # CBI (Primary for Presence-Background)
        res_meta['metrics']['cbi'] = calculate_cbi(pos_scores, similarities)
        
        # AUC (Only if "absences" exist)
        if len(neg_scores) > 0:
            auc = 0
            for p in pos_scores:
                auc += np.sum(p > neg_scores)
                auc += 0.5 * np.sum(p == neg_scores)
            auc /= (len(pos_scores) * len(neg_scores))
            res_meta['metrics']['auc'] = float(auc)
        else:
            res_meta['metrics']['auc'] = 0.5
            
    else:
        # Fallback for Presence-Only
        res_meta['metrics']['cbi'] = calculate_cbi(similarities, similarities)
        res_meta['metrics']['auc'] = 0.5

    return res_meta
