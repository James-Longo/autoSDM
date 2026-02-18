import numpy as np
import pandas as pd



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

def calculate_classifier_metrics(scores_pos, scores_neg):
    """
    Calculates CBI, AUC-ROC, and AUC-PR for model evaluation.
    scores_pos: scores for presence points
    scores_neg: scores for background/absence points
    """
    # Combined scores for CBI and ranking
    scores_all = np.concatenate([scores_pos, scores_neg])
    cbi = calculate_cbi(scores_pos, scores_all)
    
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    
    if n_pos == 0 or n_neg == 0:
        return {
            'cbi': float(cbi),
            'auc_roc': 0.5,
            'auc_pr': 0.0,
            'tss': 0.0,
            'ba': 0.5,
            'threshold_5pct': float(np.percentile(scores_pos, 5)) if n_pos > 0 else 0.0,
            'threshold_10pct': float(np.percentile(scores_pos, 10)) if n_pos > 0 else 0.0,
            'point_biserial': 0.0
        }

    # 1. AUC-ROC (Mann-Whitney U method - Rank based)
    # Ranks of all scores
    ranks = pd.Series(scores_all).rank(method='average')
    pos_ranks = ranks[:n_pos]
    auc_roc = (pos_ranks.sum() - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)

    # 2. AUC-PR (Trapezoidal integration)
    # Sort points by score descending
    sorted_indices = np.argsort(scores_all)[::-1]
    sorted_labels = np.zeros(len(scores_all))
    sorted_labels[:n_pos] = 1
    sorted_labels = sorted_labels[sorted_indices]
    
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    
    precision = tp / (tp + fp)
    recall = tp / n_pos
    
    # Add start point (Recall 0, Precision 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    
    # AUC-PR via Trapezoidal rule
    auc_pr = np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2)

    # 3. TSS and Balanced Accuracy
    sens = tp / n_pos
    spec = (n_neg - fp) / n_neg
    combined = sens + spec
    tss = np.max(combined - 1)
    ba = np.max(combined / 2)

    # 4. Suitability Thresholds (Omission rates)
    # threshold_5pct: Suitability value above which 95% of presences lie
    threshold_5pct = np.percentile(scores_pos, 5)
    threshold_10pct = np.percentile(scores_pos, 10)

    # 5. Point-Biserial Correlation (alternative to AUC for PO data)
    from scipy.stats import pointbiserialr
    # Correlation between binary labels (Presence=1, Absence=0) and continuous scores
    pb_corr, _ = pointbiserialr(np.concatenate([np.ones(n_pos), np.zeros(n_neg)]), scores_all)

    return {
        'cbi': float(cbi),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'tss': float(tss),
        'ba': float(ba),
        'threshold_5pct': float(threshold_5pct),
        'threshold_10pct': float(threshold_10pct),
        'point_biserial': float(pb_corr) if not np.isnan(pb_corr) else 0.0
    }

def analyze_embeddings(df, class_property='present'):
    """
    Analyzes embeddings: calculates the environmental centroid (arithmetic mean)
    of presence points and computes validation metrics (CBI, AUC).
    """
    import sys

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
        sys.stderr.write(f"Centroid Analysis: Dropped {before_count - after_count} points due to missing embeddings or class property.\n")
    
    if df_clean.empty:
        raise ValueError(f"No valid data found after dropping NAs. Rows: {len(df)}")
        
    # Species Embeddings (Presence points)
    if class_property in df_clean.columns and df_clean[class_property].sum() > 0:
        presence_df = df_clean[df_clean[class_property] == 1]
    else:
        presence_df = df_clean # Fallback
        
    embs = presence_df[emb_cols].values
    
    # --- Centroid Calculation ---
    # We strictly use the arithmetic mean (single centroid)
    centroid = np.mean(embs, axis=0)

    # Similarity for ALL points (Direct dot product)
    similarities = np.dot(df_clean[emb_cols].values, centroid)
    df_clean['similarity'] = similarities
    
    res_meta = {
        "centroids": [centroid.tolist() if hasattr(centroid, 'tolist') else centroid],
        "similarities": similarities.tolist() if hasattr(similarities, 'tolist') else similarities,
        "clean_data": df_clean,
        "metrics": {}
    }
    
    if class_property in df_clean.columns:
        pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
        neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
        res_meta['metrics'] = calculate_classifier_metrics(pos_scores, neg_scores)
    else:
        res_meta['metrics'] = {'cbi': 0, 'auc_roc': 0.5, 'auc_pr': 0}

    return res_meta

def analyze_ridge(df, class_property='present'):
    """
    Linear Predictor (Ordinary Least Squares / Ridge alpha=0).
    Trains on presence=1, absence=-1.
    """
    from sklearn.linear_model import Ridge
    import sys

    emb_cols = [f"A{i:02d}" for i in range(64)]
    df_clean = df.dropna(subset=emb_cols + [class_property]).copy()
    
    X = df_clean[emb_cols].values
    # Linear Predictor usually maps Presence to 1 and Absence to -1
    y = np.where(df_clean[class_property] == 1, 1.0, -1.0)
    
    # Ridge with alpha=0 is OLS
    model = Ridge(alpha=0.0)
    model.fit(X, y)
    
    similarities = model.predict(X)
    df_clean['similarity'] = similarities
    
    res_meta = {
        "weights": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "similarities": similarities,
        "clean_data": df_clean,
        "metrics": {}
    }
    
    pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
    neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
    res_meta['metrics'] = calculate_classifier_metrics(pos_scores, neg_scores)
    
    return res_meta

def analyze_knn(df, k=3, class_property='present'):
    """
    k-Nearest Neighbors (kNN) using Euclidean distance.
    Returns majority vote (0 or 1) as similarity score.
    Note: For SDM comparison, we use the probability of presence as the suitability score.
    """
    from sklearn.neighbors import KNeighborsClassifier
    import sys

    emb_cols = [f"A{i:02d}" for i in range(64)]
    df_clean = df.dropna(subset=emb_cols + [class_property]).copy()
    
    X = df_clean[emb_cols].values
    y = df_clean[class_property].values
    
    # The paper uses L2 distance (Euclidean)
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X, y)
    
    # suitability = probability of class 1
    # If the user wants "assigns that label", it would be 0 or 1.
    # But for ROC/PR/CBI metrics, we need a continuous score. 
    # probalities provide n_neighbors levels of score.
    probs = model.predict_proba(X)
    # Check if both classes are present
    if probs.shape[1] > 1:
        similarities = probs[:, 1]
    else:
        # Only one class present in training? Unlikely for kNN SDM
        similarities = probs[:, 0] if y[0] == 1 else (1.0 - probs[:, 0])
        
    df_clean['similarity'] = similarities
    
    res_meta = {
        "train_X": X, # In reality we'd save this or aKDTree
        "train_y": y,
        "k": k,
        "similarities": similarities,
        "clean_data": df_clean,
        "metrics": {}
    }
    
    pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
    neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
    res_meta['metrics'] = calculate_classifier_metrics(pos_scores, neg_scores)
    
    return res_meta
    
def analyze_mean(df, class_property='present'):
    """
    Mean Similarity: Raw dot product between the arithmetic mean of the 
    embedding values at presence points and all other points.
    """
    emb_cols = [f"A{i:02d}" for i in range(64)]
    df_clean = df.dropna(subset=emb_cols + [class_property]).copy()
    
    presence_embs = df_clean[df_clean[class_property] == 1][emb_cols].values
    if len(presence_embs) == 0:
        raise ValueError("No presence points found for mean calculation.")
        
    mean_vec = np.mean(presence_embs, axis=0)
    
    similarities = np.dot(df_clean[emb_cols].values, mean_vec)
    df_clean['similarity'] = similarities
    
    res_meta = {
        "centroid": mean_vec.tolist(),
        "similarities": similarities,
        "clean_data": df_clean,
        "metrics": {}
    }
    
    pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
    neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
    res_meta['metrics'] = calculate_classifier_metrics(pos_scores, neg_scores)
    
    return res_meta
