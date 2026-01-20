import ee
import pandas as pd
import numpy as np
import sys
import os
import json

def assign_spatial_folds(df, n_folds=5, grid_size=None):
    """
    Assigns fold IDs based on spatial grid blocks.
    If grid_size is None, it is calculated dynamically to ensure ~25 blocks.
    """
    df = df.copy()
    
    if grid_size is None:
        # Dynamic grid sizing
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # Target roughly 5x5 = 25 blocks to allow for sufficient variation
        # We take the larger dimension to drive the scale
        max_dim = max(lat_range, lon_range)
        grid_size = max_dim / 5.0
        
        # Ensure a minimum practical size (e.g. 100m ~ 0.001 deg) to avoid numerical issues
        grid_size = max(grid_size, 0.001)
        sys.stderr.write(f"Dynamic Spatial Blocking: Grid Size={grid_size:.4f} degrees (Extent: {lon_range:.2f}x{lat_range:.2f})\n")

    df['grid_x'] = (df['longitude'] / grid_size).apply(np.floor)
    df['grid_y'] = (df['latitude'] / grid_size).apply(np.floor)
    
    # Combine grid coordinates into a unique ID
    df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)
    
    unique_grids = df['grid_id'].unique()
    np.random.seed(42) # Deterministic blocking
    np.random.shuffle(unique_grids)
    
    # Assign grids to folds
    grid_to_fold = {grid: i % n_folds for i, grid in enumerate(unique_grids)}
    df['fold'] = df['grid_id'].map(grid_to_fold)
    
    return df

def calculate_classifier_metrics(scores_pos, scores_all):
    """
    Calculates CBI and AUC for a set of scores.
    """
    from autoSDM.analyzer import calculate_cbi
    cbi = calculate_cbi(scores_pos, scores_all)
    
    # AUC calculation
    # For speed and robustness, use a simple rank-based AUC
    # We need background scores
    # scores_all contains pos scores. Background = all - pos.
    # But wait, scores_all is the full set, scores_pos is a subset.
    
    n_pos = len(scores_pos)
    n_all = len(scores_all)
    n_bg = n_all - n_pos
    
    if n_bg > 0 and n_pos > 0:
        # Wilcoxon-Mann-Whitney U statistic based AUC
        from scipy.stats import mannwhitneyu
        # Mann-Whitney U test between pos and bg scores
        # We need the bg scores explicitly
        # This is a bit tricky if we only have all and pos.
        # Let's assume we can differentiate them by values if they are unique, 
        # but better to pass them explicitly or handle correctly.
        # Actually, in SDM, 'all' often means 'background' or 'presence + background'.
        # If scores_all is ALL points (including presences), then:
        
        # Simple iterative AUC (slow for large N but correct)
        # For CV we usually have small test sets, but let's be careful.
        # Let's use a faster way:
        ranks = pd.Series(scores_all).rank()
        pos_rank_sum = ranks[np.isin(scores_all, scores_pos)].sum()
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1) / 2)
        auc = u_stat / (n_pos * n_bg)
    else:
        auc = 0.5
        
    return {'cbi': cbi, 'auc': float(auc)}

def _prepare_training_data(df, nuisance_vars, ecological_vars, class_property='present', scale=10):
    """
    Shared logic for cleaning, sanitizing, and determining nuisance optima.
    """
    # 1. Class Property Detection - check multiple common naming conventions
    if class_property not in df.columns:
        for candidate in ['present', 'presence', 'Present', 'Present.', 'present?']:
            if candidate in df.columns:
                class_property = candidate
                break
        else:
            raise ValueError(f"Class property '{class_property}' not found in DataFrame. Available columns: {list(df.columns)[:10]}...")

    all_predictors = ecological_vars + nuisance_vars
    
    # 2. Sanitize column names
    name_map = {col: col.replace('.', '_') for col in all_predictors + [class_property, 'latitude', 'longitude']}
    df = df.rename(columns=name_map)
    all_predictors = [name_map[col] for col in all_predictors]
    nuisance_vars = [name_map[col] for col in nuisance_vars]
    ecological_vars = [name_map[col] for col in ecological_vars]
    class_property = name_map[class_property]

    # 3. Drop NAs (on predictors we have locally, like nuisance)
    df_clean = df.dropna(subset=nuisance_vars + [class_property]).copy()
    if df_clean.empty:
        raise ValueError("No valid training data remaining after dropping missing values.")

    # 4. Handle Categorical Nuisance Vars
    encodings = {}
    for var in nuisance_vars:
        vals = df_clean[var]
        is_string = pd.api.types.is_object_dtype(vals) or pd.api.types.is_categorical_dtype(vals)
        if is_string:
            try:
                pd.to_numeric(df_clean[var])
            except (ValueError, TypeError):
                sys.stderr.write(f"Encoding string-based nuisance variable '{var}' as factor.\n")
                codes, uniques = pd.factorize(df_clean[var])
                df_clean[var] = codes.astype(float)
                encodings[var] = uniques.tolist()

    # 5. Determine Nuisance Optima
    presence_df = df_clean[df_clean[class_property] == 1]
    if presence_df.empty:
        sys.stderr.write("Warning: No presence points found for nuisance optima.\n")
        presence_df = df_clean

    nuisance_optima = {}
    for var in nuisance_vars:
        vals = presence_df[var]
        if vals.nunique() <= 10 or var in encodings:
            optimum = float(vals.mode().iloc[0])
        else:
            counts, bin_edges = np.histogram(vals, bins='auto')
            max_idx = np.argmax(counts)
            optimum = float((bin_edges[max_idx] + bin_edges[max_idx+1]) / 2)
        nuisance_optima[var] = optimum
    
    # 6. Create FeatureCollection on Server (minimal payload)
    # We send coordinates and nuisance variables only. 
    # Alpha Earth embeddings (A00-A63) are sampled server-side to avoid 10MB payload limit.
    years = sorted(df_clean['year'].unique())
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    
    base_fcs = []
    for yr in years:
        yr_df = df_clean[df_clean['year'] == yr]
        features = []
        for _, row in yr_df.iterrows():
            props = {col: float(row[col]) for col in nuisance_vars + [class_property]}
            geom = ee.Geometry.Point([row['longitude'], row['latitude']])
            features.append(ee.Feature(geom, props))
        
        yr_fc = ee.FeatureCollection(features)
        
        # Sample embeddings server-side for this year
        year_img = ee.ImageCollection(asset_path).filter(ee.Filter.calendarRange(int(yr), int(yr), 'year')).mosaic()
        sampled_yr_fc = year_img.reduceRegions(
            collection=yr_fc,
            reducer=ee.Reducer.mean(),
            scale=scale
        )
        base_fcs.append(sampled_yr_fc)

    fc = ee.FeatureCollection(base_fcs).flatten()

    return {
        'fc': fc,
        'df_clean': df_clean,
        'nuisance_optima': nuisance_optima,
        'all_predictors': all_predictors,
        'class_property': class_property,
        'nuisance_vars': nuisance_vars,
        'ecological_vars': ecological_vars
    }

def train_maxent_model(df, nuisance_vars, ecological_vars, class_property='present', key_path=None, scale=10):
    if key_path:
        ee.Initialize(ee.ServiceAccountCredentials(json.load(open(key_path))["client_email"], key_path))

    data = _prepare_training_data(df, nuisance_vars, ecological_vars, class_property, scale=scale)

    classifier = ee.Classifier.amnhMaxent(
        autoFeature=True,
        outputFormat='cloglog'
    ).train(
        features=data['fc'],
        classProperty=data['class_property'],
        inputProperties=data['all_predictors']
    )

    # Performance on train set
    classified = data['fc'].classify(classifier, 'classification')
    presence_probs = np.array(classified.filter(ee.Filter.eq(data['class_property'], 1)).aggregate_array('classification').getInfo(), dtype=float)
    all_probs = np.array(classified.aggregate_array('classification').getInfo(), dtype=float)
    
    metrics = calculate_classifier_metrics(presence_probs, all_probs)
    sys.stderr.write(f"Maxent Analysis: CBI={metrics['cbi']:.4f}, AUC={metrics['auc']:.4f}\n")

    return classifier, data['nuisance_optima'], data['df_clean'], metrics

def run_cv_fold(fold_idx, df, nuisance_vars, ecological_vars, class_property, scale):
    import ee
    train_df = df[df['fold'] != fold_idx]
    test_df = df[df['fold'] == fold_idx]
    if test_df.empty: return None

    # Centroid
    from autoSDM.analyzer import analyze_embeddings
    c_res = analyze_embeddings(train_df, class_property=class_property)
    t_emb = test_df[[f"A{i:02d}" for i in range(64)]].values
    c_sims = np.dot(t_emb, c_res['centroid'])
    c_metrics = calculate_classifier_metrics(c_sims[test_df[class_property] == 1], c_sims)
    
    # Maxent
    m_data = _prepare_training_data(train_df, nuisance_vars, ecological_vars, class_property, scale=scale)
    m_clf = ee.Classifier.amnhMaxent(autoFeature=True, outputFormat='cloglog').train(m_data['fc'], m_data['class_property'], m_data['all_predictors'])
    t_data = _prepare_training_data(test_df, nuisance_vars, ecological_vars, class_property, scale=scale)
    t_clf = t_data['fc'].classify(m_clf, 'classification')
    m_sims = np.array(t_clf.aggregate_array('classification').getInfo(), dtype=float)
    m_metrics = calculate_classifier_metrics(m_sims[np.array(t_clf.aggregate_array(t_data['class_property']).getInfo()) == 1], m_sims)
    
    # Ensemble
    e_sims = c_sims * m_sims
    e_metrics = calculate_classifier_metrics(e_sims[test_df[class_property] == 1], e_sims)
    
    return {'centroid': c_metrics, 'maxent': m_metrics, 'ensemble': e_metrics}

def run_parallel_cv(df, nuisance_vars, ecological_vars, class_property='present', scale=10, n_folds=5):
    from concurrent.futures import ThreadPoolExecutor
    df_f = assign_spatial_folds(df, n_folds=n_folds)
    with ThreadPoolExecutor(max_workers=n_folds) as ex:
        res = [r for r in ex.map(lambda i: run_cv_fold(i, df_f, nuisance_vars, ecological_vars, class_property, scale), range(n_folds)) if r]
    
    avg = {}
    for m in ['centroid', 'maxent', 'ensemble']:
        avg[m] = {met: float(np.mean([r[m][met] for r in res])) for met in ['cbi', 'auc']}
    return avg
