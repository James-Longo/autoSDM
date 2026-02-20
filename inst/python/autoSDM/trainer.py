import ee
import pandas as pd
import numpy as np
import sys
import os
import json
from autoSDM.analyzer import calculate_classifier_metrics

def assign_spatial_folds(df, n_folds=10, grid_size=None):
    """
    Assigns fold IDs based on spatial k-means clustering of points.
    """
    from sklearn.cluster import KMeans
    df = df.copy()
    
    # Use spatial coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Deterministic K-Means clustering
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    df['fold'] = kmeans.fit_predict(coords)
    
    # Report cluster sizes
    counts = df['fold'].value_counts().sort_index()
    sys.stderr.write(f"Spatial K-Means Folding ({n_folds} clusters):\n")
    for fold, count in counts.items():
        sys.stderr.write(f"  Fold {fold}: {count} points\n")
        
    return df


def _prepare_training_data(df, ecological_vars, class_property='present', scale=10):
    """
    Shared logic for cleaning and sanitizing training data.
    """
    # 1. Class Property Detection (Skip if Discovery Mode)
    if class_property is not None:
        if class_property not in df.columns:
            for candidate in ['present', 'presence', 'Present', 'Present.', 'present?']:
                if candidate in df.columns:
                    class_property = candidate
                    break
            else:
                raise ValueError(f"Class property '{class_property}' not found in DataFrame. Available columns: {list(df.columns)[:10]}...")

    # 2. Sanitize column names
    target_cols = ['latitude', 'longitude', 'year']
    if class_property: target_cols.append(class_property)
    
    name_map = {col: col.replace('.', '_') for col in list(df.columns)} # Map everything to be safe
    df = df.rename(columns=name_map)
    ecological_vars = [name_map[col] for col in ecological_vars if col in name_map]
    if class_property: class_property = name_map[class_property]

    # 3. Drop NAs
    available_predictors = [v for v in ecological_vars if v in df.columns]
    cleaning_cols = [name_map['latitude'], name_map['longitude'], name_map['year']]
    if class_property: cleaning_cols.append(class_property)
    
    df_clean = df.dropna(subset=cleaning_cols + available_predictors).copy()
    
    if df_clean.empty:
        raise ValueError("No valid training data remaining after dropping missing values.")

    # 4. Create FeatureCollection on Server
    # OPTIMIZATION: Use MultiPoint geometries to upload all data without exceeding 10MB.
    # Grouping points by (Year, Class) into MultiPoint features reduces overhead by ~95%.
    groups = df_clean.groupby(['year'] + ([class_property] if class_property else []))
    
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    base_fcs = []
    
    total_points = 0
    sys.stderr.write(f"Uploading {len(df_clean)} points using MultiPoint compression...\n")

    for keys, group in groups:
        if class_property:
             yr, cls_val = keys
        else:
             yr = keys
             cls_val = None
             
        coords = group[['longitude', 'latitude']].values.tolist()
        total_points += len(coords)
        
        props = {}
        if class_property:
            props[class_property] = float(cls_val)
            
        mp_chunk_size = 5000
        for i in range(0, len(coords), mp_chunk_size):
            sub_coords = coords[i : i + mp_chunk_size]
            geom = ee.Geometry.MultiPoint(sub_coords)
            base_fcs.append(ee.Feature(geom, props).set('year', int(yr)))

    # 5. Sample Regions per Year
    upload_fc = ee.FeatureCollection(base_fcs)
    years = sorted(df_clean['year'].unique())
    sampled_fcs = []
    
    for yr in years:
        yr_fc = upload_fc.filter(ee.Filter.eq('year', int(yr)))
        try:
            year_img = ee.ImageCollection(asset_path).filter(ee.Filter.calendarRange(int(yr), int(yr), 'year')).mosaic()
            sampled = year_img.sampleRegions(
                collection=yr_fc,
                scale=scale,
                geometries=True
            ).filter(ee.Filter.notNull(['A00']))
            sampled_fcs.append(sampled)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to sample GEE for year {yr}: {e}\n")    

    if not sampled_fcs:
        raise ValueError("Failed to create any valid training FeatureCollections on GEE.")

    fc = ee.FeatureCollection(sampled_fcs).flatten()
    sys.stderr.write(f"Successfully created training FC for {total_points} points.\n")

    return {
        'fc': fc,
        'df_clean': df_clean,
        'all_predictors': ecological_vars,
        'class_property': class_property,
        'ecological_vars': ecological_vars
    }

def get_safe_scores_and_labels(clf, fc, class_property, name="Model"):
    """
    Safely retrieves scores and labels from a classified FeatureCollection.
    Handles mismatches, empty results, and "Too many concurrent aggregations" errors.
    """
    import time
    import random

    for attempt in range(3):
        try:
            classified = fc.classify(clf, 'classification_temp')
            # 1. Inspect first feature to find score column
            first = classified.first().getInfo()
            if not first:
                sys.stderr.write(f"Warning: {name} classified FC returned no features.\n")
                return np.array([]), np.array([])
                
            props = first['properties']
            score_col = 'classification_temp'
            if score_col not in props:
                # Fallback to probability if classification_temp is missing
                score_col = 'probability' if 'probability' in props else None
            
            if not score_col:
                sys.stderr.write(f"Warning: {name} returned no identifiable score column. Keys: {list(props.keys())}\n")
                return np.array([]), np.array([])

            # 2. Combined reduction for alignment
            res = classified.reduceColumns(
                reducer=ee.Reducer.toList().repeat(2),
                selectors=[score_col, class_property]
            ).getInfo()
            
            scores = np.array(res['list'][0], dtype=float)
            labels = np.array(res['list'][1], dtype=float)
            
            if scores.size == 0:
                sys.stderr.write(f"Warning: {name} returned no scores.\n")
                
            return scores, labels
        except Exception as e:
            err_str = str(e)
            if "Too many concurrent aggregations" in err_str or "Quotas were exceeded" in err_str:
                wait = (2 ** attempt) + random.random()
                sys.stderr.write(f"GEE Concurrency Limit hit for {name}. Retrying in {wait:.2f}s... (Attempt {attempt+1}/3)\n")
                time.sleep(wait)
                continue
            
            sys.stderr.write(f"Error retrieving scores for {name}: {e}\n")
            return np.array([]), np.array([])
    
    return np.array([]), np.array([])



def train_centroid_model(df, class_property='present', scale=10, aoi=None, year=2025):
    """
    Calculates the species centroid based on embeddings.
    If no absences are provided (Presence-Only), it samples background points from GEE
    to provide accuracy metrics (AUC, Boyce Index, etc.)
    """
    from autoSDM.analyzer import calculate_classifier_metrics

    # 1. Prepare and Sample Training Data
    data = _prepare_training_data(df, ecological_vars=[f"A{i:02d}" for i in range(64)], class_property=class_property, scale=scale)
    
    # Filter for presence points only
    presence_fc = data['fc'].filter(ee.Filter.eq(data['class_property'], 1))
    
    # Get embeddings for centroid calculation
    emb_cols = [f"A{i:02d}" for i in range(64)]
    res = presence_fc.limit(5000).reduceColumns(
        reducer=ee.Reducer.toList().repeat(64),
        selectors=emb_cols
    ).getInfo()
    
    embs = np.array(res['list']).T
    if embs.size == 0:
        raise ValueError("No presence points found after GEE sampling for centroid.")

    # 2. Centroid Calculation
    centroid = np.mean(embs, axis=0)

    # 3. Validation
    # Identify if we have absences for validation
    abs_fc = data['fc'].filter(ee.Filter.eq(data['class_property'], 0))
    has_absences = abs_fc.size().getInfo() > 0
    
    if has_absences:
        # Standard PA Validation
        all_res = data['fc'].limit(10000).reduceColumns(
            reducer=ee.Reducer.toList().repeat(65),
            selectors=emb_cols + [data['class_property']]
        ).getInfo()
        all_embs = np.array(all_res['list'][:64]).T
        all_labels = np.array(all_res['list'][64])
        similarities = np.dot(all_embs, centroid)
        metrics = calculate_classifier_metrics(similarities[all_labels == 1], similarities[all_labels == 0])
    else:
        # Presence-Only Validation: Sample background for metrics
        sys.stderr.write("Presence-only training detected for Centroid. Generating background for evaluation...\n")
        # Use presences for pos_scores
        pos_scores = np.dot(embs, centroid)
        
        # Sample background
        if aoi:
            n_bg = len(embs) * 10
            df_bg = get_background_embeddings(aoi, n_points=n_bg, scale=scale, year=year)
            if not df_bg.empty:
                bg_embs = df_bg[emb_cols].values
                neg_scores = np.dot(bg_embs, centroid)
                metrics = calculate_classifier_metrics(pos_scores, neg_scores)
            else:
                metrics = {'cbi': 0, 'auc_roc': 0.5, 'auc_pr': 0}
        else:
            sys.stderr.write("Warning: AOI missing, cannot sample background for PO evaluation metrics.\n")
            metrics = {'cbi': 0, 'auc_roc': 0.5, 'auc_pr': 0}

    sys.stderr.write(f"Environmental Centroid Analysis: CBI={metrics.get('cbi', 0):.4f}, AUC-ROC={metrics.get('auc_roc', 0.5):.4f}, AUC-PR={metrics.get('auc_pr', 0):.4f}\n")
    
    return [centroid], metrics

def run_cv_fold(fold_idx, df, ecological_vars, class_property, scale):
    import ee
    train_df = df[df['fold'] != fold_idx]
    test_df = df[df['fold'] == fold_idx]
    if test_df.empty: return None

    from autoSDM.analyzer import analyze_embeddings, analyze_ridge
    
    # 1. Centroid
    c_res = analyze_embeddings(train_df, class_property=class_property)
    t_emb = test_df[[f"A{i:02d}" for i in range(64)]].values
    c_sim_matrix = np.dot(t_emb, np.array(c_res['centroids']).T)
    c_sims = np.max(c_sim_matrix, axis=1)
    c_metrics = calculate_classifier_metrics(c_sims[test_df[class_property] == 1], c_sims[test_df[class_property] == 0])
    
    # 2. Ridge
    r_res = analyze_ridge(train_df, class_property=class_property)
    r_weights = np.array(r_res['weights'])
    r_intercept = r_res['intercept']
    r_sims = np.dot(t_emb, r_weights) + r_intercept
    r_metrics = calculate_classifier_metrics(r_sims[test_df[class_property] == 1], r_sims[test_df[class_property] == 0])
    
    # 3. Ensemble (Normalized Centroid * Normalized Ridge)
    # We normalize both to 0-1 based on the min/max of the test set scores
    # This prevents one method from dominating the product due to score scale.
    def normalize(vals):
        v_min, v_max = np.min(vals), np.max(vals)
        if v_max - v_min == 0: return np.zeros_like(vals)
        return (vals - v_min) / (v_max - v_min)
        
    e_sims = normalize(c_sims) * normalize(r_sims)
    e_sims = normalize(e_sims) # Final normalization of the ensemble product
    e_metrics = calculate_classifier_metrics(e_sims[test_df[class_property] == 1], e_sims[test_df[class_property] == 0])

    # Calculate point counts for reporting
    n_pos = int(np.sum(test_df[class_property] == 1))
    n_neg = int(np.sum(test_df[class_property] == 0))

    return {
        'centroid': c_metrics, 
        'ridge': r_metrics, 
        'ensemble': e_metrics,
        'counts': {'presence': n_pos, 'background': n_neg}
    }

def run_parallel_cv(df, ecological_vars, class_property='present', scale=10, n_folds=10):
    df_f = assign_spatial_folds(df, n_folds=n_folds)
    
    # Sequential execution is now much more stable for GEE and fast enough
    # since we avoid re-sampling imagery if embeddings are already present.
    res = []
    for i in range(n_folds):
        sys.stderr.write(f"Processing 10-fold CV: Fold {i+1}/{n_folds}...\n")
        fold_res = run_cv_fold(i, df_f, ecological_vars, class_property, scale)
        if fold_res:
            res.append(fold_res)
    
    if not res:
        sys.stderr.write("Warning: All CV folds failed or returned no data.\n")
        return {
            'centroid': {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0},
            'ridge': {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0},
            'ensemble': {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0}
        }
    
    avg = {}
    for m in ['centroid', 'ridge', 'ensemble']:
        # Filter for folds that have presences to avoid skewed averages (0.5 AUC / 0 CBI)
        valid_folds = [r for r in res if r['counts']['presence'] > 0]
        if not valid_folds:
            # Fallback if no fold has presences (shouldn't happen with valid data)
            valid_folds = res
            
        avg[m] = {
            met: float(np.mean([r[m][met] for r in valid_folds])) 
            for met in ['cbi', 'auc_roc', 'auc_pr']
        }
    return {'average': avg, 'folds': res, 'df': df_f}

def get_background_embeddings(aoi, n_points=1000, scale=100, year=2025):
    """
    Samples random background points in GEE and extracts their Alpha Earth embeddings.
    Uses image.sample() to ensure points fall on valid data pixels ("intrinsic mask").
    """
    import ee
    import pandas as pd
    import json
    sys.stderr.write(f"Sampling {n_points} background points from GEE (Year: {year})...\n")
    
    emb_cols = [f"A{i:02d}" for i in range(64)]
    
    collected_rows = []
    max_attempts = 10
    
    # Pre-define embedding image ONCE
    # We must use the mosaic to get data, but sampling at points is faster than scanning the image.
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    year_img = ee.ImageCollection(asset_path).filter(ee.Filter.calendarRange(int(year), int(year), 'year')).mosaic()
    
    for attempt in range(max_attempts):
        needed = n_points - len(collected_rows)
        if needed <= 0:
            break
            
        # Generate random points geometrically (FAST)
        # We request the full n_points batch to maximize yield in each iteration
        # Use a deterministic seed that changes per attempt to avoid repeated points
        seed = attempt 
        request_n = n_points
        sys.stderr.write(f"  Attempt {attempt+1}: Generating {request_n} random points (Need {needed}, Seed: {seed})...\n")
        
        random_pts = ee.FeatureCollection.randomPoints(aoi, request_n, seed=seed)
        
        # Extract embeddings at these points
        # sampleRegions is efficient because it only computes pixels at the points
        try:
            samples = year_img.sampleRegions(
                collection=random_pts,
                properties=[], # We don't need properties from the random points
                scale=scale,
                geometries=True
            )
            res = samples.getInfo()
        except Exception as e:
            sys.stderr.write(f"  Error in sampling attempt {attempt+1}: {e}\n")
            continue
            
        if not res['features']:
            sys.stderr.write(f"  Warning: Attempt {attempt+1} yielded no valid data points (all masked?).\n")
            continue
            
        # Parse features
        valid_count = 0
        for feat in res['features']:
            props = feat['properties']
            geom = feat['geometry']['coordinates']
            
            # Check if we have valid embeddings (A00 should exist)
            if 'A00' not in props: 
                continue
                
            row = {
                'longitude': geom[0],
                'latitude': geom[1],
                'present': 0,
                'type': 'background'
            }
            # Add embeddings
            for col in emb_cols:
                row[col] = props.get(col, 0.0)
                
            collected_rows.append(row)
            valid_count += 1
            
        sys.stderr.write(f"  Got {valid_count} valid points. Total: {len(collected_rows)}/{n_points}\n")
            
    df_bg = pd.DataFrame(collected_rows)
    
    # Shuffle and limit to exactly n_points
    if len(df_bg) > n_points:
        df_bg = df_bg.sample(n=n_points, random_state=42).reset_index(drop=True)
    elif len(df_bg) < n_points:
        sys.stderr.write(f"Warning: Could not collect enough background points after {max_attempts} attempts. Found {len(df_bg)}/{n_points}.\n")
        
    sys.stderr.write(f"Background sampling complete. Final count: {len(df_bg)}\n")
    
    return df_bg

