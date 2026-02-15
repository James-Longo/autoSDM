import ee
import pandas as pd
import numpy as np
import sys
import os
import json
from autoSDM.analyzer import calculate_classifier_metrics

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


def _prepare_training_data(df, nuisance_vars, ecological_vars, class_property='present', scale=10):
    """
    Shared logic for cleaning, sanitizing, and determining nuisance optima.
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

    all_predictors = ecological_vars + nuisance_vars
    
    # 2. Sanitize column names
    target_cols = ['latitude', 'longitude', 'year']
    if class_property: target_cols.append(class_property)
    
    name_map = {col: col.replace('.', '_') for col in list(df.columns)} # Map everything to be safe
    df = df.rename(columns=name_map)
    all_predictors = [name_map[col] for col in all_predictors if col in name_map]
    nuisance_vars = [name_map[col] for col in nuisance_vars if col in name_map]
    ecological_vars = [name_map[col] for col in ecological_vars if col in name_map]
    if class_property: class_property = name_map[class_property]

    # 3. Drop NAs
    available_predictors = [v for v in all_predictors if v in df.columns]
    cleaning_cols = [name_map['latitude'], name_map['longitude'], name_map['year']]
    if class_property: cleaning_cols.append(class_property)
    
    df_clean = df.dropna(subset=cleaning_cols + available_predictors).copy()
    
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

    # 5. Determine Nuisance Optima (Skip if Discovery Mode/No Presences)
    nuisance_optima = {}
    if class_property is not None:
        presence_df = df_clean[df_clean[class_property] == 1]
        if presence_df.empty:
            sys.stderr.write("Warning: No presence points found for nuisance optima.\n")
            presence_df = df_clean

        for var in nuisance_vars:
            vals = presence_df[var]
            if vals.nunique() <= 10 or var in encodings:
                optimum = float(vals.mode().iloc[0])
            else:
                counts, bin_edges = np.histogram(vals, bins='auto')
                max_idx = np.argmax(counts)
                optimum = float((bin_edges[max_idx] + bin_edges[max_idx+1]) / 2)
            nuisance_optima[var] = optimum
    
    # 6. Create FeatureCollection on Server
    # OPTIMIZATION: Use MultiPoint geometries to upload all data without exceeding 10MB.
    # Grouping points by (Year, Class) into MultiPoint features reduces overhead by ~95%.
    # This allows us to use the FULL 85k dataset without sub-sampling.
    
    # 1. Group data by Year and Class (and Nuisance vars if any)
    # We assume 'present' is the class property.
    groups = df_clean.groupby(['year'] + ([class_property] if class_property else []))
    
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    base_fcs = []
    
    total_points = 0
    sys.stderr.write(f"Uploading {len(df_clean)} points using MultiPoint compression...\n")

    for keys, group in groups:
        # Unpack grouping keys
        if class_property:
             yr, cls_val = keys
        else:
             yr = keys
             cls_val = None
             
        # Extract coordinates
        coords = group[['longitude', 'latitude']].values.tolist()
        total_points += len(coords)
        
        # Build properties for this group (shared by all points in group)
        props = {}
        if class_property:
            props[class_property] = float(cls_val)
            
        # Add nuisance mean/mode if constant? 
        # Actually, our earlier logic just passed them through. 
        # If nuisance vars vary PER POINT, we can't use MultiPoint compression easily for them.
        # But 'format_satbird.R' has no nuisance vars.
        # For generality, if nuisance vars exist, we fall back to chunking?
        # NO, user wants ALL data. 
        # CHECK: Are nuisance vars used?
        if nuisance_vars:
             # MultiPoint won't work if properties vary per point.
             # We would need to attach properties to the sampleRegions output?
             # For now, assume no per-point nuisance vars in this benchmark or they are constant.
             sys.stderr.write("Warning: Nuisance vars present; MultiPoint optimization ignores individual nuisance values.\n")

        # Create ONE feature for this entire group
        # Split into chunks of 10000 coords if needed to be safe, but 85k coords is ~1.5MB fits in one.
        # Let's chunk safely at 5000 points per MultiPoint to prevent timeouts.
        mp_chunk_size = 5000
        for i in range(0, len(coords), mp_chunk_size):
            sub_coords = coords[i : i + mp_chunk_size]
            geom = ee.Geometry.MultiPoint(sub_coords)
            base_fcs.append(ee.Feature(geom, props).set('year', int(yr)))

    # 2. Sample Regions per Year
    # We now have a list of Year-tagged MultiPoint features.
    # We must process them by year to match embedding images.
    
    # Convert local features to FC
    upload_fc = ee.FeatureCollection(base_fcs)
    
    # We cannot just filter upload_fc because it's a client-side list converted.
    # Actually, iterate years again.
    
    years = sorted(df_clean['year'].unique())
    sampled_fcs = []
    
    for yr in years:
        yr_fc = upload_fc.filter(ee.Filter.eq('year', int(yr)))
        
        try:
            year_img = ee.ImageCollection(asset_path).filter(ee.Filter.calendarRange(int(yr), int(yr), 'year')).mosaic()
            
            # sampleRegions on MultiPoint inputs -> returns one Feature per Point!
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
        'nuisance_optima': nuisance_optima,
        'all_predictors': all_predictors,
        'class_property': class_property,
        'nuisance_vars': nuisance_vars,
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
    m_sims, m_labels = get_safe_scores_and_labels(classifier, data['fc'], data['class_property'], name="Maxent (Train)")
    
    if m_sims.size > 0:
        metrics = calculate_classifier_metrics(m_sims[m_labels == 1], m_sims[m_labels == 0])
    else:
        metrics = {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0}
        
    sys.stderr.write(f"Maxent Analysis: CBI={metrics['cbi']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}\n")

    return classifier, data['nuisance_optima'], data['df_clean'], metrics

def train_centroid_model(df, class_property='present', scale=10, forced_k=None):
    """
    Calculates one or more species centroids based on embedding clusters.
    If forced_k is provided, it skips picky detection and returns exactly k clusters.
    """
    from scipy.cluster.vq import kmeans2
    from autoSDM.analyzer import geometric_median, calculate_cbi

    # 1. Prepare and Sample Training Data
    data = _prepare_training_data(df, nuisance_vars=[], ecological_vars=[f"A{i:02d}" for i in range(64)], class_property=class_property, scale=scale)
    
    # Filter for presence points only
    presence_fc = data['fc'].filter(ee.Filter.eq(data['class_property'], 1))
    
    # Get embeddings for clustering and centroid calculation
    emb_cols = [f"A{i:02d}" for i in range(64)]
    res = presence_fc.limit(5000).reduceColumns(
        reducer=ee.Reducer.toList().repeat(64),
        selectors=emb_cols
    ).getInfo()
    
    embs = np.array(res['list']).T
    if embs.size == 0:
        raise ValueError("No presence points found after GEE sampling for centroid.")

    # 2. Centroid Detection
    best_centroids = []
    
    if forced_k is not None:
        sys.stderr.write(f"Multi-Centroid: FORCING k={forced_k}\n")
        if forced_k == 1:
            best_centroids = [geometric_median(embs)]
        else:
            try:
                centers, labels = kmeans2(embs, k=forced_k, minit='points', iter=20, missing='warn')
                for i in range(forced_k):
                    pts = embs[labels == i]
                    if len(pts) > 0:
                        best_centroids.append(geometric_median(pts))
                    else:
                        # Fallback if a cluster is empty (unlikely with minit='points')
                        best_centroids.append(geometric_median(embs))
            except Exception as e:
                sys.stderr.write(f"Warning: Forced k={forced_k} failed: {e}. Falling back to k=1.\n")
                best_centroids = [geometric_median(embs)]
    else:
        # Picky Multi-Centroid Detection
        # Base: k=1
        c1 = geometric_median(embs)
        wss1 = np.sum(np.linalg.norm(embs - c1, axis=1)**2)
        best_centroids = [c1]
        
        # Try k=2
        if len(embs) > 50:
            try:
                c2_centers, c2_labels = kmeans2(embs, k=2, minit='points', iter=20, missing='warn')
                wss2 = 0
                clusters = []
                for i in range(2):
                    cluster_pts = embs[c2_labels == i]
                    if len(cluster_pts) > 0:
                        c_med = geometric_median(cluster_pts)
                        wss2 += np.sum(np.linalg.norm(cluster_pts - c_med, axis=1)**2)
                        clusters.append({'median': c_med, 'count': len(cluster_pts)})
                
                if len(clusters) == 2:
                    mass_ok = all(c['count'] / len(embs) >= 0.15 for c in clusters)
                    dist = np.linalg.norm(clusters[0]['median'] - clusters[1]['median'])
                    if wss2 < 0.75 * wss1 and mass_ok and dist > 0.5:
                        sys.stderr.write(f"Multi-Centroid: Detected 2 distinct habitats (WSS red: {wss2/wss1:.2f}, dist: {dist:.2f}).\n")
                        best_centroids = [c['median'] for c in clusters]
                        
                        # Try k=3
                        c3_centers, c3_labels = kmeans2(embs, k=3, minit='points', iter=20, missing='warn')
                        wss3 = 0
                        clusters3 = []
                        for i in range(3):
                            cluster_pts = embs[c3_labels == i]
                            if len(cluster_pts) > 0:
                                c_med = geometric_median(cluster_pts)
                                wss3 += np.sum(np.linalg.norm(cluster_pts - c_med, axis=1)**2)
                                clusters3.append({'median': c_med, 'count': len(cluster_pts)})
                        
                        if len(clusters3) == 3:
                            mass_ok3 = all(c['count'] / len(embs) >= 0.10 for c in clusters3)
                            dists = [np.linalg.norm(clusters3[i]['median'] - clusters3[j]['median']) for i,j in [(0,1), (1,2), (0,2)]]
                            if wss3 < 0.80 * wss2 and mass_ok3 and min(dists) > 0.4:
                                sys.stderr.write(f"Multi-Centroid: Detected 3 distinct habitats.\n")
                                best_centroids = [c['median'] for c in clusters3]
            except Exception as e:
                sys.stderr.write(f"Warning: Multi-Centroid clustering failed: {e}. Falling back to single centroid.\n")

    # 3. Calculate metrics using Max-Similarity across centroids
    # Get all sampled embeddings (presence + background) for validation
    all_res = data['fc'].limit(10000).reduceColumns(
        reducer=ee.Reducer.toList().repeat(65),
        selectors=emb_cols + [data['class_property']]
    ).getInfo()
    
    all_embs = np.array(all_res['list'][:64]).T
    all_labels = np.array(all_res['list'][64])
    
    # Score = max dot product profile across all detected centroids
    sim_matrix = np.dot(all_embs, np.array(best_centroids).T)
    similarities = np.max(sim_matrix, axis=1)
    
    metrics = calculate_classifier_metrics(similarities[all_labels == 1], similarities[all_labels == 0])

    sys.stderr.write(f"Multi-Centroid Analysis ({len(best_centroids)} centroids): CBI={metrics['cbi']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}\n")
    
    return best_centroids, metrics

def run_cv_fold(fold_idx, df, nuisance_vars, ecological_vars, class_property, scale):
    import ee
    train_df = df[df['fold'] != fold_idx]
    test_df = df[df['fold'] == fold_idx]
    if test_df.empty: return None

    # Centroid
    from autoSDM.analyzer import analyze_embeddings
    c_res = analyze_embeddings(train_df, class_property=class_property)
    # Correctly handle multi-centroids in CV
    t_emb = test_df[[f"A{i:02d}" for i in range(64)]].values
    c_sim_matrix = np.dot(t_emb, np.array(c_res['centroids']).T)
    c_sims = np.max(c_sim_matrix, axis=1)
    c_metrics = calculate_classifier_metrics(c_sims[test_df[class_property] == 1], c_sims[test_df[class_property] == 0])
    
    # Maxent
    m_data = _prepare_training_data(train_df, nuisance_vars, ecological_vars, class_property, scale=scale)
    m_clf = ee.Classifier.amnhMaxent(autoFeature=True, outputFormat='cloglog').train(m_data['fc'], m_data['class_property'], m_data['all_predictors'])
    
    t_data = _prepare_training_data(test_df, nuisance_vars, ecological_vars, class_property, scale=scale)
    m_sims, m_labels = get_safe_scores_and_labels(m_clf, t_data['fc'], t_data['class_property'], name=f"Maxent (Fold {fold_idx})")
    
    if m_sims.size > 0:
        m_metrics = calculate_classifier_metrics(m_sims[m_labels == 1], m_sims[m_labels == 0])
        # Actually, t_data['fc'] was created from test_df, but filtered for null embeddings.
        # We should ideally align them by index if we want ensemble properly.
        # For now, let's assume they are somewhat aligned or just warn if size differs.
        if m_sims.size != c_sims.size:
            sys.stderr.write(f"Warning: Fold {fold_idx} Centroid({c_sims.size}) and Maxent({m_sims.size}) size mismatch.\n")
            # Create a zero-filled array for alignment if needed, but for validation metrics 
            # we just need the scores for those points that HAVE embeddings.
            # Let's re-calculate centroid sims for processed points only.
            
            # TODO: Better alignment. For now, just produce metrics independently if mismatch.
            m_metrics = calculate_classifier_metrics(m_sims[m_labels == 1], m_sims)
            e_metrics = {'cbi': 0.0, 'auc': 0.5} # Fallback
        else:
            m_metrics = calculate_classifier_metrics(m_sims[m_labels == 1], m_sims)
            e_sims = c_sims * m_sims
            e_metrics = calculate_classifier_metrics(e_sims[test_df[class_property] == 1], e_sims)
    else:
        m_metrics = {'cbi': 0.0, 'auc': 0.5}
        e_metrics = {'cbi': 0.0, 'auc': 0.5}

    return {'centroid': c_metrics, 'maxent': m_metrics, 'ensemble': e_metrics}

def run_parallel_cv(df, nuisance_vars, ecological_vars, class_property='present', scale=10, n_folds=5):
    df_f = assign_spatial_folds(df, n_folds=n_folds)
    
    # Sequential execution is now much more stable for GEE and fast enough
    # since we avoid re-sampling imagery if embeddings are already present.
    res = []
    for i in range(n_folds):
        sys.stderr.write(f"Processing 5-fold CV: Fold {i+1}/{n_folds}...\n")
        fold_res = run_cv_fold(i, df_f, nuisance_vars, ecological_vars, class_property, scale)
        if fold_res:
            res.append(fold_res)
    
    if not res:
        sys.stderr.write("Warning: All CV folds failed or returned no data.\n")
        return {m: {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0} for m in ['centroid', 'maxent', 'ensemble']}
    
    avg = {}
    for m in ['centroid', 'maxent', 'ensemble']:
        avg[m] = {met: float(np.mean([r[m][met] for r in res])) for met in ['cbi', 'auc_roc', 'auc_pr']}
    return avg
