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

def _dot_product_on_fc(fc, weights, emb_cols, label_col, intercept=0.0):
    """
    Compute server-side dot product scores on a pre-sampled FeatureCollection.

    The FC already has embedding values as properties (from an earlier
    sampleRegions call), so NO new sampleRegions is needed.  The dot product
    expression is built client-side as a 64-step GEE graph and executed
    entirely on GEE's servers.

    Returns (sims, labels) as numpy arrays — only N scalar scores cross the wire.
    """
    import ee

    # Build the dot-product GEE expression using the already-present embedding
    # properties.  The Python loop constructs the expression graph client-side;
    # evaluation happens server-side on the filtered FC.
    intercept_num = ee.Number(float(intercept))

    def add_score(feat):
        sim = intercept_num
        for col, w in zip(emb_cols, weights):
            sim = sim.add(ee.Number(feat.get(col)).multiply(float(w)))
        return feat.set('similarity', sim)

    scored = fc.map(add_score).filter(ee.Filter.notNull(['similarity']))
    r = scored.reduceColumns(
        ee.Reducer.toList().repeat(2),
        selectors=['similarity', label_col]
    ).getInfo()
    return (
        np.array(r['list'][0], dtype=float),
        np.array(r['list'][1], dtype=float),
    )


def run_cv_fold(fold_idx, all_sampled_fc, primary_img, emb_cols,
                label_col_ridge, label_col_cent, fold_col, scale,
                train_methods=None, eval_methods=None):
    """
    Evaluate a single CV fold using a pre-sampled FeatureCollection.

    Supports all GEE classifier methods (centroid, rf, gbt, cart, svm,
    maxent) via ee.Classifier + PROBABILITY mode, and all reducer methods
    (ridge, linear, robust_linear) via reduceColumns.

    The ensemble score is the normalized product of all trained submodel scores.
    """
    import ee
    from autoSDM.analyzer import GEE_CLASSIFIER_METHODS, GEE_REDUCER_METHODS

    train_methods = train_methods or ["centroid", "ridge"]
    eval_methods  = eval_methods  or ["ensemble"]

    train_fc = all_sampled_fc.filter(ee.Filter.neq(fold_col, fold_idx))
    test_fc  = all_sampled_fc.filter(ee.Filter.eq(fold_col,  fold_idx))

    res_metrics = {}
    # Store per-method scores for ensemble combination
    method_sims   = {}  # method -> np.array of scores
    method_labels = {}  # method -> np.array of 0/1 labels

    for method in train_methods:
        # ── GEE Classifier ────────────────────────────────────────────────
        if method in GEE_CLASSIFIER_METHODS:
            # Use binary 0/1 label (centroid label column stores 1.0/0.0)
            clf = GEE_CLASSIFIER_METHODS[method]({})
            trained = clf.train(
                features=train_fc,
                classProperty=label_col_cent,
                inputProperties=emb_cols,
            )
            scored_test = test_fc.classify(trained)
            
            # Maxent uses 'probability' property, others use 'classification'
            score_col = "probability" if method == "maxent" else "classification"

            result = scored_test.reduceColumns(
                ee.Reducer.toList().repeat(2),
                selectors=[score_col, label_col_cent]
            ).getInfo()
            sims   = np.array(result["list"][0], dtype=float)
            if method == "svm":
                sims = 1.0 - sims
            labels = np.array(result["list"][1], dtype=float)

        # ── GEE Reducer ───────────────────────────────────────────────────
        elif method in GEE_REDUCER_METHODS:
            train_fc_reg = train_fc.map(lambda f: f.set("constant", 1.0))
            reducer_name, label_enc = GEE_REDUCER_METHODS[method]
            lbl_col = label_col_ridge  # binary +1/-1
            
            if reducer_name == "ridgeRegression":
                # Ridge has a native intercept at index 0
                reducer  = ee.Reducer.ridgeRegression(numX=64, numY=1, lambda_=0.1)
                result   = train_fc.reduceColumns(reducer=reducer, selectors=emb_cols + [lbl_col])
                coef_flat = [float(row[0]) for row in result.get("coefficients").getInfo()]
                intercept = coef_flat[0]; weights = coef_flat[1:]
            elif reducer_name in ("linearRegression", "robustLinearRegression"):
                # Linear needs our 'constant' column
                train_fc_reg = train_fc.map(lambda f: f.set("constant", 1.0))
                factory  = getattr(ee.Reducer, reducer_name)
                reducer  = factory(numX=65, numY=1)
                result   = train_fc_reg.reduceColumns(reducer=reducer, selectors=["constant"] + emb_cols + [lbl_col])
                coef_flat = [float(row[0]) for row in result.get("coefficients").getInfo()]
                intercept = coef_flat[0]; weights = coef_flat[1:]
            elif reducer_name == "mean":
                presence_fc = train_fc.filter(ee.Filter.eq(label_col_cent, 1))
                centroid_res = presence_fc.reduceColumns(
                    reducer=ee.Reducer.mean().repeat(len(emb_cols)), selectors=emb_cols
                ).getInfo()
                weights = [float(w) for w in centroid_res["mean"]]
                intercept = 0.0
            else:
                continue  # linearFit not implemented in CV

            sims, labels = _dot_product_on_fc(test_fc, weights, emb_cols, label_col_cent, intercept=intercept)
        else:
            continue

        method_sims[method]   = sims
        method_labels[method] = labels

        if method in eval_methods:
            res_metrics[method] = calculate_classifier_metrics(
                sims[labels == 1], sims[labels == 0]
            )

    # ── Ensemble ──────────────────────────────────────────────────────────
    if "ensemble" in eval_methods and len(method_sims) >= 2:
        def normalize(vals):
            v_min, v_max = np.min(vals), np.max(vals)
            return (vals - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(vals)

        # Align lengths
        n = min(len(v) for v in method_sims.values())
        combined = np.zeros(n)
        for m_sims in method_sims.values():
            combined = combined + normalize(m_sims[:n])
        combined = combined / len(method_sims)

        any_labels = next(iter(method_labels.values()))[:n]
        res_metrics["ensemble"] = calculate_classifier_metrics(
            combined[any_labels == 1], combined[any_labels == 0]
        )

    # ── Counts ────────────────────────────────────────────────────────────
    any_labels = next(iter(method_labels.values())) if method_labels else np.array([])
    res_metrics["counts"] = {
        "presence":   int(np.sum(any_labels == 1)),
        "background": int(np.sum(any_labels == 0)),
    }

    return res_metrics




def run_parallel_cv(df, ecological_vars, class_property='present', scale=10, year=2025, n_folds=10, train_methods=None, eval_methods=None):
    """
    10-fold spatial cross-validation.

    KEY ARCHITECTURE: embeddings are sampled from GEE exactly ONCE for all
    training+test coordinates combined.  The resulting FeatureCollection
    (with fold assignments and embedding properties) lives on the GEE server.
    Each fold simply filters it server-side — no re-upload, no re-sampling.

    Per fold:
      - centroid: reduceColumns(mean)  on train subset  → 64 numbers
      - ridge:    reduceColumns(ridge) on train subset  → 65 numbers
      - scoring:  dot product mapped over test subset   → N scalars
    Total sampleRegions calls: 1 (regardless of n_folds or n_methods).
    """
    import ee

    df_f = assign_spatial_folds(df, n_folds=n_folds)

    ASSET_PATH      = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    EMB_COLS        = [f"A{i:02d}" for i in range(64)]
    LABEL_RIDGE_COL = 'cv_ridge_label'   # +1 / -1  for ridgeRegression
    LABEL_CENT_COL  = 'cv_cent_label'    # +1 /  0  for centroid / scoring
    FOLD_COL        = 'cv_fold'
    MP_CHUNK        = 5000

    # Label encoding
    df_f[LABEL_RIDGE_COL] = np.where(df_f[class_property] == 1,  1.0, -1.0)
    df_f[LABEL_CENT_COL]  = np.where(df_f[class_property] == 1,  1.0,  0.0)
    df_f[FOLD_COL]        = df_f['fold']

    # Ensure year column — fill NaN (e.g. from background points that have no year)
    if 'year' not in df_f.columns:
        df_f['year'] = year
    else:
        df_f['year'] = df_f['year'].fillna(year).astype(int)

    # ------------------------------------------------------------------
    # 1. Upload ALL coords + fold + labels as compact MultiPoint features
    #    (grouped by year × ridge_label × fold → minimal GEE Features)
    # ------------------------------------------------------------------
    sys.stderr.write(
        f"CV: uploading {len(df_f):,} coordinate pairs to GEE once "
        f"(year × label × fold grouping) ...\n"
    )
    gee_features = []
    for (yr, rl, fi), grp in df_f.groupby(['year', LABEL_RIDGE_COL, FOLD_COL]):
        coords = grp[['longitude', 'latitude']].values.tolist()
        cl = 1.0 if rl > 0 else 0.0
        for i in range(0, len(coords), MP_CHUNK):
            geom  = ee.Geometry.MultiPoint(coords[i : i + MP_CHUNK])
            props = {
                LABEL_RIDGE_COL: float(rl),
                LABEL_CENT_COL:  float(cl),
                FOLD_COL:        int(fi),
            }
            gee_features.append(ee.Feature(geom, props).set('year', int(yr)))

    upload_fc = ee.FeatureCollection(gee_features)

    # ------------------------------------------------------------------
    # 2. Sample Alpha Earth embeddings ONCE for all coordinates
    #    geometries=False is fine — after sampling, each point's embedding
    #    is stored as properties on the FC features (server-side).
    #    The FC is filtered per-fold without any re-uploading.
    # ------------------------------------------------------------------
    sys.stderr.write("CV: sampling Alpha Earth embeddings (once for all folds) ...\n")

    all_years   = sorted(df_f['year'].unique())
    sampled_fcs = []
    for yr in all_years:
        yr_fc  = upload_fc.filter(ee.Filter.eq('year', int(yr)))
        yr_img = (
            ee.ImageCollection(ASSET_PATH)
            .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))
            .mosaic()
            .select(EMB_COLS)
        )
        sampled = yr_img.sampleRegions(
            collection=yr_fc,
            properties=[LABEL_RIDGE_COL, LABEL_CENT_COL, FOLD_COL],
            scale=scale,
            geometries=False   # no geometry needed; embeddings stored as props
        ).filter(ee.Filter.notNull(['A00']))
        sampled_fcs.append(sampled)

    all_sampled_fc = ee.FeatureCollection(sampled_fcs).flatten()
    sys.stderr.write("CV: embeddings sampled. Running fold evaluations ...\n")

    # ------------------------------------------------------------------
    # 3. Evaluate each fold using the shared server-side FC
    # ------------------------------------------------------------------
    sys.stderr.write(f"Spatial K-Means Folding ({n_folds} clusters):\n")
    for fi in range(n_folds):
        n_fi = len(df_f[df_f[FOLD_COL] == fi])
        sys.stderr.write(f"  Fold {fi}: {n_fi} points\n")

    res = []
    for i in range(n_folds):
        sys.stderr.write(f"Processing 10-fold CV: Fold {i+1}/{n_folds}...\n")
        fold_res = run_cv_fold(
            i, all_sampled_fc, None,
            EMB_COLS, LABEL_RIDGE_COL, LABEL_CENT_COL, FOLD_COL, scale,
            train_methods=train_methods, eval_methods=eval_methods
        )
        if fold_res:
            res.append(fold_res)

    if not res:
        sys.stderr.write("Warning: All CV folds failed or returned no data.\n")
        return {m: {'cbi': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0} for m in (eval_methods or ["centroid", "ridge", "ensemble"])}

    avg = {}
    for m in (eval_methods or ['centroid', 'ridge', 'ensemble']):
        valid_folds = [r for r in res if r['counts']['presence'] > 0 and m in r] or [r for r in res if m in r]
        if valid_folds:
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

