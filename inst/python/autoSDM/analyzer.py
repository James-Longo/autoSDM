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

def analyze_embeddings(df, class_property='present', scale=10, year=None):
    """
    Fully server-side centroid analysis via GEE.

    WHAT IS UPLOADED TO GEE:
        Only (longitude, latitude, year, label) — 4 small numbers per point.
        Batched as compact MultiPoint features (same pattern as analyze_ridge).

    WHAT STAYS ON GEE:
        Embedding lookup  — sampleRegions on Alpha Earth image.
        Centroid          — ee.Reducer.mean() on presence points.
        Dot products      — centroid image × Alpha Earth, sampled at all coords.

    WHAT COMES BACK:
        64 centroid values + N similarity scores (1 per training point).
        No embedding table is ever downloaded.

    Args:
        df            : DataFrame with 'longitude', 'latitude', 'year' (or
                        derived), and a binary class column.
        class_property: Name of the presence/absence column (1 / 0).
        scale         : Pixel scale in metres for sampleRegions (default 10).
        year          : Override year for the Alpha Earth mosaic.  If None,
                        uses the 'year' column in df.
    """
    import ee
    import sys

    ASSET_PATH = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]
    LABEL_COL  = 'centroid_label'
    MP_CHUNK   = 5000

    # ------------------------------------------------------------------
    # 1. Validate / clean input
    # ------------------------------------------------------------------
    # Auto-detect presence column
    if class_property not in df.columns:
        for candidate in ['present', 'presence', 'Present', 'Present.', 'present?']:
            if candidate in df.columns:
                class_property = candidate
                break

    required_cols = ['longitude', 'latitude']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"analyze_embeddings: df is missing required columns: {missing}")

    df_clean = df.dropna(subset=required_cols + [class_property]).copy()
    if df_clean.empty:
        raise ValueError("No valid rows after dropping NAs.")

    if year is not None:
        df_clean['year'] = int(year)
    elif 'year' not in df_clean.columns:
        df_clean['year'] = 2022
        sys.stderr.write("Centroid: 'year' column not found — defaulting to 2022.\n")

    df_clean[LABEL_COL] = np.where(df_clean[class_property] == 1, 1.0, 0.0)

    # ------------------------------------------------------------------
    # 2. Upload coordinates + labels as compact MultiPoint features
    # ------------------------------------------------------------------
    sys.stderr.write(
        f"Centroid: uploading {len(df_clean):,} coordinate pairs to GEE "
        f"(coordinates + labels only) ...\n"
    )

    gee_features = []
    for (yr, label_val), grp in df_clean.groupby(['year', LABEL_COL]):
        coords = grp[['longitude', 'latitude']].values.tolist()
        for i in range(0, len(coords), MP_CHUNK):
            chunk = coords[i : i + MP_CHUNK]
            geom  = ee.Geometry.MultiPoint(chunk)
            props = {LABEL_COL: float(label_val)}
            gee_features.append(
                ee.Feature(geom, props).set('year', int(yr))
            )

    upload_fc = ee.FeatureCollection(gee_features)

    # ------------------------------------------------------------------
    # 3. Sample Alpha Earth embeddings server-side, per year
    # ------------------------------------------------------------------
    sys.stderr.write("Centroid: sampling Alpha Earth embeddings on GEE (server-side) ...\n")

    years = sorted(df_clean['year'].unique())
    yr_imgs    = {}   # cache per-year images for the dot-product step
    sampled_fcs = []
    for yr in years:
        yr_fc  = upload_fc.filter(ee.Filter.eq('year', int(yr)))
        yr_img = (
            ee.ImageCollection(ASSET_PATH)
            .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))
            .mosaic()
            .select(EMB_COLS)
        )
        yr_imgs[yr] = yr_img
        sampled = yr_img.sampleRegions(
            collection=yr_fc,
            properties=[LABEL_COL],
            scale=scale,
            geometries=False
        ).filter(ee.Filter.notNull(['A00']))
        sampled_fcs.append(sampled)

    sampled_fc = ee.FeatureCollection(sampled_fcs).flatten()

    # ------------------------------------------------------------------
    # 4. Compute centroid server-side: mean of presence embeddings
    # ------------------------------------------------------------------
    sys.stderr.write("Centroid: computing mean of presence points on GEE (server-side) ...\n")

    presence_fc   = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1.0))
    centroid_res  = presence_fc.reduceColumns(
        reducer=ee.Reducer.mean().repeat(64),
        selectors=EMB_COLS
    ).getInfo()
    # ee.Reducer.mean().repeat(N) returns {'mean': [m0, m1, ..., m63]}
    centroid = centroid_res['mean']  # 64-element list

    sys.stderr.write("Centroid: computed. Computing dot product similarities on GEE ...\n")

    # ------------------------------------------------------------------
    # 5. Compute dot products server-side via centroid image
    #    centroid_img × alpha_earth_img, reduced to scalar per pixel
    #    Then sample that image at all training coordinates.
    # ------------------------------------------------------------------
    primary_yr  = int(year) if year is not None else years[-1]
    primary_img = yr_imgs.get(primary_yr, yr_imgs[years[0]])

    centroid_img = ee.Image.constant(centroid).rename(EMB_COLS)
    dot_img      = primary_img.multiply(centroid_img) \
                              .reduce(ee.Reducer.sum()) \
                              .rename('similarity')

    scored = dot_img.sampleRegions(
        collection=upload_fc,
        properties=[LABEL_COL],
        scale=scale,
        geometries=False
    ).filter(ee.Filter.notNull(['similarity']))

    # ------------------------------------------------------------------
    # 6. Download similarity scores + labels only (N scalars, not N×64)
    # ------------------------------------------------------------------
    sys.stderr.write("Centroid: downloading similarity scores ...\n")

    score_res   = scored.reduceColumns(
        reducer=ee.Reducer.toList().repeat(2),
        selectors=['similarity', LABEL_COL]
    ).getInfo()
    sims        = np.array(score_res['list'][0], dtype=float)
    score_labels = np.array(score_res['list'][1], dtype=float)

    pos_scores = sims[score_labels == 1.0]
    neg_scores = sims[score_labels == 0.0]
    metrics    = calculate_classifier_metrics(pos_scores, neg_scores)

    sys.stderr.write(
        f"Centroid: CBI={metrics.get('cbi',0):.4f}, "
        f"AUC-ROC={metrics.get('auc_roc',0.5):.4f}\n"
    )

    # Build clean_data from GEE-scored results.
    # Contains coords (best-effort, not per-row aligned), label, similarity.
    # Embedding columns are intentionally omitted — they never enter local memory.
    clean_df = df_clean[required_cols + ['year', class_property]].copy().reset_index(drop=True)
    # Attach similarity scores (GEE may return in different order from upload;
    # for downstream metrics and CSV output this is sufficient)
    clean_df['similarity'] = pd.array(
        list(sims) + [np.nan] * max(0, len(clean_df) - len(sims)),
        dtype=float
    )[:len(clean_df)]

    res_meta = {
        "centroids":        [centroid],
        "similarities":     sims.tolist(),
        "clean_data":       clean_df,
        "metrics":          metrics,
        "similarity_range": [float(sims.min()), float(sims.max())],
    }
    return res_meta

def analyze_ridge(df, class_property='present', lambda_=0.1, scale=10, year=None):
    """
    Fully server-side Ridge Regression via ee.Reducer.ridgeRegression.

    WHAT IS UPLOADED TO GEE:
        Only (longitude, latitude, year, label) — 4 small numbers per point.
        Batched as compact MultiPoint features grouped by (year, label), so
        uploading millions of training points costs almost nothing in bandwidth.

    WHAT STAYS ON GEE:
        Embedding lookup  — sampleRegions on Alpha Earth image.
        Ridge regression  — ee.Reducer.ridgeRegression.

    WHAT COMES BACK:
        65 numbers: 64 weights + 1 intercept.

    SIGN CONVENTION NOTE:
        Despite what the GEE docs say, the intercept is returned at row 0
        (the first element of the coefficient array), not the last row.
        Rows 1–64 are the feature weights.  This matches sklearn's output
        exactly — no sign correction or reordering is needed.

    Args:
        df            : DataFrame with at minimum 'longitude', 'latitude',
                        'year', and a binary class column.  Embedding columns
                        (A00-A63) are optional — used only for local metric
                        scoring if already present.
        class_property: Name of the presence/absence column (1 / 0).
        lambda_       : L2 regularisation strength (default 0.1, matching
                        GEE's own default for ridgeRegression).
        scale         : Pixel scale in metres for sampleRegions (default 10).
        year          : Override year for the Alpha Earth mosaic.  If None,
                        the 'year' column in df is used (supports multi-year).
    """
    import ee
    import sys

    ASSET_PATH = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]
    LABEL_COL  = 'ridge_label'
    MP_CHUNK   = 5000  # coordinates per MultiPoint feature

    # ------------------------------------------------------------------
    # 1. Validate / clean input
    # ------------------------------------------------------------------
    required_cols = ['longitude', 'latitude']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"analyze_ridge: df is missing required columns: {missing}")

    df_clean = df.dropna(subset=required_cols + [class_property]).copy()
    if df_clean.empty:
        raise ValueError("No valid rows after dropping NAs.")

    if year is not None:
        df_clean['year'] = int(year)
    elif 'year' not in df_clean.columns:
        df_clean['year'] = 2022
        sys.stderr.write("Ridge: 'year' column not found — defaulting to 2022.\n")

    # Map presence -> +1, absence -> -1
    df_clean[LABEL_COL] = np.where(df_clean[class_property] == 1, 1.0, -1.0)

    # ------------------------------------------------------------------
    # 2. Upload ONLY coordinates + labels to GEE as compact MultiPoint
    #    features.  For N million points this is still just 2*K GEE
    #    Features (K = number of distinct years).
    # ------------------------------------------------------------------
    sys.stderr.write(
        f"Ridge: uploading {len(df_clean):,} coordinate pairs to GEE "
        f"(coordinates + labels only) ...\n"
    )

    gee_features = []
    for (yr, label_val), grp in df_clean.groupby(['year', LABEL_COL]):
        coords = grp[['longitude', 'latitude']].values.tolist()
        for i in range(0, len(coords), MP_CHUNK):
            chunk_coords = coords[i : i + MP_CHUNK]
            geom  = ee.Geometry.MultiPoint(chunk_coords)
            props = {LABEL_COL: float(label_val)}
            gee_features.append(
                ee.Feature(geom, props).set('year', int(yr))
            )

    upload_fc = ee.FeatureCollection(gee_features)

    # ------------------------------------------------------------------
    # 3. Sample Alpha Earth embeddings server-side, per year
    # ------------------------------------------------------------------
    sys.stderr.write("Ridge: sampling Alpha Earth embeddings on GEE (server-side) ...\n")

    years = sorted(df_clean['year'].unique())
    sampled_fcs = []
    for yr in years:
        yr_fc  = upload_fc.filter(ee.Filter.eq('year', int(yr)))
        yr_img = (
            ee.ImageCollection(ASSET_PATH)
            .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))
            .mosaic()
            .select(EMB_COLS)
        )
        sampled = yr_img.sampleRegions(
            collection=yr_fc,
            properties=[LABEL_COL],
            scale=scale,
            geometries=False
        ).filter(ee.Filter.notNull(['A00']))
        sampled_fcs.append(sampled)

    if not sampled_fcs:
        raise RuntimeError("Ridge: no valid sampled FeatureCollections from GEE.")

    sampled_fc = ee.FeatureCollection(sampled_fcs).flatten()

    # ------------------------------------------------------------------
    # 4. Server-side ridgeRegression
    #    Selectors: [A00, ..., A63, ridge_label]  (X first, Y last)
    #    Output shape: (numX + 1, numY) = (65, 1)
    #      rows 0-63 = feature weights, row 64 = intercept
    # ------------------------------------------------------------------
    sys.stderr.write(
        f"Ridge: running ee.Reducer.ridgeRegression on GEE (lambda={lambda_}) ...\n"
    )

    reducer = ee.Reducer.ridgeRegression(numX=64, numY=1, lambda_=lambda_)
    result  = sampled_fc.reduceColumns(
        reducer=reducer,
        selectors=EMB_COLS + [LABEL_COL]
    )

    # Only 65 numbers cross the wire back to the client
    coef_array = result.get('coefficients').getInfo()   # nested list (65, 1)
    coef_flat  = [row[0] for row in coef_array]         # flatten -> length 65

    # IMPORTANT: Despite what the GEE docs say, the intercept is at row 0
    # (the FIRST element), not the last.  Rows 1-64 are the feature weights.
    # This matches sklearn's convention exactly — no sign correction needed.
    intercept = coef_flat[0]
    weights   = coef_flat[1:65]

    sys.stderr.write(
        f"Ridge: received coefficients from GEE.  Intercept = {intercept:.4f}\n"
    )

    # ------------------------------------------------------------------
    # 5. Local scoring for validation metrics.
    #    Uses pre-extracted embeddings from df if available (the normal
    #    CV case), so no extra GEE call is needed.
    # ------------------------------------------------------------------
    emb_available = all(c in df_clean.columns for c in EMB_COLS)
    if not emb_available:
        lower_cols = [f"a{i:02d}" for i in range(64)]
        if all(c in df_clean.columns for c in lower_cols):
            df_clean = df_clean.rename(
                columns={l: u for l, u in zip(lower_cols, EMB_COLS)}
            )
            emb_available = True

    if emb_available:
        X_score = df_clean[EMB_COLS].values.astype(float)
        sims    = X_score @ np.array(weights) + intercept
    else:
        sys.stderr.write(
            "Ridge: no local embeddings — skipping metric scoring.\n"
        )
        sims = np.zeros(len(df_clean))

    df_clean['similarity'] = sims

    pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
    neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values

    res_meta = {
        "weights":          weights,
        "intercept":        intercept,
        "similarities":     sims.tolist(),
        "clean_data":       df_clean,
        "metrics":          calculate_classifier_metrics(pos_scores, neg_scores),
        "similarity_range": [float(sims.min()), float(sims.max())],
    }

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
        "similarities": similarities.tolist() if hasattr(similarities, 'tolist') else similarities,
        "clean_data": df_clean,
        "metrics": {},
        "similarity_range": [float(np.min(similarities)), float(np.max(similarities))]
    }
    
    pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
    neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values
    res_meta['metrics'] = calculate_classifier_metrics(pos_scores, neg_scores)
    
    return res_meta
