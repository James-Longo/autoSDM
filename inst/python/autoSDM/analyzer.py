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


# ─────────────────────────────────────────────────────────────────────────────
# GEE classifier / reducer registry
# ─────────────────────────────────────────────────────────────────────────────

# Methods that use ee.Classifier.*  (output mode: PROBABILITY)
GEE_CLASSIFIER_METHODS = {
    "rf":               lambda p: _build_classifier("smileRandomForest",     {"numberOfTrees": p.get("numberOfTrees", 100), **p}),
    "gbt":              lambda p: _build_classifier("smileGradientTreeBoost",{"numberOfTrees": p.get("numberOfTrees", 100), **p}),
    "cart":             lambda p: _build_classifier("smileCart",             p),
    "svm":              lambda p: _build_classifier("libsvm",                p),
    "maxent":           lambda p: _build_classifier("amnhMaxent",            p),
}

# Methods that use ee.Reducer.* via reduceColumns on an FC
# Each entry: (reducer_factory, label_encoding)
#   label_encoding: "binary"  (+1/-1)
#                   "presence" (1/0)
GEE_REDUCER_METHODS = {
    "centroid":       ("mean",                   "presence"),
    "ridge":          ("ridgeRegression",        "binary"),
    "linear":         ("linearRegression",       "binary"),
    "robust_linear":  ("robustLinearRegression", "binary"),
}

ALL_METHODS = list(GEE_CLASSIFIER_METHODS) + list(GEE_REDUCER_METHODS)


def _build_classifier(gee_name, params):
    """
    Return an ee.Classifier instance for the given GEE method name and params dict,
    with PROBABILITY output mode set.
    """
    import ee
    factory = getattr(ee.Classifier, gee_name)
    # Always splat params — even an empty dict is fine; defaults are already merged in
    clf = factory(**params) if params else factory()
    return clf.setOutputMode("PROBABILITY")


def analyze_method(df, method, params=None, class_property="present",
                   scale=10, year=None):
    """
    Unified GEE train+score function for any supported method.

    Supported classifiers (``GEE_CLASSIFIER_METHODS``):
        centroid, rf, gbt, cart, svm, maxent

    Supported regression reducers (``GEE_REDUCER_METHODS``):
        ridge, linear, robust_linear

    WHAT IS UPLOADED:
        Only (longitude, latitude, year, label) as compact MultiPoint features.

    WHAT COMES BACK:
        Per-point similarity scores + training metrics.

    Returns dict with keys:
        method, params, metrics, similarity_range, similarities, clean_data
        + 'weights' and 'intercept' for reducer methods
    """
    import ee, sys
    params = params or {}
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]

    # ── 0. Local Fast-Path for 'mean' ─────────────────────────────────────
    if method == "mean":
        df_clean = df.dropna(subset=EMB_COLS + [class_property]).copy()
        presence_embs = df_clean[df_clean[class_property] == 1][EMB_COLS].values
        if len(presence_embs) == 0:
            raise ValueError("No presence points found for local mean calculation.")

        mean_vec = np.mean(presence_embs, axis=0)
        similarities = np.dot(df_clean[EMB_COLS].values, mean_vec)
        df_clean['similarity'] = similarities

        pos_scores = df_clean[df_clean[class_property] == 1]['similarity'].values
        neg_scores = df_clean[df_clean[class_property] == 0]['similarity'].values

        return {
            "method":           "mean",
            "params":           {},
            "weights":          mean_vec.tolist(),
            "intercept":        0.0,
            "metrics":          calculate_classifier_metrics(pos_scores, neg_scores),
            "similarity_range": [float(np.min(similarities)), float(np.max(similarities))],
            "similarities":     similarities.tolist(),
            "clean_data":       df_clean,
        }

    ASSET_PATH = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    MP_CHUNK   = 5000

    # ── 1. Validate / clean input ─────────────────────────────────────────
    if class_property not in df.columns:
        for cand in ["present", "presence", "Present", "present?"]:
            if cand in df.columns:
                class_property = cand
                break

    df_clean = df.dropna(subset=["longitude", "latitude", class_property]).copy()
    if df_clean.empty:
        raise ValueError("analyze_method: no valid rows after dropping NAs.")

    if year is not None:
        df_clean["year"] = int(year)
    elif "year" not in df_clean.columns:
        raise ValueError(f"analyze_method: 'year' parameter or column must be provided (no default fallback).")

    # ── 2. Choose label encoding ──────────────────────────────────────────
    is_classifier = method in GEE_CLASSIFIER_METHODS
    is_reducer    = method in GEE_REDUCER_METHODS

    if not is_classifier and not is_reducer:
        raise ValueError(f"Unknown method '{method}'. Choose from: {ALL_METHODS}")

    if is_reducer:
        _, label_enc = GEE_REDUCER_METHODS[method]
        if label_enc == "binary":
            LABEL_COL = "label"
            df_clean[LABEL_COL] = np.where(df_clean[class_property] == 1, 1.0, -1.0)
        else:  # presence
            LABEL_COL = "label"
            df_clean[LABEL_COL] = np.where(df_clean[class_property] == 1, 1.0, 0.0)
    else:  # classifier — GEE needs 0/1 integer labels
        LABEL_COL = "label"
        df_clean[LABEL_COL] = np.where(df_clean[class_property] == 1, 1, 0).astype(int)

    # ── 3. Upload coordinates + labels as flat lists to avoid AST limits ─────
    sys.stderr.write(
        f"{method}: uploading {len(df_clean):,} coordinate pairs to GEE "
        f"(coordinates + labels only) ...\n"
    )
    
    cols_to_upload = ["longitude", "latitude", "year", LABEL_COL]
    
    data_flat = df_clean[cols_to_upload].copy()
    if class_property != LABEL_COL:
        data_flat[class_property] = df_clean[class_property]
        cols_to_upload.append(class_property)
        
    data_list = data_flat.values.tolist()
    
    chunk_size = 5000
    fc_chunks = []
    
    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i:i+chunk_size]
        ee_chunk = ee.List(chunk)
        
        def make_feature(row):
            r = ee.List(row)
            lon = ee.Number(r.get(0))
            lat = ee.Number(r.get(1))
            yr = ee.Number(r.get(2))
            lbl = ee.Number(r.get(3))
            
            geom = ee.Geometry.Point([lon, lat])
            
            props = ee.Dictionary({
                'longitude': lon,
                'latitude': lat,
                'year': yr,
                LABEL_COL: lbl.int() if is_classifier else lbl.float()
            })
            
            if class_property != LABEL_COL:
                props = props.set(class_property, r.get(4))
                
            return ee.Feature(geom, props)
        
        fc_chunks.append(ee.FeatureCollection(ee_chunk.map(make_feature)))

    upload_fc = ee.FeatureCollection(fc_chunks).flatten()


    # ── 4. Sample Alpha Earth embeddings server-side (once) ───────────────
    sys.stderr.write(f"{method}: sampling Alpha Earth embeddings on GEE ...\n")
    years = sorted(df_clean["year"].unique())
    sampled_fcs = []
    for yr in years:
        yr_fc  = upload_fc.filter(ee.Filter.eq("year", int(yr)))
        yr_img = (
            ee.ImageCollection(ASSET_PATH)
            .filter(ee.Filter.calendarRange(int(yr), int(yr), "year"))
            .mosaic()
            .select(EMB_COLS)
        )
        sampled = yr_img.sampleRegions(
            collection=yr_fc,
            properties=[LABEL_COL, "longitude", "latitude", "year", class_property],
            scale=scale,
            geometries=False,
            tileScale=16,
        ).filter(ee.Filter.notNull(["A00"]))
        sampled_fcs.append(sampled)

    sampled_fc = ee.FeatureCollection(sampled_fcs).flatten()



    # ── 5a. Train GEE classifier ──────────────────────────────────────────
    if is_classifier:
        sys.stderr.write(f"{method}: training ee.Classifier.{GEE_CLASSIFIER_METHODS[method].__name__ if hasattr(GEE_CLASSIFIER_METHODS[method], '__name__') else method} ...\n")
        clf = GEE_CLASSIFIER_METHODS[method](params)
        trained = sampled_fc.classify(
            clf.train(
                features=sampled_fc,
                classProperty=LABEL_COL,
                inputProperties=EMB_COLS,
            )
        )
        # Maxent uses 'probability' property, others use 'classification'
        score_col = "probability" if method == "maxent" else "classification"
        
        # Retrieve probability scores and metadata
        result = trained.reduceColumns(
            ee.Reducer.toList().repeat(5),
            selectors=[score_col, LABEL_COL, "longitude", "latitude", "year"]
        ).getInfo()

        if not result["list"] or not result["list"][0]:
            raise ValueError(f"analyze_method: GEE classifier '{method}' returned no results. "
                             "This can happen if training fails or if all test points are in masked pixels.")

        sims   = np.array(result["list"][0], dtype=float)
        # libsvm in GEE returns probability of class 0. We want class 1 (presence).
        if method == "svm":
            sims = 1.0 - sims
            
        labels = np.array(result["list"][1], dtype=float)
        lons   = np.array(result["list"][2], dtype=float)
        lats   = np.array(result["list"][3], dtype=float)
        yrs    = np.array(result["list"][4], dtype=int)

        weights   = None
        intercept = 0.0


    # ── 5b. Train GEE reducer ─────────────────────────────────────────────
    elif is_reducer:
        reducer_name, _ = GEE_REDUCER_METHODS[method]
        lambda_ = params.get("lambda_", 0.1)
        
        # For regression reducers, we add a constant 1.0 column to handle the intercept (bias).
        # GEE's ridgeRegression treats the first X column as unregularized (bias).
        fc_reg = sampled_fc.map(lambda f: f.set("constant", 1.0))
        
        sys.stderr.write(
            f"{method}: running ee.Reducer.{reducer_name} on GEE "
            f"(lambda={lambda_ if 'ridge' in reducer_name else 'N/A'}) ...\n"
        )
        
        if reducer_name == "ridgeRegression":
            # GEE ridgeRegression native intercept is the first element of a (numX+1) result.
            reducer = ee.Reducer.ridgeRegression(numX=64, numY=1, lambda_=lambda_)
            result  = sampled_fc.reduceColumns(
                reducer=reducer, selectors=EMB_COLS + [LABEL_COL]
            ).getInfo()
            # coefs[0] is intercept, coefs[1:65] are weights for A00-A63.
            coef_flat = [float(row[0]) for row in result.get("coefficients")]
            intercept = coef_flat[0]
            weights   = coef_flat[1:65]
        elif reducer_name in ("linearRegression", "robustLinearRegression"):
            fc_reg = sampled_fc.map(lambda f: f.set("constant", 1.0))
            factory = getattr(ee.Reducer, reducer_name)
            reducer = factory(numX=65, numY=1)
            result  = fc_reg.reduceColumns(
                reducer=reducer, selectors=["constant"] + EMB_COLS + [LABEL_COL]
            ).getInfo()
            # Pos 0 is coefficient for 'constant' (the intercept)
            coef_flat = [float(row[0]) for row in result.get("coefficients")]
            intercept = coef_flat[0]
            weights   = coef_flat[1:65]


        elif reducer_name == "mean":
            # Centroid similarity: compute mean embedding of presence points
            presence_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1))
            centroid_res = presence_fc.reduceColumns(
                reducer=ee.Reducer.mean().repeat(64), selectors=EMB_COLS
            ).getInfo()
            weights = [float(w) for w in centroid_res["mean"]]
            intercept = 0.0
        elif reducer_name == "linearFit":
            # linearFit only takes 1 predictor; we project onto the PC1 direction
            # by using the mean embedding as a single linear predictor weight
            reducer = ee.Reducer.linearFit()
            # Use centroid dot-product distance as the single predictor
            presence_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1.0))
            centroid_res = presence_fc.reduceColumns(
                reducer=ee.Reducer.mean().repeat(64), selectors=EMB_COLS
            ).getInfo()
            centroid = centroid_res["mean"]
            # Score is the dot product (used as single X in linearFit)
            def add_dot(feat):
                s = ee.Number(0)
                for col, w in zip(EMB_COLS, centroid):
                    s = s.add(ee.Number(feat.get(col)).multiply(float(w)))
                return feat.set("dot", s)
            scored_fc = sampled_fc.map(add_dot)
            result = scored_fc.reduceColumns(
                ee.Reducer.linearFit(), selectors=["dot", LABEL_COL]
            ).getInfo()
            scale_  = result.get("scale", 1.0)
            offset_ = result.get("offset", 0.0)
            # Synthesize weights = scale * centroid, intercept = offset
            weights   = [float(w) * float(scale_) for w in centroid]
            intercept = float(offset_)
        else:
            raise ValueError(f"Unsupported reducer: {reducer_name}")

        sys.stderr.write(f"{method}: received coefficients. Intercept = {intercept:.4f}\n")

        # Score training data locally using weight vector
        X = df_clean[EMB_COLS].values.astype(float) if all(c in df_clean.columns for c in EMB_COLS) else None
        if X is not None:
            sims   = X @ np.array(weights) + intercept
            labels = df_clean[class_property].values.astype(float)
        else:
            sims   = np.zeros(len(df_clean))
            labels = df_clean[class_property].values.astype(float)

    # ── 6. Compute metrics ────────────────────────────────────────────────
    pos_scores = sims[labels == 1]
    neg_scores = sims[labels == 0]
    metrics    = calculate_classifier_metrics(pos_scores, neg_scores)
    sys.stderr.write(
        f"{method}: CBI={metrics.get('cbi', 0):.4f}, "
        f"AUC-ROC={metrics.get('auc_roc', 0.5):.4f}\n"
    )

    # ── 7. Build clean_data ───────────────────────────────────────────────
    # Reconstruct from GEE response for classifiers (handles dropped points)
    if is_classifier:
        clean_data = pd.DataFrame({
            class_property: (labels == 1).astype(int),
            "longitude":    lons,
            "latitude":     lats,
            "year":         yrs,
            "similarity":   sims,
        })
    else:
        clean_data = df_clean.copy()
        clean_data["similarity"] = sims


    result_dict = {
        "method":           method,
        "params":           params,
        "metrics":          metrics,
        "similarity_range": [float(sims.min()), float(sims.max())],
        "similarities":     sims.tolist(),
        "clean_data":       clean_data,
    }
    if weights is not None:
        result_dict["weights"]   = weights
        result_dict["intercept"] = intercept

    return result_dict


def predict_method(train_df, test_df, method, params=None, class_property="present", scale=10, year=None):
    """
    Train a model on train_df (coordinates only) and predict on test_df (coordinates + optional embeddings).
    This is used for accurate point predictions of GEE classifiers.
    """
    import ee, sys
    params = params or {}
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]

    is_classifier = method in GEE_CLASSIFIER_METHODS
    is_reducer    = method in GEE_REDUCER_METHODS

    if not is_classifier and not is_reducer:
        raise ValueError(f"predict_method: Unknown method '{method}'.")

    # 1. Prepare Training Data & Test Data IDs
    train_clean = train_df.dropna(subset=["longitude", "latitude", class_property]).copy()
    if is_reducer:
        _, label_enc = GEE_REDUCER_METHODS[method]
        LABEL_COL = "label"
        if label_enc == "binary":
            train_clean[LABEL_COL] = np.where(train_clean[class_property] == 1, 1.0, -1.0)
        else:
            train_clean[LABEL_COL] = np.where(train_clean[class_property] == 1, 1.0, 0.0)
    else:
        LABEL_COL = "label"
        train_clean[LABEL_COL] = np.where(train_clean[class_property] == 1, 1, 0).astype(int)

    test_df = test_df.copy()
    test_df['_point_id'] = np.arange(len(test_df))

    train_has_embs = all([c in train_clean.columns for c in EMB_COLS])
    test_has_embs = all([c in test_df.columns for c in EMB_COLS])

    # 2. Upload both to GEE
    def df_to_fc(df, label_col=None, include_embs=False):
        features = []
        for _, r in df.iterrows():
            geom = ee.Geometry.Point([r['longitude'], r['latitude']])
            props = {}
            if 'year' in df.columns:
                props['year'] = int(r['year'])
            elif year is not None:
                props['year'] = int(year)
            else:
                raise ValueError(f"predict_method({method}): 'year' must be provided either as a column or an argument.")
            if '_point_id' in df.columns:
                props['_point_id'] = int(r['_point_id'])
            if label_col: props[label_col] = float(r[label_col]) if is_reducer else int(r[label_col])
            if include_embs:
                for col in EMB_COLS:
                    props[col] = float(r[col])
            features.append(ee.Feature(geom, props))
        return ee.FeatureCollection(features)

    # CHUNKED UPLOAD
    def upload_large_fc(df, label_col=None, include_embs=False):
        chunks = []
        for i in range(0, len(df), 500):
            chunk = df.iloc[i:i+500]
            chunks.append(df_to_fc(chunk, label_col, include_embs))
        return ee.FeatureCollection(chunks).flatten()

    train_fc = upload_large_fc(train_clean, LABEL_COL, include_embs=train_has_embs)
    test_fc  = upload_large_fc(test_df, include_embs=test_has_embs)

    # 3. Sample Training Data on GEE (if needed)
    if train_has_embs:
        train_sampled = train_fc
    else:
        sys.stderr.write(f"predict_method({method}): sampling training embeddings on GEE...\n")
        if 'year' in train_clean.columns:
            sample_year = int(train_clean['year'].iloc[0])
        elif year is not None:
            sample_year = int(year)
        else:
            raise ValueError(f"predict_method({method}): 'year' must be provided either as a column or an argument.")
        img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")\
            .filter(ee.Filter.calendarRange(sample_year, sample_year, 'year'))\
            .mosaic().select(EMB_COLS)
        train_sampled = img.sampleRegions(collection=train_fc, properties=[LABEL_COL], scale=scale, geometries=False, tileScale=16)

    # 4. Train
    if is_classifier:
        clf = GEE_CLASSIFIER_METHODS[method](params)
        model = clf.train(features=train_sampled, classProperty=LABEL_COL, inputProperties=EMB_COLS)
    else:
        reducer_name, _ = GEE_REDUCER_METHODS[method]
        model = train_sampled.reduceColumns(
            reducer=getattr(ee.Reducer, reducer_name)(numInputBands=64),
            selectors=EMB_COLS + [LABEL_COL]
        )
        return None # Reducers handled locally

    # 5. Classify Testing Data
    if test_has_embs:
        test_sampled = test_fc
    else:
        sys.stderr.write(f"predict_method({method}): classifying testing points on GEE...\n")
        if 'year' in test_df.columns:
            sample_year = int(test_df['year'].iloc[0])
        elif year is not None:
            sample_year = int(year)
        else:
            raise ValueError(f"predict_method({method}): 'year' must be provided either as a column or an argument.")
        img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")\
            .filter(ee.Filter.calendarRange(sample_year, sample_year, 'year'))\
            .mosaic().select(EMB_COLS)
        test_sampled = img.sampleRegions(collection=test_fc, properties=['_point_id'], scale=scale, geometries=False, tileScale=16)
    
    pred_col = "classification" if method != "maxent" else "probability"
    results = test_sampled.classify(model).reduceColumns(ee.Reducer.toList(2), ["_point_id", pred_col]).getInfo()
    
    # 6. Reconstruct predictions in same order
    out_dict = dict(results['list'])
    
    sims = []
    for pid in test_df['_point_id']:
        val = out_dict.get(pid, np.nan)
        if val is not None and not np.isnan(val):
            val = float(val)
            if method == "svm":
                val = 1.0 - val
        else:
            val = np.nan
        sims.append(val)

    return np.array(sims)



