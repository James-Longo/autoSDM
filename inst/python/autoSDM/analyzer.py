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
                   scale=None, year=None):
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
    if scale is None:
        raise ValueError("analyze_method: 'scale' (resolution in meters) must be explicitly provided.")
    if year is None:
        raise ValueError("analyze_method: 'year' must be explicitly provided.")
    import ee, sys
    params = params or {}
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]


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

    # Regression weighting to balance classes 1:1
    pos_count = len(df_clean[df_clean[class_property] == 1])
    neg_count = len(df_clean[df_clean[class_property] == 0])
    sys.stderr.write(f"{method}: class counts: pos={pos_count}, neg={neg_count}\n")
    w_pos = 1.0
    w_neg = pos_count / neg_count if neg_count > 0 else 1.0
    df_clean["case_weight"] = np.where(df_clean[class_property] == 1, w_pos, w_neg)

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
    
    cols_to_upload = ["longitude", "latitude", "year", LABEL_COL, "case_weight"]
    
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
            yr  = ee.Number(r.get(2))
            lbl = ee.Number(r.get(3))
            w   = ee.Number(r.get(4))
            
            geom = ee.Geometry.Point([lon, lat])
            
            props = ee.Dictionary({
                'longitude': lon,
                'latitude': lat,
                'year': yr,
                LABEL_COL: lbl.int() if is_classifier else lbl.float(),
                'case_weight': w.float()
            })
            
            if class_property != LABEL_COL:
                props = props.set(class_property, r.get(5))
                
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
            properties=[LABEL_COL, "longitude", "latitude", "year", class_property, "case_weight"],
            scale=scale,
            geometries=False,
            tileScale=16,
        ).filter(ee.Filter.notNull(["A00"]))
        sampled_fcs.append(sampled)

    sampled_fc = ee.FeatureCollection(sampled_fcs).flatten()



    # ── 5a. Train GEE classifier ──────────────────────────────────────────
    if is_classifier:
        sys.stderr.write(f"{method}: balancing training set (1:1) for classifier...\n")
        # For classifiers that don't support weights, we subsample background to match presences
        pos_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1))
        neg_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 0))
        
        pos_count = pos_fc.size()
        
        # Subsample background points to match presence count (1:1 ratio)
        balanced_fc = pos_fc.merge(neg_fc.randomColumn().sort("random").limit(pos_count))
        
        sys.stderr.write(f"{method}: training ee.Classifier.{GEE_CLASSIFIER_METHODS[method].__name__ if hasattr(GEE_CLASSIFIER_METHODS[method], '__name__') else method} on balanced dataset ...\n")
        clf = GEE_CLASSIFIER_METHODS[method](params)
        trained = balanced_fc.classify(
            clf.train(
                features=balanced_fc,
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
        
        labels = np.array(result["list"][1], dtype=float)
        lons   = np.array(result["list"][2], dtype=float)
        lats   = np.array(result["list"][3], dtype=float)
        yrs    = np.array(result["list"][4], dtype=int)

        weights   = None
        intercept = 0.0


    # 5b. Train GEE reducer ─────────────────────────────────────────────
    elif is_reducer:
        reducer_name, _ = GEE_REDUCER_METHODS[method]
        
        sys.stderr.write(
            f"{method}: running ee.Reducer.{reducer_name} on GEE "
            f"(lambda={params.get('lambda_', 0.1) if 'ridge' in reducer_name else 'N/A'}) ...\n"
        )
        
        if reducer_name in ("ridgeRegression", "linearRegression", "robustLinearRegression"):
            if reducer_name == "ridgeRegression":
                # Ridge centering to avoid double-intercept issues
                # Calculate weighted means for X and Y
                means_res = sampled_fc.reduceColumns(
                    reducer=ee.Reducer.mean().repeat(65), # 64 EMB_COLS + 1 LABEL_COL
                    selectors=EMB_COLS + [LABEL_COL],
                    weightSelectors=["case_weight"] * 65
                ).getInfo()
                
                m_x = means_res["mean"][:64]
                m_y = means_res["mean"][64]
                
                def center_and_weight(f):
                    w = f.getNumber("case_weight")
                    sw = w.sqrt()
                    
                    # Center X
                    x_arr = f.toArray(EMB_COLS)
                    x_centered = x_arr.subtract(ee.Array(m_x))
                    
                    w_cols = [f"w_{c}" for c in EMB_COLS]
                    scaled_vals = x_centered.multiply(sw).toList()
                    props = ee.Dictionary.fromLists(w_cols, scaled_vals)
                    
                    # Center Y
                    y_centered = f.getNumber(LABEL_COL).subtract(m_y)
                    
                    return f.set(props).set({
                        "w_label": y_centered.multiply(sw)
                    })
                
                centered_fc = sampled_fc.map(center_and_weight)
                
                reducer = ee.Reducer.ridgeRegression(numX=64, numY=1, lambda_=params.get("lambda_", 0.1))
                selectors = [f"w_{c}" for c in EMB_COLS] + ["w_label"]
                
                sys.stderr.write(f"{method}: running centered ee.Reducer.ridgeRegression (weighted 1:1, lambda={params.get('lambda_', 0.1)})...\n")
                result = centered_fc.reduceColumns(reducer=reducer, selectors=selectors).getInfo()
                
                if not result or result.get("coefficients") is None:
                    sys.stderr.write(f"{method}: ERROR: GEE returned no coefficients (possibly singular matrix).\n")
                    intercept, weights = 0.0, [0.0] * 64
                else:
                    coefs = np.array(result.get("coefficients")).flatten()
                    # coefficients[0] is the ridge intercept for centered data
                    # We compute the final intercept: b0 = m_y - sum(weights * m_x) + GEE_intercept
                    weights = coefs[1:65].tolist() # GEE returns [intercept_centered, w0, w1, ...]
                    if len(weights) < 64: weights = weights + [0.0]*(64-len(weights))
                    
                    pred_at_mean = np.dot(weights, m_x)
                    intercept = m_y - pred_at_mean + float(coefs[0])
            else:
                # Linear
                def weight_inputs(f):
                    lbl = f.getNumber(LABEL_COL)
                    w = f.getNumber("case_weight")
                    sw = w.sqrt()
                    
                    # For linear, we add a constant 1.0 column to handle the intercept (bias).
                    # GEE's regression reducers treat the first X column as unregularized (bias).
                    
                    w_cols = [f"w_{c}" for c in EMB_COLS]
                    scaled_vals = f.toArray(EMB_COLS).multiply(sw).toList()
                    props = ee.Dictionary.fromLists(w_cols, scaled_vals)
                    
                    return f.set(props).set({
                        "w_constant": sw,
                        "w_label": lbl.multiply(sw)
                    })

                weighted_fc = sampled_fc.map(weight_inputs)
                
                reducer = getattr(ee.Reducer, reducer_name)(numX=65, numY=1)
                selectors = ["w_constant"] + [f"w_{c}" for c in EMB_COLS] + ["w_label"]

                sys.stderr.write(f"{method}: running ee.Reducer.{reducer_name} (weighted 1:1)...\n")
                result  = weighted_fc.reduceColumns(reducer=reducer, selectors=selectors).getInfo()
                
                if not result or result.get("coefficients") is None:
                     sys.stderr.write(f"{method}: ERROR: GEE returned no coefficients (possibly singular matrix).\n")
                     intercept, weights = 0.0, [0.0] * 64
                else:
                     coefs = np.array(result.get("coefficients")).flatten()
                     intercept = float(coefs[0])
                     weights   = coefs[1:65].tolist()
                     if len(weights) < 64:
                          weights = weights + [0.0] * (64 - len(weights))
            
            sys.stderr.write(f"{method}: weights sum_abs={np.sum(np.abs(weights)):.4f}, intercept={intercept:.4f}\n")

        elif reducer_name == "mean":
            # Centroid similarity: compute mean embedding of presence points
            presence_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1))
            res = presence_fc.reduceColumns(
                reducer=ee.Reducer.mean().repeat(64), 
                selectors=EMB_COLS
            ).getInfo()
            weights = [float(w) for w in res["mean"]]
            intercept = 0.0
        elif reducer_name == "linearFit":
            # Use presence-only mean embedding as the predictor weight
            presence_fc = sampled_fc.filter(ee.Filter.eq(LABEL_COL, 1))
            res = presence_fc.reduceColumns(
                reducer=ee.Reducer.mean().repeat(64), 
                selectors=EMB_COLS
            ).getInfo()
            centroid = res["mean"]
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

        # ── 5c. Score training data on GEE ────────────────────────────────────
        sys.stderr.write(f"{method}: calculating similarity scores on GEE (weights[0]={float(weights[0]):.4f}, intercept={float(intercept):.4f}) ...\n")
        
        # Vectorized scoring using ee.Array is significantly faster and more stable
        # weights_ee shape: [1, 64]
        weights_ee = ee.Array([float(w) for w in weights]).reshape([1, 64])
        intercept_ee = ee.Number(float(intercept))

        def score_feat(f):
            # Convert embedding properties to 1D array [64, 1]
            emb_arr = f.toArray(EMB_COLS).reshape([64, 1])
            # sim = (1,64) * (64,1) = (1,1) array
            sim = weights_ee.matrixMultiply(emb_arr).add(intercept_ee)
            # Get as scalar Number
            return f.set('similarity', sim.get([0, 0]))

        scored_fc = sampled_fc.map(score_feat)
        
        # Retrieve scores and labels
        res = scored_fc.reduceColumns(
            ee.Reducer.toList().repeat(5),
            selectors=['similarity', LABEL_COL, "longitude", "latitude", "year"]
        ).getInfo()
        
        sims   = np.array(res["list"][0], dtype=float)
        labels = np.array(res["list"][1], dtype=float)
        lons   = np.array(res["list"][2], dtype=float)
        lats   = np.array(res["list"][3], dtype=float)
        yrs    = np.array(res["list"][4], dtype=int)

    # ── 6. Compute metrics ────────────────────────────────────────────────
    if is_reducer and GEE_REDUCER_METHODS[method][1] == "binary":
        pos_scores = sims[labels > 0]
        neg_scores = sims[labels <= 0]
    else:
        pos_scores = sims[labels == 1]
        neg_scores = sims[labels == 0]
    metrics    = calculate_classifier_metrics(pos_scores, neg_scores)
    sys.stderr.write(
        f"{method}: CBI={metrics.get('cbi', 0):.4f}, "
        f"AUC-ROC={metrics.get('auc_roc', 0.5):.4f}\n"
    )

    clean_data = pd.DataFrame({
        class_property: (labels == 1).astype(int),
        "longitude":    lons,
        "latitude":     lats,
        "year":         yrs,
        "similarity":   sims,
    })


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


def predict_method(train_df, test_df, method, params=None, class_property="present", scale=None, year=None):
    """
    Train a model on train_df (coordinates only) and predict on test_df (coordinates + optional embeddings).
    This is used for accurate point predictions of GEE classifiers.
    """
    if scale is None:
        raise ValueError("predict_method: 'scale' (resolution in meters) must be explicitly provided.")
    if year is None:
        raise ValueError("predict_method: 'year' must be explicitly provided.")
    import ee, sys
    params = params or {}
    EMB_COLS   = [f"A{i:02d}" for i in range(64)]

    is_classifier = method in GEE_CLASSIFIER_METHODS
    is_reducer    = method in GEE_REDUCER_METHODS

    if not is_classifier and not is_reducer:
        raise ValueError(f"predict_method: Unknown method '{method}'.")

    # 1. Prepare Training Data & Test Data IDs
    train_clean = train_df.dropna(subset=["longitude", "latitude", class_property]).copy()
    
    # Regression weighting to balance classes 1:1
    pos_count = len(train_clean[train_clean[class_property] == 1])
    neg_count = len(train_clean[train_clean[class_property] == 0])
    w_pos = 1.0
    w_neg = pos_count / neg_count if neg_count > 0 else 1.0
    train_clean["case_weight"] = np.where(train_clean[class_property] == 1, w_pos, w_neg)

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

    # 2. Upload both to GEE
    def df_to_fc(df, label_col=None):
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
            if 'case_weight' in df.columns:
                props['case_weight'] = float(r['case_weight'])
            if label_col: props[label_col] = float(r[label_col]) if is_reducer else int(r[label_col])
            features.append(ee.Feature(geom, props))
        return ee.FeatureCollection(features)

    # CHUNKED UPLOAD
    def upload_large_fc(df, label_col=None):
        chunks = []
        for i in range(0, len(df), 500):
            chunk = df.iloc[i:i+500]
            chunks.append(df_to_fc(chunk, label_col))
        return ee.FeatureCollection(chunks).flatten()

    train_fc = upload_large_fc(train_clean, LABEL_COL)
    test_fc  = upload_large_fc(test_df)

    # 3. Sample Training Data on GEE
    sys.stderr.write(f"predict_method({method}): sampling training embeddings on GEE...\n")
    years = sorted(train_clean["year"].unique()) if 'year' in train_clean.columns else [int(year)] if year is not None else None
    if not years:
        # Fallback if no year provided, use a default (though caller should have validated)
        raise ValueError(f"predict_method({method}): 'year' must be provided for sampling.")
    
    sampled_fcs = []
    for yr in years:
        yr_fc  = train_fc.filter(ee.Filter.eq("year", int(yr)))
        yr_img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")\
            .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))\
            .mosaic().select(EMB_COLS)
        sampled_fcs.append(yr_img.sampleRegions(collection=yr_fc, properties=[LABEL_COL, "case_weight"], scale=scale, geometries=False, tileScale=16))
    train_sampled = ee.FeatureCollection(sampled_fcs).flatten()

    # 4. Sample Test Data on GEE
    sys.stderr.write(f"predict_method({method}): sampling test embeddings on GEE...\n")
    years = sorted(test_df["year"].unique()) if 'year' in test_df.columns else [int(year)] if year is not None else None
    if not years:
         raise ValueError(f"predict_method({method}): 'year' must be provided for sampling.")
         
    sampled_fcs = []
    for yr in years:
        yr_fc  = test_fc.filter(ee.Filter.eq("year", int(yr)))
        yr_img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")\
            .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))\
            .mosaic().select(EMB_COLS)
        sampled_fcs.append(yr_img.sampleRegions(collection=yr_fc, properties=['_point_id'], scale=scale, geometries=False, tileScale=16))
    test_sampled = ee.FeatureCollection(sampled_fcs).flatten()

    # 5. Train & Predict
    if is_classifier:
        clf = GEE_CLASSIFIER_METHODS[method](params)
        model = clf.train(features=train_sampled, classProperty=LABEL_COL, inputProperties=EMB_COLS)
        
        pred_col = "classification" if method != "maxent" else "probability"
        results = test_sampled.classify(model).reduceColumns(ee.Reducer.toList(2), ["_point_id", pred_col]).getInfo()
    else:
        # Reducer Path: Train (reduceColumns) and then Predict (map dot-product)
        reducer_name, _ = GEE_REDUCER_METHODS[method]
        
        if reducer_name in ("ridgeRegression", "linearRegression", "robustLinearRegression"):
            if reducer_name == "ridgeRegression":
                # Ridge centering using weighted training means
                means_res = train_sampled.reduceColumns(
                    reducer=ee.Reducer.mean().repeat(65),
                    selectors=EMB_COLS + [LABEL_COL],
                    weightSelectors=["case_weight"] * 65
                ).getInfo()
                m_x = means_res["mean"][:64]
                m_y = means_res["mean"][64]
                
                def center_and_weight_train(f):
                    w = f.getNumber("case_weight")
                    sw = w.sqrt()
                    x_centered = f.toArray(EMB_COLS).subtract(ee.Array(m_x))
                    w_cols = [f"w_{c}" for c in EMB_COLS]
                    scaled_vals = x_centered.multiply(sw).toList()
                    props = ee.Dictionary.fromLists(w_cols, scaled_vals)
                    y_centered = f.getNumber(LABEL_COL).subtract(m_y)
                    return f.set(props).set({"w_label": y_centered.multiply(sw)})
                
                train_centered = train_sampled.map(center_and_weight_train)
                reducer = ee.Reducer.ridgeRegression(numX=64, numY=1, lambda_=params.get("lambda_", 0.1))
                selectors = [f"w_{c}" for c in EMB_COLS] + ["w_label"]
                
                res = train_centered.reduceColumns(reducer=reducer, selectors=selectors).getInfo()
                if not res or res.get("coefficients") is None:
                    sys.stderr.write(f"predict_method({method}): ERROR: GEE returned no coefficients.\n")
                    intercept, weights = 0.0, [0.0] * 64
                else:
                    coefs = np.array(res.get("coefficients")).flatten()
                    weights = coefs[1:65].tolist()
                    if len(weights) < 64: weights = weights + [0.0]*(64-len(weights))
                    pred_at_mean = np.dot(weights, m_x)
                    intercept = m_y - pred_at_mean + float(coefs[0])
            else:
                # Linear
                def weight_inputs_train(f):
                    lbl = f.getNumber(LABEL_COL)
                    w = f.getNumber("case_weight")
                    sw = w.sqrt()
                    w_cols = [f"w_{c}" for c in EMB_COLS]
                    scaled_vals = f.toArray(EMB_COLS).multiply(sw).toList()
                    props = ee.Dictionary.fromLists(w_cols, scaled_vals)
                    return f.set(props).set({
                        "w_constant": sw,
                        "w_label": lbl.multiply(sw)
                    })
                
                train_weighted = train_sampled.map(weight_inputs_train)
                reducer = getattr(ee.Reducer, reducer_name)(numX=65, numY=1)
                selectors = ["w_constant"] + [f"w_{c}" for c in EMB_COLS] + ["w_label"]
                
                res = train_weighted.reduceColumns(reducer=reducer, selectors=selectors).getInfo()
                if not res or res.get("coefficients") is None:
                    sys.stderr.write(f"predict_method({method}): ERROR: GEE returned no coefficients.\n")
                    intercept, weights = 0.0, [0.0] * 64
                else:
                    coefs = np.array(res.get("coefficients")).flatten()
                    intercept, weights = float(coefs[0]), coefs[1:65].tolist()
                    if len(weights) < 64: weights = weights + [0.0]*(64-len(weights))
        elif reducer_name == "mean":
            # Centroid: mean of presence points only
            presence_fc = train_sampled.filter(ee.Filter.eq(LABEL_COL, 1))
            res = presence_fc.reduceColumns(
                reducer=ee.Reducer.mean().repeat(64), 
                selectors=EMB_COLS
            ).getInfo()
            weights = [float(w) for w in res["mean"]]
            intercept = 0.0
        else:
            raise ValueError(f"predict_method: Reducer {reducer_name} not supported for GEE prediction.")

        # Vectorized application of weights and intercept
        sys.stderr.write(f"predict_method({method}): scoring test points on GEE (weights[0]={float(weights[0]):.4f}, intercept={float(intercept):.4f}) ...\n")
        
        weights_ee = ee.Array([float(w) for w in weights]).reshape([1, 64])
        intercept_ee = ee.Number(float(intercept))

        def score_fc_point(feat):
            emb_arr = feat.toArray(EMB_COLS).reshape([64, 1])
            sim = weights_ee.matrixMultiply(emb_arr).add(intercept_ee)
            return feat.set('similarity', sim.get([0, 0]))

        results = test_sampled.map(score_fc_point).reduceColumns(ee.Reducer.toList(2), ["_point_id", "similarity"]).getInfo()

    # 6. Reconstruct predictions in same order
    out_dict = dict(results['list'])
    
    sims = []
    for pid in test_df['_point_id']:
        val = out_dict.get(pid, np.nan)
        if val is not None and not np.isnan(val):
            val = float(val)
        else:
            val = np.nan
        sims.append(val)

    return np.array(sims)
