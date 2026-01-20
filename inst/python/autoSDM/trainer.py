import ee
import pandas as pd
import numpy as np
import sys
import os
import json

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

def train_rf_model(df, nuisance_vars, ecological_vars, class_property='present', key_path=None, scale=10):
    if key_path:
        ee.Initialize(ee.ServiceAccountCredentials(json.load(open(key_path))["client_email"], key_path))

    data = _prepare_training_data(df, nuisance_vars, ecological_vars, class_property, scale=scale)
    
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=500,
        minLeafPopulation=10,
        bagFraction=0.6
    ).setOutputMode('PROBABILITY').train(
        features=data['fc'],
        classProperty=data['class_property'],
        inputProperties=data['all_predictors']
    )

    thresholds = calculate_classifier_performance(classifier, data['fc'], data['class_property'])
    sys.stderr.write(f"RF Analysis: AUC={thresholds['auc']:.4f}, Thresholds={thresholds}\n")

    return classifier, data['nuisance_optima'], data['df_clean'], thresholds

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

    thresholds = calculate_classifier_performance(classifier, data['fc'], data['class_property'])
    sys.stderr.write(f"Maxent Analysis: AUC={thresholds['auc']:.4f}, Thresholds={thresholds}\n")

    return classifier, data['nuisance_optima'], data['df_clean'], thresholds

def calculate_classifier_performance(classifier, fc, class_property):
    # Perform classification server-side
    classified = fc.classify(classifier, 'classification')
    
    # 1. Inspect one feature to find score column name
    # Explicitly select only properties we might need to minimize payload
    sample = ee.Feature(classified.first()).select(['classification', 'probability', class_property]).getInfo()
    if not sample:
        sys.stderr.write("Warning: Classified collection is empty.\n")
        return {'95tpr': 0.1, '95tnr': 0.9, 'balanced': 0.5, 'auc': 0.0}
        
    props = sample['properties']
    score_col = 'classification' if 'classification' in props else 'probability'

    # 2. Performance calculation server-side to avoid payload limits
    presence_only = classified.filter(ee.Filter.eq(class_property, 1))
    absence_only = classified.filter(ee.Filter.eq(class_property, 0))
    
    try:
        # Calculate thresholds server-side using percentiles
        threshold_res = presence_only.reduceColumns(
            reducer=ee.Reducer.percentile([5, 50]),
            selectors=[score_col]
        ).getInfo()
        
        t_95tpr = float(threshold_res['p5'])
        t_median = float(threshold_res['p50'])

        # Calculate AUC server-side
        # We use a custom logic for AUC if possible, or pull if small.
        # Given potential for 10k points, let's try to pull ONLY the score column to minimize payload.
        # But even better, let's use ee.Classifier.explain() or similar if possible.
        # Actually, for 10k points, pull of a single column is ~100KB, which is safe.
        
        # Pull presence and absence scores separately
        presence_probs = np.array(presence_only.aggregate_array(score_col).getInfo(), dtype=float)
        absence_probs = np.array(absence_only.aggregate_array(score_col).getInfo(), dtype=float)
        
        if presence_probs.size == 0:
            raise ValueError(f"No valid scores in '{score_col}' for presence")
            
        threshold_95tpr = t_95tpr
        
        if absence_probs.size > 0:
            # We can still calculate balanced threshold locally on the 1D arrays
            y_scores = np.concatenate([presence_probs, absence_probs])
            threshold_candidates = np.unique(y_scores)
            
            # Efficient local search for balanced threshold
            # Subsample candidates if too many
            if len(threshold_candidates) > 500:
                threshold_candidates = np.percentile(threshold_candidates, np.linspace(0, 100, 500))

            best_t, max_sum = 0, 0
            for t in threshold_candidates:
                tpr = np.mean(presence_probs >= t)
                tnr = np.mean(absence_probs < t)
                if tpr + tnr >= max_sum:
                    max_sum = tpr + tnr
                    best_t = t
            threshold_balanced = float(best_t)
            threshold_95tnr = float(np.percentile(absence_probs, 95))
            
            # AUC (Wilcoxon-Mann-Whitney)
            auc = 0
            for p in presence_probs:
                auc += np.sum(p > absence_probs)
                auc += 0.5 * np.sum(p == absence_probs)
            auc /= (len(presence_probs) * len(absence_probs))
        else:
            threshold_95tnr = t_median
            threshold_balanced = t_median
            auc = 0.5
            
    except Exception as e:
        sys.stderr.write(f"Warning: performance calculation failed: {e}\n")
        return {'95tpr': 0.1, '95tnr': 0.9, 'balanced': 0.5, 'auc': 0.0}

    return {
        '95tpr': threshold_95tpr,
        '95tnr': threshold_95tnr,
        'balanced': threshold_balanced,
        'auc': float(auc),
        'score_column': score_col
    }
