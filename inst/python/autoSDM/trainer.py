import ee
import pandas as pd
import numpy as np
import sys
import os
import json

def _prepare_training_data(df, nuisance_vars, ecological_vars, class_property='present'):
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
    
    # 2. Sanitize column names for GEE
    name_map = {col: col.replace('.', '_') for col in all_predictors + [class_property, 'latitude', 'longitude']}
    df = df.rename(columns=name_map)
    all_predictors = [name_map[col] for col in all_predictors]
    nuisance_vars = [name_map[col] for col in nuisance_vars]
    ecological_vars = [name_map[col] for col in ecological_vars]
    class_property = name_map[class_property]

    # 3. Drop NAs
    missing_counts = df[all_predictors + [class_property]].isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    if not columns_with_missing.empty:
        sys.stderr.write("WARNING: Missing values detected in training columns. Dropping rows.\n")

    df_clean = df.dropna(subset=all_predictors + [class_property]).copy()
    if df_clean.empty:
        raise ValueError("No valid training data remaining after dropping missing values.")

    # 3.5. Subsample if too large for GEE payload (Maxent limit is ~5000-10000 points depending on dimensions)
    MAX_TRAIN_POINTS = 4000
    if len(df_clean) > MAX_TRAIN_POINTS:
        sys.stderr.write(f"Training data ({len(df_clean)} pts) exceeds GEE payload safety limit. Subsampling absences...\n")
        presence_df = df_clean[df_clean[class_property] == 1]
        absence_df = df_clean[df_clean[class_property] == 0]
        
        target_absences = max(500, MAX_TRAIN_POINTS - len(presence_df))
        if len(absence_df) > target_absences:
            absence_df = absence_df.sample(n=target_absences, random_state=42)
            df_clean = pd.concat([presence_df, absence_df])
            sys.stderr.write(f"New training size: {len(df_clean)} pts ({len(presence_df)} presences, {len(absence_df)} absences).\n")

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

    # 5. Determine Nuisance Optima (on presence points)
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
    
    # 6. Convert to ee.FeatureCollection
    features = []
    for _, row in df_clean.iterrows():
        props = {col: float(row[col]) for col in all_predictors + [class_property]}
        geom = ee.Geometry.Point([row['longitude'], row['latitude']])
        features.append(ee.Feature(geom, props))
    fc = ee.FeatureCollection(features)

    return {
        'fc': fc,
        'df_clean': df_clean,
        'nuisance_optima': nuisance_optima,
        'all_predictors': all_predictors,
        'class_property': class_property,
        'nuisance_vars': nuisance_vars,
        'ecological_vars': ecological_vars
    }

def train_rf_model(df, nuisance_vars, ecological_vars, class_property='present', key_path=None):
    if key_path:
        ee.Initialize(ee.ServiceAccountCredentials(json.load(open(key_path))["client_email"], key_path))

    data = _prepare_training_data(df, nuisance_vars, ecological_vars, class_property)
    
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

def train_maxent_model(df, nuisance_vars, ecological_vars, class_property='present', key_path=None):
    if key_path:
        ee.Initialize(ee.ServiceAccountCredentials(json.load(open(key_path))["client_email"], key_path))

    data = _prepare_training_data(df, nuisance_vars, ecological_vars, class_property)

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
    classified = fc.classify(classifier, 'classification')
    
    # Inspect first feature for scores
    sample = classified.first().getInfo()
    if not sample:
        sys.stderr.write("Warning: Classified collection is empty.\n")
        return {'95tpr': 0.1, '95tnr': 0.9, 'balanced': 0.5, 'auc': 0.0}
        
    props = sample['properties']
    score_col = 'classification' if 'classification' in props else 'probability'

    presence_only = classified.filter(ee.Filter.eq(class_property, 1))
    absence_only = classified.filter(ee.Filter.eq(class_property, 0))
    
    try:
        presence_probs_raw = presence_only.aggregate_array(score_col).getInfo()
        absence_probs_raw = absence_only.aggregate_array(score_col).getInfo()
        
        presence_probs = np.array([float(v) for v in presence_probs_raw if v is not None], dtype=float)
        absence_probs = np.array([float(v) for v in absence_probs_raw if v is not None], dtype=float)
        
        if presence_probs.size == 0:
            raise ValueError(f"No valid scores in '{score_col}' for presence")
            
        threshold_95tpr = float(np.percentile(presence_probs, 5))
        
        if absence_probs.size > 0:
            threshold_95tnr = float(np.percentile(absence_probs, 95))
            y_scores = np.concatenate([presence_probs, absence_probs])
            threshold_candidates = np.unique(y_scores)
            best_t, max_sum = 0, 0
            for t in threshold_candidates:
                tpr = np.mean(presence_probs >= t)
                tnr = np.mean(absence_probs < t)
                if tpr + tnr >= max_sum:
                    max_sum = tpr + tnr
                    best_t = t
            threshold_balanced = float(best_t)
            
            # AUC (Wilcoxon-Mann-Whitney)
            auc = 0
            for p in presence_probs:
                auc += np.sum(p > absence_probs)
                auc += 0.5 * np.sum(p == absence_probs)
            auc /= (len(presence_probs) * len(absence_probs))
        else:
            threshold_95tnr = float(np.percentile(presence_probs, 50))
            threshold_balanced = float(np.percentile(presence_probs, 50))
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
