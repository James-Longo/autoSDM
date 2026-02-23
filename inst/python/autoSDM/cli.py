# Suppress Google API Python version warnings
import warnings
warnings.filterwarnings("ignore", message=".*Python version.*will stop supporting.*", category=FutureWarning)

import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
import concurrent.futures
from autoSDM.extrapolate import generate_prediction_map, download_mask, export_to_gcs, merge_rasters
from shapely.geometry import mapping


def get_cached_cv(args, df):
    """
    Runs cross-validation or loads from a shared temp cache file.
    The cache dir is supplied by R via --cv-cache-dir (a tempdir path) so the
    files never appear in the output directory. The points CSV is not persisted —
    k-means fold assignment is deterministic (random_state=42), so re-running it
    on the same data always produces the same splits.
    """
    import ee, json, os, sys, tempfile
    from autoSDM.trainer import run_parallel_cv

    # Use caller-supplied temp dir, or fall back to the system temp dir
    cache_dir = getattr(args, 'cv_cache_dir', None) or tempfile.gettempdir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_json = os.path.join(cache_dir, "cv_cache.json")

    train_methods = args.train_methods.split(",") if hasattr(args, 'train_methods') and args.train_methods else ["centroid", "ridge"]
    eval_methods = args.eval_methods.split(",") if hasattr(args, 'eval_methods') and args.eval_methods else ["centroid", "ridge", "ensemble"]

    # If already computed by an earlier pipelined run, check what's missing
    missing_eval_methods = []
    cv_results = {}
    if os.path.exists(cache_json):
        sys.stderr.write("CV: Loading previously computed CV results from cache...\n")
        with open(cache_json, 'r') as f:
            cv_results = json.load(f)

        cached_methods = list(cv_results.get('average', {}).keys())
        missing_eval_methods = [m for m in eval_methods if m not in cached_methods]

        if not missing_eval_methods:
            return cv_results
        else:
            sys.stderr.write(f"CV: Cache missing metrics for {missing_eval_methods}. Computing missing methods...\n")
    else:
        missing_eval_methods = eval_methods

    try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
    except: pass

    sys.stderr.write(f"Running 10-fold Spatial k-Means Cross-Validation (Training: {','.join(train_methods)} | Evaluating: {','.join(missing_eval_methods)})...\n")
    new_cv_results = run_parallel_cv(
        df=df,
        ecological_vars=[f"A{i:02d}" for i in range(64)],
        scale=args.scale,
        year=args.year,
        n_folds=10,
        train_methods=train_methods,
        eval_methods=missing_eval_methods
    )

    # Merge new results with existing cache (if any)
    if 'average' in new_cv_results:
        if not cv_results:
            cv_results = new_cv_results
        else:
            cv_results['average'].update(new_cv_results['average'])
            for i, fold in enumerate(cv_results['folds']):
                if i < len(new_cv_results['folds']):
                    for key in missing_eval_methods:
                        if key in new_cv_results['folds'][i]:
                            fold[key] = new_cv_results['folds'][i][key]

        with open(cache_json, 'w') as f:
            json.dump({k: v for k, v in cv_results.items() if k != 'df'}, f)

    return cv_results


def process_species_task(sp, df_sp, output_dir, args, aoi, master_sampled_df, year=None):
    if year is None:
        raise ValueError("process_species_task: 'year' must be explicitly provided.")
    import ee
    try:
        sp_dir = os.path.join(output_dir, str(sp))
        os.makedirs(sp_dir, exist_ok=True)

        # Merge master embeddings into this species dataframe
        df_sp = pd.merge(df_sp, master_sampled_df, on=['longitude', 'latitude', 'year'], how='inner')
        if df_sp.empty:
            sys.stderr.write(f"--- Warning: No valid points for {sp} after embedding merge ---\n")
            return

        from autoSDM.analyzer import analyze_method, GEE_CLASSIFIER_METHODS
        from autoSDM.extrapolate import get_prediction_image

        method = args.method if args.method != "ensemble" else "centroid"
        has_absences = (df_sp['present'] == 0).any() if 'present' in df_sp.columns else False

        if not has_absences and aoi:
            n_bg = args.count if args.count else len(df_sp) * 10
            sys.stderr.write(f"{sp} ({method}): generating {n_bg} background points...\n")
            from autoSDM.trainer import get_background_embeddings
            df_bg = get_background_embeddings(aoi, n_points=n_bg, scale=args.scale, year=year)
            if not df_bg.empty:
                df_sp = pd.concat([df_sp, df_bg], ignore_index=True)

        params = {}
        if method in ("rf", "gbt"): params = {"numberOfTrees": getattr(args, 'n_trees', 100)}
        if method == "svm":          params = {"kernelType": getattr(args, 'svm_kernel', 'RBF')}

        res  = analyze_method(df_sp, method=method, params=params, scale=args.scale, year=year)
        meta = {"method": method, "params": params, "metrics": res['metrics'],
                "similarity_range": res.get('similarity_range')}
        if 'weights' in res:
            meta['weights'] = res['weights']; meta['intercept'] = res['intercept']

        json_path = os.path.join(sp_dir, f"{method}.json")
        # Save CSV so classifiers can reload it for map generation
        csv_path  = os.path.join(sp_dir, f"{method}.csv")
        res['clean_data'].to_csv(csv_path, index=False)
        if method in GEE_CLASSIFIER_METHODS:
            meta['training_csv'] = os.path.abspath(csv_path)

        with open(json_path, 'w') as f:
            json.dump(meta, f)

        img, _ = get_prediction_image(meta, aoi=aoi, year=year, scale=args.scale)
        if aoi:
            img = img.clip(aoi)

        with open(os.path.join(sp_dir, "results.json"), 'w') as f:
            json.dump({"species": sp, "method": method, "metrics": res['metrics']}, f)

        if aoi and not args.view:
            from autoSDM.extrapolate import download_mask
            download_mask(img, aoi, args.scale, os.path.join(sp_dir, f"{sp}_{method}_{args.scale}m.tif"))

        sys.stderr.write(f"--- Completed: {sp} ---\n")
    except Exception as e:
        sys.stderr.write(f"--- Failed: {sp} | {e} ---\n")
        import traceback
        traceback.print_exc()


def run_multi_species_pipeline(args, df, output_dir):
    species_list = df[args.species_col].unique()
    sys.stderr.write(f"--- Orchestrating Multi-Species Pipeline for {len(species_list)} species on GEE ---\n")
    
    # 1. Deduplicate coordinates for MASTER SAMPLING
    sys.stderr.write("--- Deduplicating coordinates for Master Sampling ---\n")
    unique_coords = df[['longitude', 'latitude', 'year']].drop_duplicates().copy()
    sys.stderr.write(f"Unique coordinates to sample: {len(unique_coords)} (from {len(df)} total records)\n")
    
    import ee
    # 2. Extract Embeddings ONCE for everyone
    from autoSDM.trainer import _prepare_training_data
    # We use a dummy nuisance var to satisfy the cleaner, or pass empty
    # We extract A00-A63
    master_data = _prepare_training_data(
        unique_coords, 
        ecological_vars=[f"A{i:02d}" for i in range(64)], 
        class_property=None, # Not needed for sampling
        scale=args.scale
    )
    
    # Download sampled embeddings to local df for fast partitioning
    sys.stderr.write("Downloading sampled embeddings from GEE for local partitioning...\n")
    emb_cols = [f"A{i:02d}" for i in range(64)]
    sample_res = master_data['fc'].reduceColumns(
        reducer=ee.Reducer.toList().repeat(64 + 3), # embs + lat/lon/year
        selectors=emb_cols + ['longitude', 'latitude', 'year']
    ).getInfo()
    
    master_sampled_df = pd.DataFrame(sample_res['list']).T
    master_sampled_df.columns = emb_cols + ['longitude', 'latitude', 'year']
    sys.stderr.write(f"Master sampled data size: {len(master_sampled_df)}\n")

    aoi = None
    if args.lat is not None and args.lon is not None and args.radius is not None:
        aoi = ee.Geometry.Point([args.lon, args.lat]).buffer(args.radius)
    elif args.aoi_path:
        import geopandas as gpd
        gdf = gpd.read_file(args.aoi_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
        # Use full polygon geometry instead of bounding box 
        aoi = ee.Geometry(mapping(gdf.geometry.unary_union))
    
    # Run in parallel blocks of concurrent GEE requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for sp in species_list:
            df_sp = df[df[args.species_col] == sp]
            futures.append(executor.submit(process_species_task, sp, df_sp, output_dir, args, aoi, master_sampled_df, args.year))
        
        concurrent.futures.wait(futures)

def main():
    parser = argparse.ArgumentParser(description="autoSDM CLI")
    parser.add_argument("mode", choices=["analyze", "extrapolate", "ensemble", "predict", "background"])
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", help="Optional GEE service account key path. If not provided, uses existing session auth.")
    parser.add_argument("--project", help="Google Cloud Project ID for Earth Engine initialization.")
    parser.add_argument("--aoi-path", help="Path to GeoJSON or Shapefile (polygon) for mapping extent.")
    parser.add_argument("--meta", action="append", help="Path to meta JSON(s) (for extrapolate/ensemble mode)")
    parser.add_argument("--meta2", help="DEPRECATED: Use multiple paths in --meta instead")
    parser.add_argument("--scale", type=int, help="Scale in meters (e.g. 10 or 100). Required.")
    parser.add_argument("--view", action="store_true", help="Generate a web map for visualization instead of downloading GeoTIFFs.")
    parser.add_argument("--gcs-bucket", help="GCS bucket name for server-side export (bypasses local download)")
    parser.add_argument("--wait", action="store_true", help="Wait for the export task(s) to complete and show progress updates.")
    parser.add_argument("--zip", action="store_true", help="Zip the output rasters (only for local download mode).")
    parser.add_argument("--method",
        choices=["centroid", "rf", "gbt", "cart", "svm", "maxent",
                 "ridge", "linear", "robust_linear", "ensemble", "mean"],
        default="centroid",
        help="Modeling method (GEE classifier or regression reducer).")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of trees for rf/gbt (default: 100)")
    parser.add_argument("--svm-kernel", default="RBF", choices=["LINEAR","POLY","RBF","SIGMOID"], help="SVM kernel type (default: RBF)")
    parser.add_argument("--lambda", type=float, default=0.1, dest="lambda_", help="Regularisation strength for ridge/linear reducers (default: 0.1)")
    parser.add_argument("--prefix", help="Prefix for output raster filenames (default: 'prediction_map')")
    parser.add_argument("--only-similarity", action="store_true", help="Only generate/download similarity map (skip masks)")
    parser.add_argument("--lat", type=float, help="Latitude for AOI center")
    parser.add_argument("--lon", type=float, help="Longitude for AOI center")
    parser.add_argument("--radius", type=float, help="Radius (in meters) for AOI")
    parser.add_argument("--species-col", help="Column name for species in multi-species mode.")
    parser.add_argument("--count", type=int, default=None, help="Number of background points to sample.")
    parser.add_argument("--cv", action="store_true", help="Run spatial block cross-validation.")
    parser.add_argument("--train-methods", help="Comma-separated list of methods to train during CV (e.g., 'centroid,ridge')")
    parser.add_argument("--eval-methods", help="Comma-separated list of methods to evaluate during CV (e.g., 'ensemble')")
    parser.add_argument("--cv-cache-dir", help="Directory for the inter-process CV cache file (default: system tempdir). Set to R's tempdir() to keep output folders clean.")
    parser.add_argument("--year", type=int, help="Alpha Earth Mosaic year for mapping (2017-2025). Required.")
    
    args = parser.parse_args()
    sys.stderr.write(f"DEBUG: autoSDM CLI starting. Mode={args.mode}, Input={args.input}\n")
    sys.stderr.flush()
    
    if args.mode == "background":
        # Background Sampling Mode
        import ee
        try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
        except: pass

        # Reconstruct AOI
        aoi = None
        if args.lat is not None and args.lon is not None and args.radius is not None:
            aoi = ee.Geometry.Point([args.lon, args.lat]).buffer(args.radius)
        elif args.aoi_path:
            import geopandas as gpd
            gdf = gpd.read_file(args.aoi_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
            # Use full polygon geometry
            aoi = ee.Geometry(mapping(gdf.geometry.unary_union))
        
        if not aoi:
            sys.stderr.write("Error: AOI required for background sampling (lat/lon/radius or aoi-path).\n")
            sys.exit(1)

        from autoSDM.trainer import get_background_embeddings
        n_bg = args.count if args.count else 1000
        sys.stderr.write(f"Sampling {n_bg} background points from GEE...\n")
        df_bg = get_background_embeddings(aoi, n_points=n_bg, scale=args.scale, year=args.year)
        
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        df_bg.to_csv(args.output, index=False)
        sys.stderr.write(f"Background sampling complete. Results saved to {args.output}\n")
        sys.exit(0)

    elif args.mode == "analyze":
        df = pd.read_csv(args.input)
        
        # Multi-Species Orchestration (Shifted from R to GEE)
        if args.species_col and args.method == "ensemble":
            import ee
            try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
            except: pass
            run_multi_species_pipeline(args, df, args.output)
            sys.exit(0)

        import ee
        try:
            ee.Initialize(project=args.project) if args.project else ee.Initialize()
        except Exception:
            pass

        from autoSDM.analyzer import analyze_method, GEE_CLASSIFIER_METHODS, GEE_REDUCER_METHODS

        method = args.method

        # Build method-specific hyperparameter dict
        params = {}
        if method in ("rf",):
            params = {"numberOfTrees": args.n_trees}
        elif method == "gbt":
            params = {"numberOfTrees": args.n_trees}
        elif method == "svm":
            params = {"kernelType": args.svm_kernel}
        elif method in ("ridge", "linear", "robust_linear"):
            params = {"lambda_": args.lambda_}

        if method == "ensemble":
            run_multi_species_pipeline(args, df, args.output)
            return

        # ── Unified path for all classifiers, reducers, and local mean ────
        needs_bg = method in GEE_CLASSIFIER_METHODS or method in GEE_REDUCER_METHODS
        has_absences = (df['present'] == 0).any() if 'present' in df.columns else False

        if needs_bg and not has_absences:
            sys.stderr.write(f"{method}: presence-only data — generating background points on GEE...\n")
            aoi_geom = None
            if args.lat is not None and args.lon is not None and args.radius is not None:
                aoi_geom = ee.Geometry.Point([args.lon, args.lat]).buffer(args.radius)
            elif args.aoi_path:
                import geopandas as gpd
                gdf = gpd.read_file(args.aoi_path)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                aoi_geom = ee.Geometry(mapping(gdf.geometry.unary_union))
            else:
                min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
                min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
                aoi_geom = ee.Geometry.Rectangle([min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1])

            from autoSDM.trainer import get_background_embeddings
            n_bg = args.count if args.count else len(df) * 10
            sys.stderr.write(f"{method}: generating {n_bg} background points (Override: {bool(args.count)})...\n")
            if 'type' not in df.columns:
                df['type'] = df['present'].map({1: 'presence', 0: 'absence'})
            df_bg = get_background_embeddings(aoi_geom, n_points=n_bg, scale=args.scale, year=args.year)
            if not df_bg.empty:
                df = pd.concat([df, df_bg], ignore_index=True)

        res = analyze_method(df, method=method, params=params, scale=args.scale, year=args.year)

        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        res['clean_data'].to_csv(args.output, index=False)

        meta = {
            "method":           method,
            "params":           params,
            "metrics":          res['metrics'],
            "cv_results":       None,
            "similarity_range": res.get('similarity_range'),
            "similarities":     res.get('similarities'),
        }
        # Standardised model weights/intercept
        if 'weights' in res:
            meta['weights']   = res['weights']
            meta['intercept'] = res['intercept']

        # Classifier methods: store CSV path so get_prediction_image() can reload
        # the full training data (with lat/lon + both classes) at map time
        meta['training_csv'] = os.path.abspath(args.output)

        if args.cv:
            cv_results = get_cached_cv(args, df)
            cv_res = cv_results.get('average', {})
            meta["cv_results"] = cv_res.get(method)


        if args.output.endswith('.csv'):
            meta_path = args.output.replace('.csv', '.json')
        else:
            meta_path = args.output + ".json"
            
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        sys.stderr.write(f"Analysis complete. Results saved to {args.output} and {meta_path}\n")

    elif args.mode == "predict":
        # Point Prediction Mode
        df = pd.read_csv(args.input)
        
        import ee
        try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
        except: pass
            
        # 2. Load Metadata
        with open(args.meta[0]) as f:
            meta = json.load(f)
        
        method = meta.get('method', 'centroid')
        
        # 4. Point Prediction (Always on GEE)
        training_csv = meta.get('training_csv')
        if training_csv and os.path.exists(training_csv):
            sys.stderr.write(f"Predicting {method} similarities on GEE...\n")
            df_train = pd.read_csv(training_csv)
            from autoSDM.analyzer import predict_method
            sims = predict_method(df_train, df, method, params=meta.get('params'), scale=args.scale, year=args.year)
            df['similarity'] = sims
        else:
            sys.stderr.write(f"Warning: No training_csv found for {method}. Similarity scores will be NaN.\n")
            df['similarity'] = np.nan
            



        # 5. Save Results
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        sys.stderr.write(f"Point predictions complete. Results saved to {args.output}\n")

    elif args.mode == "extrapolate" or args.mode == "ensemble":
        if not args.meta and args.mode != "ensemble":
            sys.stderr.write(f"Error: --meta required for {args.mode} mode\n")
            sys.exit(1)
            
        df = pd.read_csv(args.input)
        # Normalize column names to lowercase for consistency
        df.columns = [c.lower() for c in df.columns]

        # Ensure year column is present for extractor
        if 'year' not in df.columns:
            if 'observation date' in df.columns:
                 sys.stderr.write("Deriving 'year' from 'observation date'...\n")
                 # Use errors='coerce' to handle messy dates and fillna with args.year
                 df['year'] = pd.to_datetime(df['observation date'], errors='coerce').dt.year
                 df['year'] = df['year'].fillna(args.year).astype(int)
            else:
                 df['year'] = args.year

        import ee
        try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
        except: pass

        from autoSDM.extrapolate import get_prediction_image

        
        # Determine AOI if lat/lon/radius are provided
        import ee
        aoi = None
        if args.lat is not None and args.lon is not None and args.radius is not None:
            sys.stderr.write(f"Using specified AOI: {args.radius}m radius around {args.lat}, {args.lon}\n")
            aoi = ee.Geometry.Point([args.lon, args.lat]).buffer(args.radius)
        elif args.aoi_path:
            sys.stderr.write(f"Loading AOI from {args.aoi_path}...\n")
            import geopandas as gpd
            gdf = gpd.read_file(args.aoi_path)
            # Re-project to 4326 if needed
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # Combine all features into one geometry
            # Use unary_union to get the actual borders
            merged_geom = gdf.geometry.unary_union
            aoi = ee.Geometry(mapping(merged_geom))
            sys.stderr.write(f"AOI loaded from borders of {args.aoi_path}\n")
        
        if args.mode == "ensemble":
            meta_paths = args.meta
            if not meta_paths or len(meta_paths) < 1:
                sys.stderr.write("Error: At least one --meta file required for ensemble/extrapolate mode.\n")
                sys.exit(1)
            sys.stderr.write(f"Ensemble mode: Loading {len(meta_paths)} meta files: {meta_paths}\n")
            
            prediction_map = None
            ensemble_images = []
            for i, mp in enumerate(meta_paths):
                sys.stderr.write(f"Processing meta file {i+1}/{len(meta_paths)}: {mp}\n")
                with open(mp) as f:
                    mdata = json.load(f)
                
                img_i, aoi_i = get_prediction_image(mdata, df, aoi=aoi, year=args.year, scale=args.scale)
                if aoi is None: aoi = aoi_i
                
                # Normalize based on similarity_range from training metadata
                s_range = mdata.get('similarity_range', [0.0, 1.0])
                if s_range is None or len(s_range) < 2:
                    s_range = [0.0, 1.0]

                s_min, s_max = s_range[0], s_range[1]
                
                # Rescale band to 0-1
                if s_max - s_min > 1e-9:
                    norm_img = img_i.select('similarity').subtract(s_min).divide(s_max - s_min)
                else:
                    norm_img = img_i.select('similarity')

                ensemble_images.append(norm_img)
            
            prediction_map = ee.ImageCollection.fromImages(ensemble_images).mean().rename('similarity')


            thresholds = {}
        else:
            # Standard Extrapolate Mode (Single method)
            with open(args.meta[0]) as f:
                meta = json.load(f)
            
            prediction_map_raw, aoi = get_prediction_image(meta, df, aoi=aoi, year=args.year, scale=args.scale)
            
            # Normalize single map if range exists
            s_range = meta.get('similarity_range', [0.0, 1.0])
            s_min, s_max = s_range[0], s_range[1]
            if s_max - s_min > 1e-9:
                sys.stderr.write(f"  Normalizing (single) layer to [0, 1] using range: {s_min:.4f} to {s_max:.4f}\n")
                prediction_map = prediction_map_raw.select('similarity').subtract(s_min).divide(s_max - s_min).rename('similarity')
            else:
                prediction_map = prediction_map_raw

            thresholds = {}

        # --- Evaluate Metrics on Training Data (Ensemble) ---
        calculated_metrics = None
        cv_res_data = None
        if args.mode == "ensemble":
            # Identify class property
            class_prop = None
            for c in ['present', 'presence', 'present?', 'presence?', 'status']:
                if c in df.columns:
                    class_prop = c
                    break
            
            if class_prop is None:
                # If not found, check if it's presence-only (all 1s) or if we need to CREATE it
                sys.stderr.write("Class property not found. Assuming presence data and creating 'present' column...\n")
                df['present'] = 1
                class_prop = 'present'

            # Ensure all entries have a value (1 for presence if we just created it)
            df[class_prop] = df[class_prop].fillna(1)
            
            has_absences = (df[class_prop] == 0).any()
            
            if not has_absences:
                from autoSDM.trainer import get_background_embeddings
                n_bg = args.count if args.count else len(df) * 10
                sys.stderr.write(f"Presence-only data detected for Ensemble evaluation. Generating {n_bg} background points on GEE...\n")
                
                # Use AOI from bounds or arguments
                aoi_bg = aoi
                if aoi_bg is None:
                    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
                    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
                    aoi_bg = ee.Geometry.Rectangle([min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1])

                # Label existing points
                if 'type' not in df.columns:
                    df['type'] = 'presence'

                df_bg = get_background_embeddings(aoi_bg, n_points=n_bg, scale=args.scale, year=args.year)
                if not df_bg.empty:
                    # Sync columns (embeddings)
                    df = pd.concat([df, df_bg], ignore_index=True)
                    sys.stderr.write(f"Added {len(df_bg)} background points.\n")
                    sys.stderr.flush()
            
            if class_prop in df.columns:
                try:
                    sys.stderr.write("Calculating ensemble accuracy metrics on training data...\n")
                    from autoSDM.trainer import _prepare_training_data
                    from autoSDM.analyzer import calculate_classifier_metrics

                    if 'year' not in df.columns:
                        df['year'] = args.year
                        
                    features = []
                    
                    # Helper for safe integer conversion
                    def safe_int(v):
                        try: return int(v)
                        except: return 0
                    
                    # Group by class to creating optimized MultiPoint features
                    unique_classes = df[class_prop].unique()
                    for cls_val in unique_classes:
                        subset = df[df[class_prop] == cls_val]
                        if subset.empty: continue
                        coords = subset[['longitude', 'latitude']].values.tolist()
                        
                        # Upload as individual points (safe for <5000)
                        # This avoids sampleRegions/MultiPoint complexities
                        for xy in coords:
                            feat = ee.Feature(ee.Geometry.Point(xy), {class_prop: safe_int(cls_val)})
                            features.append(feat)
                            
                    fc_eval = ee.FeatureCollection(features)
                    
                    try:
                        sz = fc_eval.size().getInfo()
                        sys.stderr.write(f"DEBUG: Validation FC size: {sz}\n")
                        bnds = prediction_map.bandNames().getInfo()
                        sys.stderr.write(f"DEBUG: Prediction Map bands: {bnds}\n")
                        sys.stderr.flush()
                    except Exception as e:
                        sys.stderr.write(f"DEBUG: Failed to get info: {e}\n")

                    # Sample prediction map at training locations
                    sys.stderr.write("DEBUG: Starting sampleRegions...\n")
                    sys.stderr.flush()
                    sampled = prediction_map.sampleRegions(
                        collection=fc_eval,
                        scale=args.scale, 
                        geometries=False,
                        tileScale=16
                    )

                    # Retrieve results
                    sys.stderr.write("DEBUG: Calling getInfo()...\n")
                    sys.stderr.flush()
                    info = sampled.getInfo()
                    sys.stderr.write("DEBUG: Got info.\n")
                    sys.stderr.flush()
                    
                    if info['features']:
                        scs = []
                        lbls = []
                        # class_prop might have been renamed by _prepare_training_data (e.g. dots to underscores)
                        # But we are using raw DF now, so it matches.
                        safe_class_prop = class_prop 

                        for f in info['features']:
                            p = f['properties']
                            # prediction_map band is 'similarity'
                            if 'similarity' in p and safe_class_prop in p:
                                scs.append(p['similarity'])
                                lbls.append(p[safe_class_prop])
                        
                        scs = np.array(scs)
                        lbls = np.array(lbls)
                        
                        pos = scs[lbls == 1]
                        neg = scs[lbls == 0]
                        
                        if len(pos) > 0 and len(neg) > 0:
                            calculated_metrics = calculate_classifier_metrics(pos, neg)
                            sys.stderr.write(f"Ensemble Metrics: CBI={calculated_metrics.get('cbi', 0):.3f}, AUC-ROC={calculated_metrics.get('auc_roc', 0.5):.3f}\n")
                        else:
                            sys.stderr.write("Warning: Not enough points (presence/absence) for metrics calculation.\n")
                    else:
                        sys.stderr.write("Warning: Sampling returned no features (empty intersection?).\n")

                except Exception as e:
                    sys.stderr.write(f"Warning: Failed to calculate ensemble metrics: {e}\n")
                    import traceback
                    traceback.print_exc()

            # --- Run 10-fold Spatial CV for Ensemble ---
            if args.cv:
                try:
                    # Label input df (presence vs absence)
                    if 'type' not in df.columns:
                        df['type'] = df[class_prop].map({1: 'presence', 0: 'absence'}).fillna('background')

                    cv_results = get_cached_cv(args, df)
                    
                    # Export CV points CSV
                    if 'df' in cv_results:
                        cv_points_path = args.output.replace('.json', '_points.csv') if args.output.endswith('.json') else args.output + "_points.csv"
                        cv_results['df'].to_csv(cv_points_path, index=False)
                        sys.stderr.write(f"CV points with fold assignments saved to {cv_points_path}\n")

                    # Report all folds
                    sys.stderr.write("-" * 65 + "\n")
                    sys.stderr.write(f"{'Fold':<6} | {'Pres':<6} | {'BG':<6} | {'CBI':<10} | {'AUC-ROC':<10} | {'AUC-PR':<10}\n")
                    sys.stderr.write("-" * 65 + "\n")
                    
                    csv_rows = []
                    folds_list = cv_results.get('folds', [])
                    for i, fold_res in enumerate(folds_list):
                        m = fold_res['ensemble']
                        cnt = fold_res['counts']
                        p_cnt = cnt.get('presence', 0)
                        b_cnt = cnt.get('background', 0)
                        
                        sys.stderr.write(f"{i+1:<6} | {p_cnt:<6} | {b_cnt:<6} | {m.get('cbi',0):<10.3f} | {m.get('auc_roc',0.5):<10.3f} | {m.get('auc_pr',0):<10.3f}\n")
                        
                        csv_rows.append({
                            'fold': i + 1,
                            'presence_count': p_cnt,
                            'background_count': b_cnt,
                            'cbi': m.get('cbi', 0),
                            'auc_roc': m.get('auc_roc', 0.5),
                            'auc_pr': m.get('auc_pr', 0)
                        })
                    
                    sys.stderr.write("-" * 65 + "\n")
                    
                    # Report average (filtered for folds with presences)
                    avg = cv_results.get('average', {}).get('ensemble', {})
                    if avg:
                        sys.stderr.write(f"{'AVG*':<6} | {'(val)':<6} | {'':<6} | {avg.get('cbi', 0):<10.3f} | {avg.get('auc_roc', 0.5):<10.3f} | {avg.get('auc_pr', 0):<10.3f}\n")
                        sys.stderr.write("-" * 65 + "\n")
                        sys.stderr.write("*Average excludes folds with 0 presence points.\n\n")

                    # Export CV Metrics CSV
                    if csv_rows:
                        cv_metrics_path = args.output.replace('.json', '_cv_metrics.csv') if args.output.endswith('.json') else args.output + "_cv_metrics.csv"
                        pd.DataFrame(csv_rows).to_csv(cv_metrics_path, index=False)
                        sys.stderr.write(f"CV metrics per fold saved to {cv_metrics_path}\n")
                    
                    cv_res_data = {k: v for k, v in cv_results.items() if k != 'df'}
                except Exception as e:
                    sys.stderr.write(f"Warning: Failed to run ensemble CV: {e}\n")
                    import traceback
                    traceback.print_exc()
                    cv_res_data = None
            else:
                cv_res_data = None

            # Define meta for saving later
            meta = {
                "scale": args.scale,
                "mode": "ensemble",
                "metrics": calculated_metrics,
                "cv_results": cv_res_data
            }


        
        # Define output path base
        base_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "outputs"
        os.makedirs(base_dir, exist_ok=True)
        
        if args.view:
            # Visualization Mode
            import ee
            
            # Helper to generate Tile Layer
            def get_layer_config(image, viz_params, name):
                try:
                    map_id = image.getMapId(viz_params)
                    url = map_id['tile_fetcher'].url_format
                    sys.stderr.write(f"Generated Map URL for {name}: {url}\n")
                    return {
                        'url': url,
                        'name': name
                    }
                except Exception as e:
                    sys.stderr.write(f"Error generating MapID for {name}: {e}\n")
                    return None

            layers = []
            

            # 1. Similarity Layer
            sim_viz = {'min': 0, 'max': 1, 'palette': ['000000', '0000FF', '00FF00', 'FFFF00', 'FF0000']}
            layers.append(get_layer_config(prediction_map.select('similarity'), sim_viz, "Similarity (Filtered)"))
            
            # 2. Masks
            for name, val in thresholds.items():
                if val is not None:
                    # Make masks semi-transparent red
                    mask_viz = {'min': 0, 'max': 1, 'palette': ['00000000', 'FF000088']} 
                    layers.append(get_layer_config(prediction_map.select(f"binary_mask_{name}").selfMask(), mask_viz, f"Mask > {name}"))
            
            # Generate HTML
            center = aoi.centroid().coordinates().getInfo()
            sys.stderr.write(f"Map Center: {center}\n")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GEE Prediction Map ({args.scale}m)</title>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                <style>
                    body, html, #map {{ height: 100%; margin: 0; }}
                    .info {{ padding: 6px 8px; font: 14px/16px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div id="map"></div>
                <script>
                    var map = L.map('map').setView([{center[1]}, {center[0]}], 12);
                    
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '&copy; OpenStreetMap contributors'
                    }}).addTo(map);
                    
                    L.marker([{center[1]}, {center[0]}]).addTo(map)
                        .bindPopup("AOI Centroid").openPopup();
                    
                    var overlays = {{}};
            """
            
            for layer in layers:
                if layer:
                    html_content += f"""
                    var layer_{layer['name'].replace(' ', '_').replace('(', '').replace(')', '')} = L.tileLayer('{layer['url']}', {{
                        attribution: 'Google Earth Engine'
                    }}).addTo(map);
                    overlays["{layer['name']}"] = layer_{layer['name'].replace(' ', '_').replace('(', '').replace(')', '')};
                    """
            
            html_content += """
                    L.control.layers(null, overlays).addTo(map);
                </script>
            </body>
            </html>
            """
            
            map_name = f"map_{args.scale}m.html"
            map_path = os.path.join(base_dir, map_name)
            with open(map_path, 'w') as f:
                f.write(html_content)
                
            results = {'map_html': os.path.abspath(map_path), 'scale': args.scale}
            
            with open(args.output, 'w') as f:
                json.dump(results, f)

            sys.stderr.write(f"Visualization complete. Map saved to {map_name}\n")
            
        else:
            # Download Mode or GCS Export Mode
            results = {}
            
            # Apply AOI clipping if available to respect non-rectangular borders
            if aoi:
                prediction_map = prediction_map.clip(aoi)
            
            if args.gcs_bucket:
                # GCS Export Mode: Individual Rasters
                sys.stderr.write(f"Exporting individual rasters to GCS bucket: {args.gcs_bucket}...\n")
                
                results['gee_tasks'] = []
                
                # Get band names to export individually
                bands = prediction_map.bandNames().getInfo()
                
                tasks = []
                prefix = args.prefix if args.prefix else "prediction_map"
                for band in bands:
                    filename_prefix = f"{prefix}_{args.scale}m_{band}"
                    sys.stderr.write(f"Triggering export for band: {band}...\n")
                    task = export_to_gcs(
                        image=prediction_map.select(band).toFloat(),
                        region=aoi,
                        scale=args.scale,
                        bucket=args.gcs_bucket,
                        filename_prefix=filename_prefix
                    )
                    tasks.append(task)
                    results['gee_tasks'].append({
                        'band': band,
                        'task_id': task.id,
                        'gcs_path': f"gs://{args.gcs_bucket}/{filename_prefix}.tif"
                    })
                
                results['gcs_bucket'] = args.gcs_bucket
                results['monitoring_url'] = "https://code.earthengine.google.com/tasks"
                results['scale'] = args.scale
                results['mode'] = 'gcs_export'
                
                if args.wait:
                    import time
                    sys.stderr.write(f"Waiting for {len(tasks)} export tasks to complete...\n")
                    active_tasks = list(tasks)
                    while active_tasks:
                        # Update statuses
                        statuses = [t.status() for t in active_tasks]
                        active_ids = [s['id'] for s in statuses if s['state'] in ['READY', 'RUNNING']]
                        
                        # Print summary of progress
                        for s in statuses:
                            if s['state'] in ['READY', 'RUNNING']:
                                progress = s.get('progress', '0.0')
                                sys.stderr.write(f"Task {s['id']} ({s['description']}): {s['state']} - {float(progress)*100:.1f}%\n")
                        
                        # Filter out completed/failed
                        active_tasks = [t for t in active_tasks if t.active()]
                        if active_tasks:
                            time.sleep(20)
                    
                    sys.stderr.write("All GCS export tasks finished.\n")
            else:
                # Local Download Mode
                # 1. Download Continuous Similarity Map
                prefix = args.prefix if args.prefix else "prediction_map"
                sim_filename = f"{prefix}_{args.scale}m.tif"
                sim_path = os.path.join(base_dir, sim_filename)
                sim_files = download_mask(prediction_map.select('similarity'), aoi, args.scale, sim_path)
                # result = merge_rasters(sim_files, sim_path) # Disabled to save memory
                results['similarity_map'] = [os.path.abspath(f) for f in sim_files] if len(sim_files) > 1 else (os.path.abspath(sim_files[0]) if sim_files else None)
                
                # Cleanup tiles if merged - DISABLED
                # if final_sim and len(sim_files) > 1:
                #     for f in sim_files:
                #         if os.path.abspath(f) != os.path.abspath(final_sim) and os.path.exists(f):
                #             os.remove(f)
                
                # 2. Download Threshold Masks
                if not args.only_similarity:
                    for name, val in thresholds.items():
                        if name in ['auc', 'score_column'] or not isinstance(val, (int, float)):
                            continue
                        band_name = f"binary_mask_{name}"
                        mask_filename = f"{prefix}_{args.scale}m_{band_name}.tif"
                        mask_path = os.path.join(base_dir, mask_filename)
                        mask_files = download_mask(prediction_map.select(band_name), aoi, args.scale, mask_path)
                        # final_mask = merge_rasters(mask_files, mask_path) # Disabled to save memory
                        results[band_name] = [os.path.abspath(f) for f in mask_files] if len(mask_files) > 1 else (os.path.abspath(mask_files[0]) if mask_files else None)
                        
                        # Cleanup tiles - DISABLED
                        # if final_mask and len(mask_files) > 1:
                        #     for f in mask_files:
                        #         if os.path.abspath(f) != os.path.abspath(final_mask) and os.path.exists(f):
                        #             os.remove(f)
                
                results['scale'] = args.scale
                results['mode'] = 'local_download'

                # Zip individual rasters if requested
                if args.zip:
                    import zipfile
                    zip_path = os.path.join(base_dir, f"{prefix}_{args.scale}m_results.zip")
                    sys.stderr.write(f"Zipping results to {zip_path}...\n")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        # Gather all TIFFs created
                        for key, val in results.items():
                            if key in ['gee_tasks', 'mode', 'scale']:
                                continue
                            if isinstance(val, str) and val.endswith('.tif'):
                                zipf.write(val, os.path.basename(val))
                            elif isinstance(val, list):
                                for f in val:
                                    if isinstance(f, str) and f.endswith('.tif'):
                                        zipf.write(f, os.path.basename(f))
                    results['zip_file'] = os.path.abspath(zip_path)
            
            if calculated_metrics:
                results['metrics'] = calculated_metrics
            
            if cv_res_data:
                results['cv_results'] = cv_res_data

            # Save results, filtering out non-serializable DataFrames
            serializable_results = {k: v for k, v in results.items() if k not in ['df', 'results_df']}
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f)
                
            if args.gcs_bucket:
                sys.stderr.write(f"Export tasks started. Monitor via: https://code.earthengine.google.com/tasks\n")
            else:
                sys.stderr.write(f"Extrapolation complete at {args.scale}m. Maps saved to {base_dir}\n")

if __name__ == "__main__":
    main()
