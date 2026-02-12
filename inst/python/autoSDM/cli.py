# Suppress Google API Python version warnings
import warnings
warnings.filterwarnings("ignore", message=".*Python version.*will stop supporting.*", category=FutureWarning)

import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
from autoSDM.extractor import GEEExtractor
from autoSDM.analyzer import analyze_embeddings
from autoSDM.extrapolate import generate_prediction_map, download_mask, export_to_gcs, merge_rasters

def main():
    parser = argparse.ArgumentParser(description="autoSDM CLI")
    parser.add_argument("mode", choices=["extract", "analyze", "extrapolate", "ensemble", "predict"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", help="Optional GEE service account key path. If not provided, uses existing session auth.")
    parser.add_argument("--project", help="Google Cloud Project ID for Earth Engine initialization.")
    parser.add_argument("--aoi-path", help="Path to GeoJSON or Shapefile (polygon) for mapping extent.")
    parser.add_argument("--meta", help="Path to meta JSON (for extrapolate/ensemble mode)")
    parser.add_argument("--meta2", help="Path to second meta JSON (for ensemble mode)")
    parser.add_argument("--coarse-meta", help="Path to coarse meta JSON (for filtering 10m output)")
    parser.add_argument("--scale", type=int, default=10, help="Scale in meters (default: 10)")
    parser.add_argument("--view", action="store_true", help="Generate a web map for visualization instead of downloading GeoTIFFs.")
    parser.add_argument("--gcs-bucket", help="GCS bucket name for server-side export (bypasses local download)")
    parser.add_argument("--wait", action="store_true", help="Wait for the export task(s) to complete and show progress updates.")
    parser.add_argument("--zip", action="store_true", help="Zip the output rasters (only for local download mode).")
    parser.add_argument("--method", choices=["centroid", "maxent"], default="centroid", help="Mapping method: 'centroid' (presence-only) or 'maxent' (Maxent).")
    parser.add_argument("--nuisance-vars", help="Comma-separated list of columns to treat as nuisance variables.")
    parser.add_argument("--prefix", help="Prefix for output raster filenames (default: 'prediction_map')")
    parser.add_argument("--only-similarity", action="store_true", help="Only generate/download similarity map (skip masks)")
    parser.add_argument("--lat", type=float, help="Latitude for AOI center")
    parser.add_argument("--lon", type=float, help="Longitude for AOI center")
    parser.add_argument("--radius", type=float, help="Radius (in meters) for AOI")
    
    parser.add_argument("--background-method", choices=["sample_extent", "buffer"], help="Method to generate background points if presence-only data is provided.")
    parser.add_argument("--background-buffer", nargs=2, type=float, help="Min and Max distance (in meters) for buffer-based background sampling. e.g. --background-buffer 100 1000")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold Spatial Block Cross-Validation.")
    
    args = parser.parse_args()
    
    if args.mode == "extract":
        df = pd.read_csv(args.input)
        extractor = GEEExtractor(args.key, project=args.project)
        res = extractor.extract_embeddings(
            df, 
            scale=args.scale, 
            background_method=args.background_method,
            background_buffer=args.background_buffer
        )
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        res.to_csv(args.output, index=False)
        sys.stderr.write(f"Extraction complete at {args.scale}m. Results saved to {args.output}\n")
        
    elif args.mode == "analyze":
        df = pd.read_csv(args.input)
        import ee
        try:
            ee.Initialize(project=args.project) if args.project else ee.Initialize()
        except Exception:
            pass

        if args.method == "centroid":

            res = analyze_embeddings(df)
            # Save similarities
            res['clean_data']['similarity'] = res['similarities']
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            res['clean_data'].to_csv(args.output, index=False)
            
            # Save metadata
            meta = {
                "method": "centroid",
                "centroid": res['centroid'].tolist(),
                "metrics": res['metrics'],
                "cv_results": None
            }
            
            if args.cv:
                from autoSDM.trainer import run_parallel_cv
                import ee
                try: ee.Initialize()
                except: pass
                
                sys.stderr.write("Running 5-fold Spatial Block Cross-Validation (Centroid)...\n")
                meta["cv_results"] = run_parallel_cv(
                    df=df,
                    nuisance_vars=[],
                    ecological_vars=[f"A{i:02d}" for i in range(64)],
                    scale=args.scale
                )
        elif args.method == "maxent":
            # Maxent Mode - requires GEE for cloud training
            import ee
            from autoSDM.trainer import train_maxent_model, run_parallel_cv
            
            # Initialize GEE if not already done
            try:
                ee.Initialize()
            except Exception:
                pass 
            
            nuisance_vars = args.nuisance_vars.split(',') if args.nuisance_vars else []
            ecological_vars = [f"A{i:02d}" for i in range(64)]
            
            classifier, nuisance_optima, df_clean, metrics = train_maxent_model(
                df=df,
                nuisance_vars=nuisance_vars,
                ecological_vars=ecological_vars,
                key_path=args.key,
                scale=args.scale
            )
            
            # Run Parallel CV if requested
            cv_results = None
            if args.cv:
                sys.stderr.write("Running 5-fold Spatial Block Cross-Validation...\n")
                cv_results = run_parallel_cv(
                    df=df,
                    nuisance_vars=nuisance_vars,
                    ecological_vars=ecological_vars,
                    scale=args.scale
                )
            
            # Save cleaned data
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            df_clean.to_csv(args.output, index=False)
            
            # Save metadata
            meta = {
                "method": args.method,
                "classifier_serialized": classifier.serialize(),
                "nuisance_optima": nuisance_optima,
                "ecological_vars": ecological_vars,
                "nuisance_vars": nuisance_vars,
                "metrics": metrics,
                "cv_results": cv_results
            }

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
        
        # 1. Extract Embeddings for these specific points
        extractor = GEEExtractor(args.key, project=args.project)
        df_emb = extractor.extract_embeddings(df, scale=args.scale)
        
        if df_emb.empty:
            sys.stderr.write("Error: Point extraction returned no data.\n")
            sys.exit(1)
            
        # 2. Load Metadata
        with open(args.meta) as f:
            meta = json.load(f)
        
        method = meta.get('method', 'centroid')
        
        # 3. Optional Coarse Filter
        if args.coarse_meta:
            with open(args.coarse_meta) as f:
                coarse_meta = json.load(f)
            
            # Extract coarse embeddings for filtering
            df_coarse = extractor.extract_embeddings(df, scale=1000)
            if not df_coarse.empty:
                coarse_centroid = np.array(coarse_meta['centroid'])
                coarse_emb_cols = [f"A{i:02d}" for i in range(64)]
                coarse_sims = np.dot(df_coarse[coarse_emb_cols].values, coarse_centroid)
                
                # Filter df_emb based on coarse threshold
                valid_mask = coarse_sims >= coarse_meta.get('threshold_5pct', -1)
                df_emb = df_emb[valid_mask].copy()
                sys.stderr.write(f"Coarse filter dropped {np.sum(~valid_mask)} points.\n")

        # 4. Run Predictions
        emb_cols = [f"A{i:02d}" for i in range(64)]
        if method == "centroid":
            centroid = np.array(meta['centroid'])
            df_emb['similarity'] = np.dot(df_emb[emb_cols].values, centroid)
        
        elif method == "maxent":
            # Maxent requires GEE classification for the points
            import ee
            try: ee.Initialize(project=args.project) if args.project else ee.Initialize()
            except: pass
            
            classifier = ee.deserializer.fromJSON(meta['classifier_serialized'])
            ecological_vars = meta['ecological_vars']
            nuisance_vars = meta['nuisance_vars']
            nuisance_optima = meta['nuisance_optima']
            
            # Create FeatureCollection for the points with optimal nuisance values
            features = []
            for idx, row in df_emb.iterrows():
                props = {v: float(row[v]) for v in ecological_vars if v in row}
                for v, opt in nuisance_optima.items():
                    props[v] = float(opt)
                geom = ee.Geometry.Point([row['longitude'], row['latitude']])
                features.append(ee.Feature(geom, props).set('orig_index', str(idx)))
            
            fc = ee.FeatureCollection(features)
            
            from autoSDM.trainer import get_safe_scores_and_labels
            scores, orig_indices = get_safe_scores_and_labels(classifier, fc, 'orig_index', name="Maxent Prediction")
            
            # Map scores back to df_emb
            df_emb['similarity'] = np.nan
            for s, idx_str in zip(scores, orig_indices):
                df_emb.at[int(float(idx_str)), 'similarity'] = s

        # 5. Save Results
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        df_emb.to_csv(args.output, index=False)
        sys.stderr.write(f"Point predictions complete. Results saved to {args.output}\n")

    elif args.mode == "ensemble":
        if not args.meta:
            sys.stderr.write(f"Error: --meta required for {args.mode} mode\n")
            sys.exit(1)
            
        df = pd.read_csv(args.input)
        from autoSDM.extrapolate import get_prediction_image
        extractor = GEEExtractor(args.key, project=args.project) # Initialize GEE

        
        # Load Coarse Meta for Filtering (Optional)
        coarse_filter = None
        if args.coarse_meta:
            with open(args.coarse_meta) as f:
                coarse_meta = json.load(f)
            coarse_filter = {
                'centroid': coarse_meta['centroid'],
                'threshold': coarse_meta['threshold_5pct'] # Use 5pct of coarse for filtering
            }
        
        # Determine AOI if lat/lon/radius are provided
        import ee
        aoi = None
        if args.lat is not None and args.lon is not None and args.radius is not None:
            sys.stderr.write(f"Using specified AOI: {args.radius}m radius around {args.lat}, {args.lon}\n")
            aoi = ee.Geometry.Point([args.lon, args.lat]).buffer(args.radius)
        elif args.aoi_path:
            sys.stderr.write(f"Loading AOI from {args.aoi_path}...\n")
            import geopandas as gpd
            from shapely.geometry import mapping
            gdf = gpd.read_file(args.aoi_path)
            # Re-project to 4326 if needed
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # Combine all features into one geometry
            # If the GDF has many features, unary_union can be slow and create a massive payload.
            # We use the total bounds (envelope) for GEE processing to avoid the 10MB payload limit.
            bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
            aoi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
            sys.stderr.write(f"AOI simplified to bounding box: {bounds}\n")
        
        if args.mode == "ensemble":
            if not args.meta2:
                sys.stderr.write("Error: --meta2 is required for ensemble mode.\n")
                sys.exit(1)
            with open(args.meta) as f:
                meta1 = json.load(f)
            with open(args.meta2) as f:
                meta2 = json.load(f)
            
            img1, aoi = get_prediction_image(meta1, df, coarse_filter=coarse_filter, aoi=aoi)
            img2, _ = get_prediction_image(meta2, df, coarse_filter=coarse_filter, aoi=aoi)
            
            # Ensemble: Product of similarities (Agreement)
            prediction_map = img1.select('similarity').multiply(img2.select('similarity')).rename('similarity')
            thresholds = {}
        else:
            # Standard Extrapolate Mode
            with open(args.meta) as f:
                meta = json.load(f)
            
            prediction_map, aoi = get_prediction_image(meta, df, coarse_filter=coarse_filter, aoi=aoi)
            
            thresholds = {}
        
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
            
            # 0. Coarse Filter Layer (Debug)
            if coarse_filter:
                # Re-create the coarse mask image for visualization
                # We need to replicate the logic inside generate_prediction_map or assume it works
                # Let's just visualize the one used inside if we can access it? 
                # We can't easily access internal variables of the function.
                # Let's recreate it here for the map.
                yr = 2025
                img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filter(ee.Filter.calendarRange(yr, yr, 'year')).mosaic()
                coarse_centroid_img = ee.Image.constant(list(coarse_filter['centroid']))
                # Use native projection logic same as interpolate.py
                first_img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filter(ee.Filter.calendarRange(yr, yr, 'year')).first()
                native_proj = first_img.projection()
                
                coarse_dot = img.multiply(coarse_centroid_img).reduce(ee.Reducer.sum())
                coarse_mask_viz = coarse_dot.reproject(crs=native_proj, scale=1000).gte(coarse_filter['threshold'])
                layers.append(get_layer_config(coarse_mask_viz.selfMask(), {'palette': ['orange']}, "Coarse Mask (1km)"))

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
            
            with open(args.output, 'w') as f:
                json.dump(results, f)
                
            if args.gcs_bucket:
                sys.stderr.write(f"Export tasks started. Monitor via: https://code.earthengine.google.com/tasks\n")
            else:
                sys.stderr.write(f"Extrapolation complete at {args.scale}m. Maps saved to {base_dir}\n")

if __name__ == "__main__":
    main()
