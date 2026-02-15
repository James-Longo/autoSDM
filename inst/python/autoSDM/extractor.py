import pandas as pd
import json
import os
import sys
import ee
import numpy as np

class GEEExtractor:
    def __init__(self, json_key_path=None, project=None):
        if not json_key_path:
            json_key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
        
        if json_key_path and os.path.exists(json_key_path):
            with open(json_key_path) as f:
                key_data = json.load(f)
                self.sa_email = key_data["client_email"]
                # Try to pick up project from key if not provided
                if not project:
                    project = key_data.get("project_id")
            
            credentials = ee.ServiceAccountCredentials(self.sa_email, json_key_path)
            ee.Initialize(credentials, project=project)
            sys.stderr.write(f"Initialized GEE with service account: {self.sa_email} (Project: {project})\n")
        else:
            # Attempt to use session auth if no key is provided
            try:
                ee.Initialize(project=project)
                sys.stderr.write(f"Initialized GEE using default session credentials (Project: {project}).\n")
            except Exception as e:
                sys.stderr.write(f"GEE Initialization failed: {e}\n")
                sys.stderr.write("Please run 'earthengine authenticate' or provide a service account key.\n")
                raise


    def extract_embeddings(self, df, scale=10, background_method=None, background_buffer=None):
        """
        Extracts Alpha Earth embeddings for locations in a DataFrame.
        Expects lowercase column names: year, longitude, latitude
        """
        asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        
        # 1. Background Generation (if presence-only and requested)
        needs_bg = False
        if background_method and background_method != "none":
            if 'present' not in df.columns:
                needs_bg = True
                df['present'] = 1
            elif df['present'].nunique() == 1 and df['present'].iloc[0] == 1:
                needs_bg = True
            elif (df['present'] == 1).all():
                needs_bg = True
        
        if needs_bg:
            bg_df = self.generate_background_points(
                presence_df=df[df['present'] == 1],
                method=background_method,
                buffer_params=background_buffer,
                scale=scale
            )
            df = pd.concat([df, bg_df], ignore_index=True)

        # Alpha Earth range: 2017-2025
        before_yr_count = len(df)
        df = df[(df['year'] >= 2017) & (df['year'] <= 2025)].copy()
        after_yr_count = len(df)
        
        if after_yr_count < before_yr_count:
            if after_yr_count == 0:
                sys.stderr.write("No locations within the 2017-2025 year range found.\n")
                return pd.DataFrame()
            else:
                sys.stderr.write(f"Dropped {before_yr_count - after_yr_count} points outside Alpha Earth range (2017-2025).\n")

        years = sorted(df['year'].unique())
        
        # CLIENT-SIDE CHUNKING with CONCURRENT processing
        # Use sampleRegions (optimized for points) instead of reduceRegions
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        chunk_size = 1000
        total_count = len(df)
        num_chunks = (total_count + chunk_size - 1) // chunk_size
        max_workers = 4  
        
        sys.stderr.write(f"Processing {total_count} points in {num_chunks} chunks (size {chunk_size}, {max_workers} parallel)...\n")
        
        # Collect result dataframes to merge at the end
        results_list = []
        df_lock = threading.Lock()
        completed_count = [0]
        
        def process_chunk(chunk_idx, chunk_df):
            try:
                chunk_years = sorted(chunk_df['year'].unique())
                chunk_extracted_data = []
                
                for yr in chunk_years:
                    yr_df = chunk_df[chunk_df['year'] == yr]
                    if yr_df.empty: continue
                    
                    features = []
                    for idx, row in yr_df.iterrows():
                        # Proper aggregate mean: buffer the point into a 640m box
                        geom = ee.Geometry.Point([row['longitude'], row['latitude']]).buffer(scale/2).bounds()
                        features.append(ee.Feature(geom, {'orig_index': int(idx)}))
                    
                    if not features: continue
                    fc = ee.FeatureCollection(features)
                    
                    img = ee.ImageCollection(asset_path)\
                        .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))\
                        .mosaic()
                    
                    # Compute mean of ALL land pixels in the box
                    reduced = img.reduceRegions(
                        collection=fc,
                        reducer=ee.Reducer.mean(),
                        scale=10 # Native 10m scale ensures accuracy
                    )
                    
                    res = reduced.getInfo()['features']
                    for feat in res:
                        props = feat['properties']
                        if 'A00' in props and props['A00'] is not None:
                            chunk_extracted_data.append(props)
                
                if chunk_extracted_data:
                    chunk_res_df = pd.DataFrame(chunk_extracted_data)
                    with df_lock:
                        results_list.append(chunk_res_df)
                
                with df_lock:
                    completed_count[0] += 1
                    sys.stderr.write(f"Completed chunk {completed_count[0]}/{num_chunks}\n")
            except Exception as e:
                with df_lock:
                    completed_count[0] += 1
                sys.stderr.write(f"Error in chunk {chunk_idx}: {e}\n")
        
        # Submit all chunks concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, total_count, chunk_size):
                chunk_df = df.iloc[i : i + chunk_size].copy()
                chunk_idx = i // chunk_size + 1
                futures.append(executor.submit(process_chunk, chunk_idx, chunk_df))
            
            # Wait for all to finish
            for f in as_completed(futures):
                pass
        
        # Prepare columns in main DataFrame if they don't exist
        emb_cols = [f"A{i:02d}" for i in range(64)]
        for col in emb_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Batch update the main DataFrame
        if results_list:
            # Drop duplicates in results if any (though orig_index should be unique)
            final_updates = pd.concat(results_list).drop_duplicates(subset='orig_index').set_index('orig_index')
            df.update(final_updates)
        
        # Post-processing: Drop failed rows
        emb_cols = [f"A{i:02d}" for i in range(64)]
        if all(col in df.columns for col in emb_cols):
            before_count = len(df)
            df = df.dropna(subset=emb_cols).copy()
            after_count = len(df)
            if after_count < before_count:
                sys.stderr.write(f"Dropped {before_count - after_count} points where Alpha Earth embeddings were unavailable.\n")
        
        return df

    def generate_background_points(self, presence_df, method="sample_extent", buffer_params=None, scale=10):
        """
        Generates background points for presence-only data using GEE.
        """
        import numpy as np
        
        sys.stderr.write(f"Generating background points (method={method}, scale={scale}m)...\n")
        
        # 1. Prepare presence geometries
        features = []
        for _, row in presence_df.iterrows():
            features.append(ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']])))
        presence_fc = ee.FeatureCollection(features)
        presence_geom_union = presence_fc.geometry()

        # 2. Define exclusion distance (at least resolution's distance)
        exclusion_dist = scale
        if method == "buffer" and buffer_params:
            # For buffer mode, ensure min distance applies
            exclusion_dist = max(scale, buffer_params[0])
        
        exclusion_mask = presence_geom_union.buffer(exclusion_dist)

        # 3. Define sampling region and count
        n_pres = len(presence_df)
        n_bg = n_pres * 10
        
        # Determine deterministic seed from presence data
        # Use sum of coordinates as a simple hash
        seed = int(abs(presence_df['longitude'].sum() + presence_df['latitude'].sum()) * 10000) % 1000000
        sys.stderr.write(f"Using deterministic seed: {seed}\n")

        if method == "sample_extent":
            bounds = [
                presence_df['longitude'].min(),
                presence_df['latitude'].min(),
                presence_df['longitude'].max(),
                presence_df['latitude'].max()
            ]
            sampling_region = ee.Geometry.Rectangle(bounds).difference(exclusion_mask)
            bg_fc = ee.FeatureCollection.randomPoints(sampling_region, n_bg, seed=seed)
        elif method == "buffer" and buffer_params:
            min_dist, max_dist = buffer_params
            
            # Use mapping to sample 10 points per presence point
            def sample_per_point(feat):
                p = feat.geometry()
                # Local ring for this point
                ring = p.buffer(max_dist).difference(exclusion_mask)
                # Note: randomPoints seed in map() is tricky, we use a per-feature offset
                # GEE doesn't allow dynamic seeds easily in map, but feature ID helps
                return ee.FeatureCollection.randomPoints(ring, 10, seed=seed)

            bg_fc = presence_fc.map(sample_per_point).flatten()
        else:
            raise ValueError(f"Invalid background method: {method}")

        # 4. Extract coordinates from GEE in chunks to avoid the 5000 feature limit
        bg_coords = []
        chunk_size = 4000
        
        # Get total size safely
        try:
            total_count = int(bg_fc.size().getInfo())
        except:
            total_count = n_bg

        bg_list = bg_fc.toList(total_count)
        for i in range(0, total_count, chunk_size):
            sys.stderr.write(f"Retrieving background chunk {i//chunk_size + 1}...\n")
            chunk = ee.List(bg_list.slice(i, i + chunk_size))
            features = chunk.getInfo()
            for feat in features:
                coords = feat['geometry']['coordinates']
                bg_coords.append({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'present': 0
                })
        
        if not bg_coords:
            sys.stderr.write("Warning: No background points could be generated. Check AOI/extent.\n")
            return pd.DataFrame()

        bg_df = pd.DataFrame(bg_coords)
        
        # 5. Assign years randomly from presence distribution (deterministic)
        presence_years = presence_df['year'].values
        rng = np.random.RandomState(seed)
        bg_df['year'] = rng.choice(presence_years, size=len(bg_df))
        
        sys.stderr.write(f"Generated {len(bg_df)} background points.\n")
        return bg_df
