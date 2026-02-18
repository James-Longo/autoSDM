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


    def extract_embeddings(self, df, scale=10):
        """
        Extracts Alpha Earth embeddings for locations in a DataFrame.
        Expects lowercase column names: year, longitude, latitude
        """
        asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        
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
        
        chunk_size = 2000
        total_count = len(df)
        num_chunks = (total_count + chunk_size - 1) // chunk_size
        max_workers = 5  
        
        sys.stderr.write(f"Processing {total_count} points in {num_chunks} chunks (size {chunk_size}, {max_workers} parallel)...\n")
        
        # Collect result dataframes to merge at the end
        results_list = []
        df_lock = threading.Lock()
        completed_count = [0]
        
        def process_chunk(chunk_idx, chunk_df):
            try:
                chunk_extracted_data = []
                # Group by year to minimize ImageCollection filtering
                for yr, yr_df in chunk_df.groupby("year"):
                    if yr_df.empty: continue
                    
                    # Efficient feature creation
                    features = []
                    half_s = scale / 2
                    for idx, row in yr_df.iterrows():
                        # exact buffer/bounds for publication quality
                        geom = ee.Geometry.Point([row['longitude'], row['latitude']]).buffer(scale/2).bounds()
                        features.append(ee.Feature(geom, {'orig_index': int(idx)}))
                    
                    fc = ee.FeatureCollection(features)
                    
                    img = ee.ImageCollection(asset_path)\
                        .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))\
                        .mosaic()
                    
                    # Force native 10m scale for the reduction to ensure every pixel is counted
                    reduced = img.reduceRegions(
                        collection=fc,
                        reducer=ee.Reducer.mean(),
                        scale=10 
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
                    percent = (completed_count[0] / num_chunks) * 100
                    sys.stderr.write(f"\rProgress: {percent:.1f}% ({completed_count[0]}/{num_chunks} chunks)")
                    sys.stderr.flush()
                    if completed_count[0] == num_chunks:
                        sys.stderr.write("\n")
            except Exception as e:
                with df_lock:
                    completed_count[0] += 1
                sys.stderr.write(f"\nError in chunk {chunk_idx}: {e}\n")
        
        # Submit all chunks concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_futures = []
            for i in range(0, total_count, chunk_size):
                chunk_df = df.iloc[i : i + chunk_size].copy()
                chunk_idx = i // chunk_size + 1
                chunk_futures.append(executor.submit(process_chunk, chunk_idx, chunk_df))
            
            for f in as_completed(chunk_futures):
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
