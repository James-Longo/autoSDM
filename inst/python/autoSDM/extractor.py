import pandas as pd
import json
import os
import sys
import ee

class GEEExtractor:
    def __init__(self, json_key_path=None):
        if not json_key_path:
            json_key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
        
        if json_key_path and os.path.exists(json_key_path):
            with open(json_key_path) as f:
                key_data = json.load(f)
                self.sa_email = key_data["client_email"]
            
            credentials = ee.ServiceAccountCredentials(self.sa_email, json_key_path)
            ee.Initialize(credentials)
            sys.stderr.write(f"Initialized GEE with service account: {self.sa_email}\n")
        else:
            # Attempt to use session auth if no key is provided
            try:
                ee.Initialize()
                sys.stderr.write("Initialized GEE using default session credentials.\n")
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
        if background_method:
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

        # Alpha Earth range: 2017-2024
        df = df[(df['year'] >= 2017) & (df['year'] <= 2024)].copy()
        if df.empty:
            sys.stderr.write("No locations within the 2017-2024 year range found.\n")
            return pd.DataFrame()

        years = sorted(df['year'].unique())
        all_results = []
        
        for yr in years:
            yr_df = df[df['year'] == yr].copy()
            sys.stderr.write(f"Processing {len(yr_df)} points for year {int(yr)} at scale {scale}m...\n")
            
            img = ee.ImageCollection(asset_path)\
                .filter(ee.Filter.calendarRange(int(yr), int(yr), 'year'))\
                .mosaic()
            
            CHUNK_SIZE = 4000
            features = []
            for idx, row in yr_df.iterrows():
                geom = ee.Geometry.Point([row['longitude'], row['latitude']])
                feat = ee.Feature(geom, {'orig_index': str(idx)})
                features.append(feat)
            
            for i in range(0, len(features), CHUNK_SIZE):
                chunk = features[i:i + CHUNK_SIZE]
                fc = ee.FeatureCollection(chunk)
                
                sampled = img.reduceRegions(
                    collection=fc,
                    reducer=ee.Reducer.mean(),
                    scale=scale
                )
                
                try:
                    res = sampled.getInfo()
                    found_any = False
                    
                    for feat in res['features']:
                        props = feat['properties']
                        if 'A00' in props and props['A00'] is not None:
                            idx = int(props.pop('orig_index'))
                            for k, v in props.items():
                                yr_df.at[idx, k] = v
                            found_any = True
                    
                    if not found_any:
                        sys.stderr.write(f"Warning: No valid embeddings found for {int(yr)} chunk {i//CHUNK_SIZE}\n")
                        
                except Exception as e:
                    sys.stderr.write(f"Error during GEE sampling for {int(yr)} chunk {i//CHUNK_SIZE}: {str(e)}\n")
            
            all_results.append(yr_df)
            
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results)

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
        
        if method == "sample_extent":
            bounds = [
                presence_df['longitude'].min(),
                presence_df['latitude'].min(),
                presence_df['longitude'].max(),
                presence_df['latitude'].max()
            ]
            sampling_region = ee.Geometry.Rectangle(bounds).difference(exclusion_mask)
            bg_fc = ee.FeatureCollection.randomPoints(sampling_region, n_bg)
        elif method == "buffer" and buffer_params:
            min_dist, max_dist = buffer_params
            
            # Use mapping to sample 10 points per presence point
            def sample_per_point(feat):
                p = feat.geometry()
                # Local ring for this point
                ring = p.buffer(max_dist).difference(exclusion_mask)
                return ee.FeatureCollection.randomPoints(ring, 10)

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
        
        # 5. Assign years randomly from presence distribution
        presence_years = presence_df['year'].values
        bg_df['year'] = np.random.choice(presence_years, size=len(bg_df))
        
        sys.stderr.write(f"Generated {len(bg_df)} background points.\n")
        return bg_df
