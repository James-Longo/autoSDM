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

    def extract_embeddings(self, df, scale=10):
        """
        Extracts Alpha Earth embeddings for locations in a DataFrame.
        Expects lowercase column names: year, longitude, latitude
        """
        asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        
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
