import ee
import pandas as pd
import json
import os
import sys
import requests

def merge_rasters(file_list, output_filename):
    """
    Merges multiple GeoTIFF tiles into a single GeoTIFF using rasterio.
    Memory-efficient windowed approach.
    """
    if not file_list:
        return None
    if len(file_list) == 1:
        if file_list[0] != output_filename:
            if os.path.exists(output_filename):
                os.remove(output_filename)
            os.rename(file_list[0], output_filename)
        return output_filename
        
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    
    # First pass: Determine bounds and metadata
    src_files = []
    try:
        for f in file_list:
            if os.path.exists(f):
                src_files.append(rasterio.open(f))
        
        if not src_files:
            return None
            
        # Determine output bounds
        xs = []
        ys = []
        for src in src_files:
            left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)
        
        # Determine output dimensions at original resolution
        res = src_files[0].res
        width = int(round((dst_e - dst_w) / res[0]))
        height = int(round((dst_n - dst_s) / res[1]))
        transform = rasterio.transform.from_origin(dst_w, dst_n, res[0], res[1])
        
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": transform,
            "bigtiff": "IF_NEEDED" # Important for massive files
        })
        
        # Second pass: Write tile by tile
        with rasterio.open(output_filename, "w", **out_meta) as dst:
            for src in src_files:
                # Read data from source
                data = src.read()
                # Determine where it goes in the destination
                window = from_bounds(*src.bounds, transform=dst.transform)
                # Write to destination window
                dst.write(data, window=window)
                
        return output_filename
        
    finally:
        for src in src_files:
            src.close()

def generate_prediction_map(centroid, df=None, coarse_filter=None, aoi=None):
    """
    Returns:
        image: ee.Image with similarity band.
        aoi: ee.Geometry.
    """
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    
    # Get the latest Alpha Earth image (2024)
    yr = 2024
    img = ee.ImageCollection(asset_path)\
        .filter(ee.Filter.calendarRange(yr, yr, 'year'))\
        .mosaic()
        
    if img is None:
        raise RuntimeError(f"Could not find AlphaEarth image for year {yr}")

    if aoi is None:
        if df is None:
            raise ValueError("Either 'df' or 'aoi' must be provided to determine the mapping extent.")
        # Calculate AOI from survey data
        min_lat = df['Latitude'].min()
        max_lat = df['Latitude'].max()
        min_lon = df['Longitude'].min()
        max_lon = df['Longitude'].max()
        
        # Add a small buffer to the AOI (approx 1km)
        buffer = 0.01 
        aoi = ee.Geometry.Rectangle([
            min_lon - buffer, 
            min_lat - buffer, 
            max_lon + buffer, 
            max_lat + buffer
        ])
    
    # Create centroid constant image
    centroid_img = ee.Image.constant(list(centroid))
    
    # Calculate Dot Product (Cosine Similarity)
    dot_product = img.multiply(centroid_img).reduce(ee.Reducer.sum()).rename('similarity')
    
    # --- Coarse Filtering Logic ---
    if coarse_filter:
        sys.stderr.write("Applying coarse scale (1km) filter...\n")
        coarse_centroid_img = ee.Image.constant(list(coarse_filter['centroid']))
        coarse_dot = img.multiply(coarse_centroid_img).reduce(ee.Reducer.sum())
        
        # We avoid explicit .reproject(scale=1000) here because it causes 
        # "Reprojection output too large" errors during large-scale exports.
        # GEE will handle the sampling automatically.
        coarse_mask = coarse_dot.gte(coarse_filter['threshold'])
        
        dot_product = dot_product.updateMask(coarse_mask)

    return dot_product, aoi

def generate_classifier_prediction_map(classifier, df=None, ecological_vars=None, nuisance_optima=None, coarse_filter=None, aoi=None):
    """
    Generates a prediction map for a classifier (RF, Maxent, etc.) at standardized nuisance levels.
    
    Args:
        classifier: Trained ee.Classifier.
        df: DataFrame for AOI (optional if aoi is provided).
        ecological_vars: List of ecological (Alpha Earth) band names.
        nuisance_optima: Dict of {band_name: optimal_value}.
        coarse_filter: Optional coarse filter dict.
        aoi: ee.Geometry (optional).
    
    Returns:
        image: ee.Image with probability band.
        aoi: ee.Geometry.
    """
    asset_path = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    yr = 2024
    img = ee.ImageCollection(asset_path)\
        .filter(ee.Filter.calendarRange(yr, yr, 'year'))\
        .mosaic()
        
    if img is None:
        raise RuntimeError(f"Could not find AlphaEarth image for year {yr}")

    if aoi is None:
        if df is None:
            raise ValueError("Either 'df' or 'aoi' must be provided to determine the mapping extent.")
        # Calculate AOI
        min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()
        min_lon, max_lon = df['Longitude'].min(), df['Longitude'].max()
        buffer = 0.01 
        aoi = ee.Geometry.Rectangle([min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer])
    
    # 1. Start with ecological stack
    prediction_stack = img.select(ecological_vars)
    
    # 2. Add constant nuisance bands
    for var, val in nuisance_optima.items():
        constant_img = ee.Image.constant(val).rename(var)
        prediction_stack = prediction_stack.addBands(constant_img)
        
    # 3. Classify and take the first band (which is the probability/classification)
    standardized_map = prediction_stack.classify(classifier).select(0).rename('similarity')
    
    # 4. Apply Coarse Filter if provided
    if coarse_filter:
        sys.stderr.write("Applying coarse scale (1km) filter...\n")
        coarse_centroid_img = ee.Image.constant(list(coarse_filter['centroid']))
        coarse_dot = img.multiply(coarse_centroid_img).reduce(ee.Reducer.sum())
        coarse_mask = coarse_dot.gte(coarse_filter['threshold'])
        standardized_map = standardized_map.updateMask(coarse_mask)

    return standardized_map, aoi

def get_prediction_image(meta, df=None, coarse_filter=None, aoi=None):
    """
    Reconstructs an ee.Image prediction from metadata.
    """
    method = meta.get('method', 'centroid')
    if method == "centroid":
        return generate_prediction_map(
            centroid=meta['centroid'],
            df=df,
            coarse_filter=coarse_filter,
            aoi=aoi
        )
    else:
        classifier = ee.deserializer.fromJSON(meta['classifier_serialized'])
        return generate_classifier_prediction_map(
            classifier=classifier,
            df=df,
            ecological_vars=meta['ecological_vars'],
            nuisance_optima=meta['nuisance_optima'],
            coarse_filter=coarse_filter,
            aoi=aoi
        )

import math

def get_grid_dimensions(aoi, scale):
    """
    Estimates pixel dimensions of the AOI at the given scale.
    """
    coords = aoi.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Approx degrees to meters
    width_m = (max_lon - min_lon) * 111320 * math.cos(math.radians((min_lat + max_lat) / 2))
    height_m = (max_lat - min_lat) * 111320
    
    w_px = width_m / scale
    h_px = height_m / scale
    
    return w_px, h_px, (min_lon, min_lat, max_lon, max_lat)

import concurrent.futures

def download_mask(mask, aoi, scale, filename):
    """
    Downloads the mask. If dimensions exceed GEE limits (10k px), 
    it automatically tiles the download using parallel workers.
    Returns a list of downloaded file paths.
    """
    w_px, h_px, bounds = get_grid_dimensions(aoi, scale)
    min_lon, min_lat, max_lon, max_lat = bounds
    
    CHUNK_SIZE = 500
    
    # Check if single tile is sufficient
    if w_px <= CHUNK_SIZE and h_px <= CHUNK_SIZE:
        if _download_single_tile(mask, aoi, scale, filename):
            return [filename]
        return []

    # Tiled download
    sys.stderr.write(f"Mask too large ({int(w_px)}x{int(h_px)} px) for single download. Tiling...\n")
    
    n_x = math.ceil(w_px / CHUNK_SIZE)
    n_y = math.ceil(h_px / CHUNK_SIZE)
    
    lon_step = (max_lon - min_lon) / (w_px / CHUNK_SIZE)
    lat_step = (max_lat - min_lat) / (h_px / CHUNK_SIZE)
    
    name_root, ext = os.path.splitext(filename)
    tasks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for i in range(n_x):
            for j in range(n_y):
                x0 = min_lon + i * lon_step
                x1 = min_lon + (i + 1) * lon_step # simpler arithmetic
                # Clamp to max bounds to avoid floating point overshoot
                if i == n_x - 1: x1 = max_lon
                
                y0 = min_lat + j * lat_step
                y1 = min_lat + (j + 1) * lat_step
                if j == n_y - 1: y1 = max_lat
                
                tile_aoi = ee.Geometry.Rectangle([x0, y0, x1, y1])
                tile_name = f"{name_root}_{i}_{j}{ext}"
                
                tasks.append(executor.submit(_download_single_tile, mask, tile_aoi, scale, tile_name))
    
    return [t.result() for t in tasks if t.result()]

def export_to_gcs(image, region, scale, bucket, filename_prefix):
    """
    Exports an image to Google Cloud Storage as Cloud Optimized GeoTIFFs (COG).
    Uses EPSG:3857 (native for Alpha Earth) to avoid reprojection overhead.
    """
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"export_{filename_prefix}_{scale}m",
        bucket=bucket,
        fileNamePrefix=filename_prefix,
        scale=scale,
        region=region,
        crs='EPSG:3857', # Native projection for Alpha Earth embeddings
        fileFormat='GeoTIFF',
        formatOptions={
            'cloudOptimized': True
        },
        maxPixels=1e13,
        shardSize=4096 # Balanced size for compute stability (reduced from 10000)
    )
    task.start()
    sys.stderr.write(f"Started GEE export task: {task.id} -> gs://{bucket}/{filename_prefix}*\n")
    return task

def _download_single_tile(mask, region, scale, filename):
    import time
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return filename
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            name = os.path.splitext(os.path.basename(filename))[0]
            url = mask.getDownloadURL({
                'name': name,
                'scale': scale,
                'region': region,
                'format': 'GeoTIFF'
            })
            
            response = requests.get(url, stream=True, timeout=120)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                sys.stderr.write(f"Downloaded {filename}\n")
                return filename
            elif response.status_code == 400:
                # Likely an empty tile (masked out)
                return None
            else:
                sys.stderr.write(f"Attempt {attempt+1} failed for {filename}: HTTP {response.status_code}\n")
                
        except Exception as e:
            sys.stderr.write(f"Attempt {attempt+1} error for {filename}: {str(e)}\n")
        
        if attempt < max_retries - 1:
            time.sleep(5)
            
    return None

