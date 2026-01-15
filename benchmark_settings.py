import os
import time
import json
import pandas as pd
import ee
import math
import concurrent.futures
from autoSDM.extrapolate import get_prediction_image, merge_rasters

# We need to re-implement download_mask essentially to allow parameter injection
def download_mask_benchmark(mask, aoi, scale, filename, chunk_size, max_workers):
    import autoSDM.extrapolate as interp
    
    w_px, h_px, bounds = interp.get_grid_dimensions(aoi, scale)
    min_lon, min_lat, max_lon, max_lat = bounds
    
    if w_px <= chunk_size and h_px <= chunk_size:
        # For benchmark we want to force tiling or at least test the overhead
        pass

    n_x = math.ceil(w_px / chunk_size)
    n_y = math.ceil(h_px / chunk_size)
    
    lon_step = (max_lon - min_lon) / (w_px / chunk_size)
    lat_step = (max_lat - min_lat) / (h_px / chunk_size)
    
    name_root, ext = os.path.splitext(filename)
    tasks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(n_x):
            for j in range(n_y):
                x0 = min_lon + i * lon_step
                x1 = min_lon + (i + 1) * lon_step
                if i == n_x - 1: x1 = max_lon
                
                y0 = min_lat + j * lat_step
                y1 = min_lat + (j + 1) * lat_step
                if j == n_y - 1: y1 = max_lat
                
                tile_aoi = ee.Geometry.Rectangle([x0, y0, x1, y1])
                tile_name = f"{name_root}_{i}_{j}{ext}"
                
                tasks.append(executor.submit(interp._download_single_tile, mask, tile_aoi, scale, tile_name))
    
    return [t.result() for t in tasks if t.result()]

def run_benchmarks():
    print("--- Benchmark: Testing Download Settings (Scale 100m) ---")
    
    from autoSDM.extractor import GEEExtractor
    extractor = GEEExtractor(os.environ.get("GEE_SERVICE_ACCOUNT_KEY"))
        
    input_csv = "comparison_extracted_100m.csv"
    df = pd.read_csv(input_csv)
    
    # Use centroid metadata
    with open("outputs/speed_test_100m/centroid_results.csv.json") as f:
        meta = json.load(f)
        
    img, aoi = get_prediction_image(meta, df)
    mask = img.select('similarity')

    combinations = [
        (500, 8),   # Original
        (500, 32),  # More workers, small tiles
        (1000, 8),  # Bigger tiles, few workers
        (1000, 32), # Bigger tiles, more workers
        (1500, 32)  # Even bigger tiles
    ]
    
    results = []
    
    for chunk_size, max_workers in combinations:
        print(f"\nTesting: CHUNK_SIZE={chunk_size}, MAX_WORKERS={max_workers}")
        
        test_dir = f"outputs/bench_{chunk_size}_{max_workers}"
        os.makedirs(test_dir, exist_ok=True)
        filename = os.path.join(test_dir, "test.tif")
        
        # Clear existing files to avoid cache effects (though GEE is remote)
        for f in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, f))
            
        t0 = time.time()
        try:
            files = download_mask_benchmark(mask, aoi, 100, filename, chunk_size, max_workers)
            # We don't merge for the benchmark to isolate download speed
            duration = time.time() - t0
            print(f"Result: {len(files)} tiles downloaded in {duration:.2f}s")
            results.append({
                "chunk_size": chunk_size,
                "max_workers": max_workers,
                "duration": duration,
                "tile_count": len(files)
            })
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                "chunk_size": chunk_size,
                "max_workers": max_workers,
                "duration": -1,
                "error": str(e)
            })

    print("\n--- Final Summary ---")
    print(f"{'ChunkSize':<10} | {'Workers':<8} | {'Duration':<10} | {'Tiles':<6}")
    print("-" * 45)
    for r in results:
        t_str = f"{r['duration']:.2f}s" if r['duration'] > 0 else "FAILED"
        print(f"{r['chunk_size']:<10} | {r['max_workers']:<8} | {t_str:<10} | {r.get('tile_count', 0):<6}")

if __name__ == "__main__":
    run_benchmarks()
