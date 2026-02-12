# autoSDM Detailed Walkthrough

This guide covers the advanced usage of the `autoSDM` pipeline, including multi-model comparisons and high-resolution ensembling.

## 1. Simplified Workflow (R)
 
 The easiest way to use `autoSDM` is via the R interface, which consolidates the entire process into a single function call.
 
 ```r
 library(autoSDM)
 
 # 1. Prepare your data
 # Rename columns and ensure correct types
 formatted_data <- format_data(
   raw_data, 
   coords = c("Longitude", "Latitude"), 
   year = "Year", 
   presence = "present?"
 )
 
 # 2. Run Ensemble Analysis
 # This single command performs:
 # - Embedding extraction
 # - Centroid Analysis (geometric median)
 # - Maxent Analysis (bias-corrected)
 # - Ensemble Extrapolation (Agreement Map)
 results <- autoSDM(
   formatted_data, 
   aoi = list(lat=44.3, lon=-71.3, radius=5000),
   nuisance_vars = c("ObserverID", "Time"),
   scale = 10
 )
 ```
 
 ## 2. CLI Usage (Advanced)
 
 If you prefer the command line or need granular control over each step:

### 2.1. Data Preparation

Your CSV should contain `Latitude`, `Longitude`, `Year`, and a presence column (e.g., `present?`). 
You can include nuisance variables to account for sampling bias.

### 2.2. Extraction & Robust Analysis

We extract embeddings at our target resolution. `autoSDM` uses the **Geometric Median** for centroid analysis, which is the "central" point in 64-dimensional space, effectively ignoring outliers in your presence data.

```bash
# 1. Extract 100m embeddings
python -m autoSDM.cli extract --input sightings.csv --output extraction_100m.csv --scale 100

# 2. Analyze using Centroid Method
python -m autoSDM.cli analyze --input extraction_100m.csv --output centroid_results.csv --method centroid
```

## 3. Advanced Modeling (Maxent)

For presence-absence data, you can train a Maxent model. `autoSDM` handles nuisance standardization automatically by holding non-ecological variables at their "optima" during map generation.

```bash
# Analyze using Maxent with Nuisance variables
python -m autoSDM.cli analyze --input extraction_100m.csv --output maxent_results.csv --method maxent --nuisance ObserverID,Count.Time1
```

## 4. High-Resolution Extrapolation

Mapping at 10m involves downloading thousands of tiles. The pipeline handles this automatically:
- **32 Parallel Workers**: High-speed concurrent downloads.
- **Windowed Merging**: Merges tiles tile-by-tile to disk, allowing you to create 20GB+ maps on standard hardware.
- **Agreement Mapping**: Create ensemble maps between two model types.

```bash
# Create individual maps (Similarity & Probability)
python -m autoSDM.cli extrapolate --input centroid_results.csv --output centroid_map.json --meta centroid_results.csv.json --scale 10 --prefix centroid

# Create an Ensemble Map (Maxent * Centroid)
# This highlights areas where both models agree suitability is high.
python -m autoSDM.cli ensemble --input extraction_10m.csv --meta maxent_results.csv.json --meta2 centroid_results.csv.json --output ensemble_results.json --scale 10 --prefix ensemble
```

## 5. Performance Optimization

The CLI includes several flags for efficiency:
- `--only-similarity`: Skips binary mask generation to save download time and disk space.
- `--zip`: Packages all generated GeoTIFFs into a single archive automatically.
- `--key`: Specify your GEE Service Account key path directly.

## 6. Interpreting the Outputs

### `centroid_[scale].tif`
A continuous map where values represent the cosine similarity to the species' Geometric Median centroid. Values closer to 1.0 indicate a closer match to the species' average niche.

### `maxent_[scale].tif`
A probability map (0-1) representing the likelihood of presence based on the trained AMNH-Maxent classifier, standardized for detection bias.

### `ensemble_[scale].tif`
The product of two models. This map is more conservative than either individual model, as it requires *both* models to agree for an area to show high suitability.

---

## Technical Note: Memory Constraints
When running at 10m scale for large regions, the resulting rasters can exceed 10GB in size. Always ensure you have sufficient disk space. The pipeline uses windowed raster I/O to ensure RAM usage remains constant regardless of the map size.
