library(terra)
library(jsonlite)

# Paths
results_dir <- "C:/Users/james/OneDrive/Documents/autoSDM/outputs/r studio tested/US + NB x QC"
ensemble_json <- file.path(results_dir, "ensemble_results_10m.json")
benchmark_shp <- "C:/Users/james/OneDrive/Documents/autoSDM/benchmarks/doi_10_5061_dryad_m0cfxppgc__v20250624/BITH_Threshold_2025/BITH_Threshold_2024.shp"

# Load ensemble info and tiles
tiles <- list.files(results_dir, pattern = "autoSDM_ensemble_10m_.*\\.tif$", full.names = TRUE)
message(sprintf("Found %d tiles.", length(tiles)))

# Create VRT and Load
vrt_file <- tempfile(fileext = ".vrt")
vrt(tiles, vrt_file)
r <- rast(vrt_file)

# Load and align benchmark shapefile
s <- vect(benchmark_shp)
if (crs(r) != crs(s)) {
  message("Reprojecting shapefile...")
  s <- project(s, r)
}

# Efficiency: Crop prediction to the benchmark area
message("Cropping and rasterizing benchmark...")
r_crop <- crop(r, s)
s_rast <- rasterize(s, r_crop, field = 1, background = 0)

# --- Threshold Optimization ---
message("Optimizing threshold for max IoU...")

# Define threshold candidates (from 0.05 to 0.95)
thresholds <- seq(0.05, 0.95, by = 0.05)

# Calculate IoU for each threshold
iou_values <- sapply(thresholds, function(t) {
  r_bin <- r_crop >= t
  
  # Calculate Intersection and Union
  inter <- global((r_bin == 1) & (s_rast == 1), "sum", na.rm=TRUE)$sum
  uni   <- global((r_bin == 1) | (s_rast == 1), "sum", na.rm=TRUE)$sum
  
  return(if(uni > 0) inter / uni else 0)
})

# Find the best threshold
best_idx <- which.max(iou_values)
best_threshold <- thresholds[best_idx]
best_iou <- iou_values[best_idx]

message(sprintf("Best Threshold Found: %f", best_threshold))
message(sprintf("Maximum IoU: %f", best_iou))

# --- Final Metrics with Optimal Threshold ---
r_final_bin <- r_crop >= best_threshold

tp <- global((r_final_bin == 1) & (s_rast == 1), "sum", na.rm = TRUE)$sum
fp <- global((r_final_bin == 1) & (s_rast == 0), "sum", na.rm = TRUE)$sum
fn <- global((r_final_bin == 0) & (s_rast == 1), "sum", na.rm = TRUE)$sum
tn <- global((r_final_bin == 0) & (s_rast == 0), "sum", na.rm = TRUE)$sum

message(sprintf("Final Sensitivity: %f", tp / (tp + fn)))
message(sprintf("Final Specificity: %f", tn / (tn + fp)))

# Optional: Save the best binary map
# writeRaster(r_final_bin, file.path(results_dir, "optimized_bith_map.tif"), overwrite=TRUE)