# autoSDM: High-Resolution SDM with Alpha Earth Embeddings

`autoSDM` is a toolset for Species Distribution Modeling (SDM) that leverages Google's Alpha Earth satellite embeddings. By using 64-dimensional dense vectors representing the earth's surface characteristics, `autoSDM` simplifies covariate management and provides 10m-resolution mapping capabilities.

## Key Features

- **Presence-Only Modeling**: Automatically defaults to Similarity Search (Centroid) when only presence data is available.
- **Arithmetic Mean Centroids**: Uses the Arithmetic Mean to define environmental centroids in ecological niche space, ensuring rapid and robust characterization.
- **Presence-Absence Modeling**: Automatically applies Ridge Regression (Linear) when validated absences are present.
- **Flexible Mapping**: Generate individual distribution maps or combined "Agreement Maps" by ensembling multiple models (e.g., Centroid * Ridge).
- **Big-Data Ready**: Automatically handles large rasters using windowed, memory-efficient merging and parallelized downloads.

---

## Installation

To install autoSDM directly from GitHub:

```r
# install.packages("devtools")
devtools::install_github("James-Longo/autoSDM")
```

**Python Setup:**
autoSDM automatically handles Python dependencies via reticulate. On your first run, it will check your active Python environment and install required packages (earthengine-api, pandas, geopandas, etc.) if they are missing.

**Google Earth Engine:**
You must have a Google Earth Engine account. The package will attempt to auto-discover your GEE project or prompt you to select one from your authorized projects.

---

## Quick Start (R)

```r
library(autoSDM)

# 1. Format your data (Standardizes columns to longitude, latitude, year, present)
data <- format_data(raw_data, coords = c("lon", "lat"), year = "date", presence = "presence")

# 2. Run the pipeline
# Automatically detects if data is Presence-Only or Presence-Absence
# Generates a distribution map for the specified area of interest (AOI)
results <- autoSDM(data, aoi = list(lat=44.5, lon=-71.5, radius=10000))
```

---

## Core Pipeline

The pipeline is driven by three primary stages, which can be invoked together via the main R function or separately via the CLI.

### 1. Extract
Extracts Alpha Earth embeddings from Google Earth Engine for a set of occurrences.
- **Scale-Aware**: Native support for 10m, 100m, and 1000m scales.
- **Temporal Alignment**: Automatically matches observation years (2017-2025) to the correct embedding mosaic.

### 2. Analyze
Trains the selected model and determines ecological metrics.
- **Presence-Only (Centroid)**: Calculates the mean of presence embeddings and dot-product similarity.
- **Presence-Absence (Ridge)**: Trains a linear ridge regression model (presence=1, absence=-1).
- **Validation**: Automatically calculates continuous Boyce Index (CBI), AUC-ROC, and AUC-PR metrics.

### 3. Extrapolate
Projects models onto maps.
- **Efficient Generation**: Projects model weights directly on GEE servers where possible.
- **Parallel downloads**: High-throughput tiling system for downloading large AOIs.

---

## Output Structure

Results are organized into project-specific files within your specified results directory (e.g., `benchmarks/[species_name]/results/`):

```text
[project_name]/
├── centroid.tif    (Similarity Map - generated if PO data used)
├── ridge.tif       (Suitability Map - generated if PA data used)
├── centroid.json   (Model metadata, centroids, and validation metrics)
├── ridge.json      (Model weights and validation metrics)
└── results.json    (Combined summary for the run)
```

---

## License and Credits
Developed for Advanced Species Distribution Modeling. Leverages the Alpha Earth Embedding dataset provided by Google.
