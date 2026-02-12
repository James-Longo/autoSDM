# autoSDM: High-Resolution SDM with Alpha Earth Embeddings

`autoSDM` is a state-of-the-art toolset for Species Distribution Modeling (SDM) that leverages Google's **Alpha Earth satellite embeddings**. By using 64-dimensional dense vectors representing the earth's surface characteristics, `autoSDM` eliminates the need for manual covariate selection and provides unprecedented 10m-resolution mapping capabilities.

## 🎯 Key Features

- **Multi-Model Support**: Choose between **Centroid** (Presence-Only) and **Maxent** (Presence/Absence) approaches.
- **Robust Characterization**: Uses the **Geometric Median** to define the species' environmental centroid, providing high resistance to outliers in ecological niche space.
- **Unified Ensemble**: Generate "Agreement Maps" by ensembling multiple models (e.g., Centroid $\times$ Maxent) to highlight high-confidence habitat.
- **Hierarchical Filtering**: Efficiency at scale. Use 1000m coarse models to mask the compute-heavy 10m high-resolution runs.
- **Big-Data Ready**: Automatically handles multi-gigabyte rasters using windowed, memory-efficient merging and 32-worker parallel downloads.

---

## � Installation

To install `autoSDM` directly from GitHub:

```r
# install.packages("devtools")
devtools::install_github("James-Longo/autoSDM")
```

**Python Setup:**
`autoSDM` automatically handles Python dependencies. On your first run, it will check your active Python environment (via `reticulate`) and install required packages (`earthengine-api`, `pandas`, `geopandas`, etc.) if they are missing.

*Ensure you have Python installed on your system.*

**Google Earth Engine:**
You must have a GEE account. Before running the package, ensure you have authenticated and initialized your Earth Engine session:

```r
library(rgee)
# One-time authentication
ee_Authenticate()

# Initialize session (required for every session)
ee_Initialize()
```

If you are using a specific Google Cloud project, provide it during initialization: `ee_Initialize(project = "your-project-id")`.

---

## �🚀 Quick Start (R)

```r
library(autoSDM)

# 1. Format your data
data <- format_data(raw_data, coords = c("lon", "lat"), year = "date", presence = "presence")

# 2. Run the full pipeline
# Extrapolates an ensemble model to a 10km radius around a central point
results <- autoSDM(data, aoi = list(lat=44.5, lon=-71.5, radius=10000))
```

---

## 🛠 Core Pipeline

The pipeline is driven by three primary CLI commands. All results, including GeoTIFF maps and metadata JSONs, are automatically organized within the `outputs/` directory.

### 1. Extract
Extracts Alpha Earth embeddings from Google Earth Engine for a set of occurrences.
- **Scale-Aware**: Native support for 10m, 100m, and 1000m scales.
- **Temporal Alignment**: Automatically matches observation years (2017–2024) to the correct embedding mosaic.

### 2. Analyze
Trains the selected model and determines ecological thresholds.
- **Presence-Only (Centroid)**: Calculates the Geometric Median and dot-product similarity.
- **Presence/Absence (Maxent)**: Trains a classifier and optimizes predictions for detection bias.
- **Advanced Tuning**: Automatically calculates **95% TPR**, **95% TNR**, **Balanced**, and **AUC** metrics.

### 3. Extrapolate & Ensemble
Projects models onto high-resolution maps.
- **Memory-Efficient**: Uses windowed-writing to merge thousands of GEE tiles into a single seamless GeoTIFF without crashing your local RAM.
- **Ensemble Mode**: Combine model outputs (e.g., `maxent * centroid`) to identify areas of strict agreement.
- **Parallel downloads**: 32 threads ensure maximum throughput from the GEE servers.

---

## 📁 Output Structure

Results are organized into project-specific and model-specific subdirectories within the `outputs/` folder for clarity:

```text
outputs/[project_name]/
├── centroid/
│   ├── centroid_10m.tif          (Continuous Similarity Map)
│   ├── centroid_results.csv.json (Model metadata & Centroid vector)
│   └── centroid.log              (Execution diagnostics)
├── maxent/
│   ├── maxent_10m.tif            (Maxent Probability Map)
│   └── maxent.log
└── ensemble/
    └── ensemble_10m.tif          (Agreement Map: Centroid * Maxent)
```

---

## ⚖️ License & Credits
Developed for Advanced Species Distribution Modeling. Leverages the Alpha Earth Embedding dataset provided by Google.
