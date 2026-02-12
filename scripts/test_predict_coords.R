# Source local R files
r_files <- list.files("/home/james-longo/Projects/autoSDM/R", full.names = TRUE)
for (f in r_files) source(f)

library(jsonlite)
library(terra)
library(sf)
library(reticulate)

# Force the correct virtualenv
env_name <- "r-autoSDM"
if (virtualenv_exists(env_name)) {
    use_virtualenv(env_name, required = TRUE)
}

# 0. Initialize GEE
message("Initializing GEE...")
ee <- import("ee")
tryCatch(
    {
        ee$Initialize()
        message("GEE Initialized successfully.")
    },
    error = function(e) {
        message("GEE Initialization failed: ", e$message)
        stop("Abort: GEE not initialized.")
    }
)

# 1. Dummy training data
train_data <- data.frame(
    longitude = c(-71.1, -71.2, -71.3, -71.0, -71.15),
    latitude = c(42.3, 42.35, 42.4, 42.45, 42.32),
    year = c(2022, 2022, 2023, 2023, 2024),
    present = c(1, 1, 1, 1, 1)
)

# 2. Prediction coordinates
test_coords <- data.frame(
    longitude = c(-71.12, -71.25),
    latitude = c(42.33, 42.38),
    year = c(2025, 2025)
)

# 3. AOI for the map
aoi <- list(lat = 42.35, lon = -71.15, radius = 5000)

# 4. Run autoSDM with predict_coords
message("Running autoSDM with coordinate-based prediction test...")
results <- autoSDM(
    data = train_data,
    aoi = aoi,
    output_dir = "test_output",
    scale = 1000,
    predict_coords = test_coords
)

# 5. Check results
if (!is.null(results$point_predictions)) {
    message("SUCCESS: Point predictions found!")
    print(results$point_predictions)
} else {
    stop("FAILURE: Point predictions not found in results.")
}

message("Test completed successfully!")
