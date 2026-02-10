devtools::install_github("James-Longo/autoSDM", dependencies = TRUE)

library(autoSDM)

# Set paths
input_path <- "/home/james-longo/Projects/autoSDM/benchmarks/ebd_bicthr_winter_processed.csv"
output_dir <- "/home/james-longo/Projects/autoSDM/outputs/bith_winter"

# Load data
message("Loading processed BITH data...")
data <- read.csv(input_path)

# 1. Format data for autoSDM
# autoSDM::format_data(data, coords = c("lon", "lat"), year = "date", presence = "presence")
# My processed data has: latitude, longitude, date, present
formatted_data <- format_data(
  data,
  coords = c("longitude", "latitude"),
  year = "date",
  presence = "present"
)

# 2. Define AOI covering the wintering range
# Hispaniola (main wintering area) central point
# Roughly lat 18.2, lon -70.5
# Radius: 1000km should cover Caribbean islands
aoi <- list(lat = 18.5, lon = -71.2, radius = 1000000)

# 3. Run autoSDM pipeline
# We'll run at 100m first to be sure it works, then 10m if requested?
# User wants a range map, 100m is usually sufficient for a general range map.
# But README says 10m-resolution mapping capabilities.
# Let's try scale=100 first for speed and verification.
message("Starting autoSDM pipeline...")
results <- autoSDM(
  formatted_data,
  aoi = aoi,
  output_dir = output_dir,
  scale = 100
)

message("autoSDM run complete!")
print(results)
