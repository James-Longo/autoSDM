library(vroom)
library(dplyr)

# Set paths
ebd_path <- "/home/james-longo/Projects/autoSDM/benchmarks/ebd_bicthr_unv_smp_relDec-2025/ebd_bicthr_unv_smp_relDec-2025.txt"
output_path <- "/home/james-longo/Projects/autoSDM/benchmarks/ebd_bicthr_winter_processed.csv"

# Read data
message("Reading EBD data...")
ebd <- vroom(ebd_path, delim = "\t", quote = "", na = c("", "NA"))

# Filter data
message("Filtering data...")
processed_data <- ebd %>%
  filter(
    # Outside US and Canada
    !(`COUNTRY CODE` %in% c("US", "CA")),
    # Best practice protocols
    `PROTOCOL NAME` %in% c("Stationary", "Traveling"),
    # Complete checklists
    `ALL SPECIES REPORTED` == 1,
    # Distance limit for traveling checklists (< 5km)
    is.na(`EFFORT DISTANCE KM`) | `EFFORT DISTANCE KM` < 5
  ) %>%
  # Select and rename columns for autoSDM
  select(
    longitude = LATITUDE,  # Wait, I saw LATITUDE and LONGITUDE in the head output. 
    latitude = LONGITUDE,   # The head output showed: LATITUDE=32.2616590, LONGITUDE=-64.8766750
    # Actually, the head output order was LATITUDE then LONGITUDE.
    # Let me double check the column names from the head output in Step 32.
    # 23: LATITUDE, 24: LONGITUDE
    date = `OBSERVATION DATE`
  ) %>%
  # Add presence column (standard for autoSDM)
  mutate(present = 1)

# Note: I swapped lat/lon in select because autoSDM::format_data expects longitude first?
# Let's check format_data again.
# coords: A character vector of length 2 specifying the longitude and latitude columns IN ORDER: c(longitude_col, latitude_col).
# So I should keep them as lat and lon for now and let format_data handle it, or rename correctly.

# Let's re-read the head output carefully.
# LATITUDE: 32.2616590 (Positive, so North, typical for Bermuda)
# LONGITUDE: -64.8766750 (Negative, so West, typical for Bermuda)
# Okay, so:
# latitude = LATITUDE
# longitude = LONGITUDE

processed_data <- ebd %>%
  filter(
    !(`COUNTRY CODE` %in% c("US", "CA")),
    `PROTOCOL NAME` %in% c("Stationary", "Traveling"),
    `ALL SPECIES REPORTED` == 1,
    is.na(`EFFORT DISTANCE KM`) | `EFFORT DISTANCE KM` < 5
  ) %>%
  select(
    latitude = LATITUDE,
    longitude = LONGITUDE,
    date = `OBSERVATION DATE`
  ) %>%
  mutate(present = 1)

# Save processed data
message(sprintf("Saving %d rows to %s", nrow(processed_data), output_path))
write.csv(processed_data, output_path, row.names = FALSE)
