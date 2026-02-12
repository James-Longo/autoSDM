#' autoSDM: Automated Species Distribution Modeling
#'
#' This is the main entry point for the autoSDM pipeline. It performs embedding extraction,
#' multi-model analysis (Centroid + Maxent), and generates an ensemble extrapolation map.
#'
#' @param data A data frame formatted via `format_data()`. Must have standardized lowercase columns
#'   (longitude, latitude, year, present). Any additional columns are treated as nuisance variables.
#' @param aoi Mandatory. Either a list with `lat`, `lon`, and `radius` (in meters), or a character string path to a polygon file (GeoJSON or Shapefile).
#' @param output_dir Optional. Directory to save results. Defaults to the current working directory.
#' @param scale Optional. Resolution in meters for the final map. Defaults to 10.
#' @param background_method Optional. Method to generate background points if presence-only data is provided. Defaults to "sample_extent". Options: "sample_extent", "buffer".
#' @param background_buffer Optional. Numeric vector of length 2: c(min_dist, max_dist) in meters for buffer-based sampling.
#' @param python_path Optional. Path to Python executable. Auto-detected if not provided.
#' @param gee_project Optional. Google Cloud Project ID for Earth Engine. Required for newer API versions.
#' @param cv Optional. Boolean whether to run 5-fold Spatial Block Cross-Validation. Defaults to FALSE.
#' @param predict_coords Optional. Data frame of coordinates to predict at.
#' @return A list containing model metadata, performance metrics, and paths to the generated maps.

#' @export
autoSDM <- function(data, aoi, output_dir = getwd(), scale = 10, background_method = "sample_extent", background_buffer = NULL, python_path = NULL, gee_project = NULL, cv = FALSE, predict_coords = NULL) {
  # 1. Validate standardized column names
  required_cols <- c("longitude", "latitude", "year")
  missing <- setdiff(required_cols, names(data))
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing required columns: %s\nPlease use format_data() to standardize your data.",
      paste(missing, collapse = ", ")
    ))
  }

  # 2. Auto-detect nuisance variables
  standard_cols <- c("longitude", "latitude", "year", "present")
  all_cols <- names(data)
  nuisance_vars <- setdiff(all_cols, standard_cols)

  if (length(nuisance_vars) > 0) {
    message(sprintf("Detected nuisance variables: %s", paste(nuisance_vars, collapse = ", ")))
  }

  # 3. Python Configuration
  # Check for virtualenv and initialize dependencies
  python_path_detected <- ensure_autoSDM_dependencies()
  python_path <- if (!is.null(python_path)) python_path else python_path_detected
  python_path <- resolve_python_path(python_path)

  if (is.null(python_path) || !file.exists(python_path)) {
    stop("Could not find a valid Python environment.\nPlease ensure Python is installed and detected by `reticulate::py_config()`, or provide the `python_path` argument explicitly.")
  }

  # Locate the python source directory (inst/python)
  pkg_py_path <- ""
  if (file.exists(file.path(getwd(), "inst", "python"))) {
    pkg_py_path <- file.path(getwd(), "inst", "python")
  } else {
    pkg_py_path <- system.file("python", package = "autoSDM")
  }

  if (pkg_py_path != "") {
    Sys.setenv(PYTHONPATH = pkg_py_path)
    message(sprintf("Added to PYTHONPATH: %s", pkg_py_path))
  }

  # 4. Check GEE Readiness
  message(sprintf("Using Python: %s", python_path))
  ensure_gee_authenticated(project = gee_project)
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # 5. Paths for results
  extract_csv <- file.path(output_dir, "autoSDM_extracted_embeddings.csv")
  centroid_meta <- file.path(output_dir, "model_centroid.json")
  maxent_meta <- file.path(output_dir, "model_maxent.json")
  ensemble_results_json <- file.path(output_dir, "ensemble_results.json")

  # 6. Step # 6. Extract Embeddings (Satellite Data)
  message("--- Step 1: Extracting Alpha Earth Embeddings ---")
  embedded_data <- extract_embeddings(
    data,
    scale = scale,
    python_path = python_path,
    gee_project = gee_project,
    background_method = background_method,
    background_buffer = background_buffer
  )
  write.csv(embedded_data, extract_csv, row.names = FALSE)

  # 7. Step 2: Analysis (Centroid + Maxent)
  message("--- Step 2: Running Centroid Analysis ---")
  res_centroid <- analyze_embeddings(
    embedded_data,
    method = "centroid",
    python_path = python_path,
    gee_project = gee_project,
    cv = cv
  )
  # Save RDS for R users
  saveRDS(res_centroid, file.path(output_dir, "analysis_centroid.rds"))

  message("--- Step 3: Running Maxent Analysis ---")
  res_maxent <- analyze_embeddings(
    embedded_data,
    method = "maxent",
    nuisance_vars = nuisance_vars,
    python_path = python_path,
    gee_project = gee_project,
    cv = cv
  )
  saveRDS(res_maxent, file.path(output_dir, "analysis_maxent.rds"))

  # 8. Step 4: Ensemble Extrapolation
  message("--- Step 4: Generating Ensemble Extrapolation Map ---")

  args <- c(
    "-m", "autoSDM.cli", "ensemble",
    "--input", shQuote(extract_csv),
    "--output", shQuote(ensemble_results_json),
    "--meta", shQuote(file.path(output_dir, "autoSDM_extracted_embeddings.json")), # CLI appends .json to output path
    "--meta2", shQuote(file.path(output_dir, "autoSDM_extracted_embeddings.json")), # This depends on how analyze_embeddings saved them
    "--scale", scale,
    "--prefix", "ensemble"
  )

  # Wait, my analyze_embeddings call above uses CLI internally which saves to tmp files.
  # Let's just use the metadata paths directly.
  # Actually, analyze_embeddings.R needs to save the JSON to the output_dir if we want to use them for ensemble.
  # Let's refine analyze_embeddings.R to take an optional output path for the JSON.
  # For now, I'll just run the CLI again for the ensemble inputs to be sure.

  nuisance_arg <- if (length(nuisance_vars) > 0) c("--nuisance-vars", paste(nuisance_vars, collapse = ",")) else NULL
  cv_arg <- if (cv) "--cv" else NULL
  proj_arg <- if (!is.null(gee_project)) c("--project", shQuote(gee_project)) else NULL

  # Re-run analysis via CLI to ensure files are in output_dir
  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(file.path(output_dir, "centroid.csv")), "--method", "centroid", cv_arg, proj_arg))
  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(file.path(output_dir, "maxent.csv")), "--method", "maxent", nuisance_arg, cv_arg, proj_arg))

  ensemble_args <- c(
    "-m", "autoSDM.cli", "ensemble",
    "--input", shQuote(extract_csv),
    "--output", shQuote(ensemble_results_json),
    "--meta", shQuote(file.path(output_dir, "centroid.json")),
    "--meta2", shQuote(file.path(output_dir, "maxent.json")),
    "--scale", scale,
    "--prefix", "autoSDM_ensemble"
  )
  if (!is.null(gee_project)) ensemble_args <- c(ensemble_args, "--project", shQuote(gee_project))

  # Handle AOI
  if (is.list(aoi) && !is.null(aoi$lat)) {
    ensemble_args <- c(ensemble_args, "--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
  } else if (is.character(aoi)) {
    ensemble_args <- c(ensemble_args, "--aoi-path", shQuote(aoi))
  } else if (inherits(aoi, c("sf", "sfc", "SpatVector"))) {
    if (inherits(aoi, "SpatVector")) aoi <- sf::st_as_sf(aoi)
    aoi_path <- file.path(output_dir, "aoi_temp.geojson")
    sf::st_write(sf::st_transform(aoi, 4326), aoi_path, driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
    ensemble_args <- c(ensemble_args, "--aoi-path", shQuote(aoi_path))
  }

  status <- system2(python_path, args = ensemble_args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Ensemble extrapolation failed.")
  }

  # Load results
  final_results <- jsonlite::fromJSON(ensemble_results_json)

  # 9. Optional: Predict at specific coordinates
  if (!is.null(predict_coords)) {
    message("--- Step 5: Predicting at specific coordinates ---")
    # We use the centroid model for point predictions by default in this ensemble context,
    # or we could predict for both. Let's start with centroid.
    point_preds <- predict_at_coords(
      predict_coords,
      analysis_meta_path = file.path(output_dir, "centroid.json"),
      scale = scale,
      python_path = python_path,
      gee_project = gee_project
    )
    final_results$point_predictions <- point_preds
  }

  message("autoSDM pipeline complete!")
  return(final_results)
}
