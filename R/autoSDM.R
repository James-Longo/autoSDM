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
#' @param python_path Optional. Path to Python executable. Auto-detected if not provided.
#' @return A list containing model metadata, performance metrics, and paths to the generated maps.
#' @export
autoSDM <- function(data, aoi, output_dir = getwd(), scale = 10, python_path = NULL) {
  # 1. Validate standardized column names
  required_cols <- c("longitude", "latitude", "year")
  missing <- setdiff(required_cols, names(data))
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing required columns: %s\nPlease use format_data() to standardize your data.",
      paste(missing, collapse = ", ")
    ))
  }

  # 2. Auto-detect nuisance variables (any column that's not a standard column)
  standard_cols <- c("longitude", "latitude", "year", "present")
  all_cols <- names(data)
  nuisance_vars <- setdiff(all_cols, standard_cols)

  if (length(nuisance_vars) > 0) {
    message(sprintf("Detected nuisance variables: %s", paste(nuisance_vars, collapse = ", ")))
  }

  # 3. Python Configuration
  python_path <- resolve_python_path(python_path)

  if (is.null(python_path) || !file.exists(python_path)) {
    stop("Could not find a valid Python environment.\nPlease ensure Python is installed and detected by `reticulate::py_config()`, or provide the `python_path` argument explicitly.")
  }

  # 4. Auto-configure Dependencies & PYTHONPATH
  message("Checking Python dependencies...")
  ensure_autoSDM_dependencies(python_path)

  # Locate the python source directory (inst/python)
  pkg_py_path <- system.file("python", package = "autoSDM")

  # Fallback for local development (if package not installed but loaded)
  if (pkg_py_path == "") {
    if (file.exists(file.path(getwd(), "inst", "python"))) {
      pkg_py_path <- file.path(getwd(), "inst", "python")
    }
  }

  if (pkg_py_path != "") {
    # Prepend to PYTHONPATH so 'autoSDM' module is found
    old_pythonpath <- Sys.getenv("PYTHONPATH")
    new_pythonpath <- if (old_pythonpath == "") pkg_py_path else paste(pkg_py_path, old_pythonpath, sep = .Platform$path.sep)

    Sys.setenv(PYTHONPATH = new_pythonpath)
    on.exit(if (old_pythonpath == "") Sys.unsetenv("PYTHONPATH") else Sys.setenv(PYTHONPATH = old_pythonpath), add = TRUE)

    message(sprintf("Added to PYTHONPATH: %s", pkg_py_path))
  } else {
    warning("Could not locate 'inst/python' source. Python 'autoSDM' module might not be found.")
  }

  # 5. Check GEE Readiness
  message(sprintf("Using Python: %s", python_path))

  message("Checking Google Earth Engine authentication...")
  gee_check_script <- "import ee; ee.Initialize(); print('OK')"
  auth_check <- tryCatch(
    {
      system2(python_path, args = c("-c", shQuote(gee_check_script)), stdout = TRUE, stderr = TRUE)
    },
    error = function(e) ""
  )

  if (length(auth_check) == 0 || !any(auth_check == "OK")) {
    stop("Google Earth Engine is not initialized or authenticated in the Python environment.\nPlease run 'earthengine authenticate' in your terminal or ensure your credentials are set.")
  }

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # 6. Initialize Paths
  extract_csv <- file.path(output_dir, "autoSDM_extracted_embeddings.csv")
  centroid_meta <- file.path(output_dir, "model_centroid.csv")
  maxent_meta <- file.path(output_dir, "model_maxent.csv")
  ensemble_results_json <- file.path(output_dir, "ensemble_results_10m.json")

  # 7. Step 1: Extract Embeddings
  message("--- Step 1: Extracting Alpha Earth Embeddings ---")
  data_with_emb <- extract_embeddings(data, scale = scale, python_path = python_path)
  vroom::vroom_write(data_with_emb, extract_csv, delim = ",")

  # 8. Step 2 & 3: Multi-Model Analysis
  message("--- Step 2: Running Centroid Analysis ---")
  res_centroid <- analyze_embeddings(data_with_emb, method = "centroid", python_path = python_path)

  message("--- Step 3: Running Maxent Analysis ---")
  res_maxent <- analyze_embeddings(data_with_emb, method = "maxent", nuisance_vars = nuisance_vars, python_path = python_path)

  # 9. Step 4: Ensemble Extrapolation
  message("--- Step 4: Generating Ensemble Extrapolation Map ---")

  args <- c(
    "-m", "autoSDM.cli", "ensemble",
    "--input", shQuote(extract_csv),
    "--output", shQuote(ensemble_results_json),
    "--meta", shQuote(paste0(centroid_meta, ".json")),
    "--meta2", shQuote(paste0(maxent_meta, ".json")),
    "--scale", scale,
    "--prefix", "autoSDM_ensemble"
  )

  # Handle AOI
  if (is.list(aoi) && !is.null(aoi$lat)) {
    # Simple circle: lat, lon, radius
    args <- c(args, "--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
  } else if (is.character(aoi)) {
    # Path to polygon file
    args <- c(args, "--aoi-path", shQuote(aoi))
  } else if (inherits(aoi, c("sf", "sfc"))) {
    # sf geometry object - write to temp GeoJSON
    aoi_path <- file.path(output_dir, "aoi_temp.geojson")
    sf::st_write(sf::st_transform(aoi, 4326), aoi_path, driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
    args <- c(args, "--aoi-path", shQuote(aoi_path))
  } else {
    stop("AOI must be a list(lat, lon, radius), a path to a polygon file, or an sf geometry object.")
  }

  # Run the models to save the meta files permanently in output_dir
  nuisance_arg <- if (length(nuisance_vars) > 0) c("--nuisance-vars", paste(nuisance_vars, collapse = ",")) else NULL

  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(centroid_meta), "--method", "centroid"))
  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(maxent_meta), "--method", "maxent", nuisance_arg))

  # Run ensemble
  status <- system2(python_path, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Ensemble extrapolation failed.")
  }

  # Load results
  final_results <- jsonlite::fromJSON(ensemble_results_json)

  message("autoSDM pipeline complete!")
  return(final_results)
}
