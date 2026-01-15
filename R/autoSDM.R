#' autoSDM: Automated Species Distribution Modeling
#'
#' This is the main entry point for the autoSDM pipeline. It performs embedding extraction,
#' multi-model analysis (Centroid + Maxent), and generates an ensemble extrapolation map.
#'
#' @param data A data frame formatted via `format_data()`.
#' @param aoi Mandatory. Either a list with `lat`, `lon`, and `radius` (in meters), or a character string path to a polygon file (GeoJSON or Shapefile).
#' @param output_dir Optional. Directory to save results. Defaults to the current working directory.
#' @param nuisance_vars Optional. Character vector of columns to treat as nuisance variables.
#' @param scale Optional. Resolution in meters for the final map. Defaults to 10.
#' @return A list containing model metadata, performance metrics, and paths to the generated maps.
#' @export
autoSDM <- function(data, aoi, output_dir = getwd(), nuisance_vars = NULL, scale = 10) {
  # 2. Python Configuration
  # If py_venv_path is set globally, use it (for backward compatibility or testing)
  if (exists("py_venv_path")) {
    python_path <- file.path(py_venv_path, "Scripts", "python.exe")
    if (!file.exists(python_path)) python_path <- file.path(py_venv_path, "bin", "python")
  } else {
    # Otherwise, use reticulate to find the active python
    tryCatch(
      {
        python_path <- reticulate::py_config()$python
      },
      error = function(e) {
        stop("Could not find a Python environment. Please run install_autoSDM() or configure reticulate.")
      }
    )
  }

  if (!file.exists(python_path)) {
    stop(paste("Python executable not found at:", python_path))
  }

  # 1. Check GEE Readiness logic remains... (but using the found python_path)
  message(sprintf("Using Python: %s", python_path))

  message("Checking Google Earth Engine authentication...")
  auth_check <- system2(python_path, args = c("-c", "import ee; try: ee.Initialize(); print('OK')\nexcept: exit(1)"), stdout = TRUE, stderr = NULL)

  if (length(auth_check) == 0 || auth_check != "OK") {
    stop("Google Earth Engine is not initialized or authenticated in the Python environment.\nPlease run 'earthengine authenticate' in your terminal or ensure your credentials are set.")
  }

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # 2. Initialize Paths
  extract_csv <- file.path(output_dir, "autoSDM_extracted_embeddings.csv")
  centroid_meta <- file.path(output_dir, "model_centroid.csv")
  maxent_meta <- file.path(output_dir, "model_maxent.csv")
  ensemble_results_json <- file.path(output_dir, "ensemble_results_10m.json")

  # 3. Step 1: Extract Embeddings
  message("--- Step 1: Extracting Alpha Earth Embeddings ---")
  data_with_emb <- extract_embeddings(data, scale = scale, python_path = python_path)
  vroom::vroom_write(data_with_emb, extract_csv, delim = ",")

  # 4. Step 2 & 3: Multi-Model Analysis
  message("--- Step 2: Running Centroid Analysis ---")
  # We call the internal analyze function
  res_centroid <- analyze_embeddings(data_with_emb, method = "centroid", python_path = python_path)

  message("--- Step 3: Running Maxent Analysis ---")
  res_maxent <- analyze_embeddings(data_with_emb, method = "maxent", nuisance_vars = nuisance_vars, python_path = python_path)

  # We need the .json metadata paths produced by the internal analyze function
  # Since' analyze_embeddings' uses temp files and cleans them up, we should
  # probably modify it or just call the CLI directly here to get permanent files.
  # Let's call the CLI directly for better control in this pipeline.

  # 5. Step 4: Ensemble Extrapolation
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
    args <- c(args, "--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
  } else if (is.character(aoi)) {
    args <- c(args, "--aoi-path", shQuote(aoi))
  } else {
    stop("AOI must be a list(lat, lon, radius) or a path to a polygon file.")
  }

  # Actually run the models to save the meta files permanently in output_dir
  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(centroid_meta), "--method", "centroid"))
  system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(maxent_meta), "--method", "maxent", if (!is.null(nuisance_vars)) c("--nuisance-vars", paste(nuisance_vars, collapse = ","))))

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
