#' autoSDM: Automated Species Distribution Modeling
#'
#' This is the main entry point for the autoSDM pipeline. It performs embedding extraction,
#' multi-model analysis (Centroid + Maxent), and generates an ensemble extrapolation map.
#'
#' @param data A data frame formatted via `format_data()`. Must have standardized lowercase columns
#'   (longitude, latitude, year, present). Any additional columns are treated as nuisance variables.
#' @param aoi Optional. Either a list with `lat`, `lon`, and `radius` (in meters), or a character string path to a polygon file (GeoJSON or Shapefile). Required for map generation. If NULL and `predict_coords` is provided, only point predictions are returned (no map).
#' @param output_dir Optional. Directory to save results. Defaults to the current working directory.
#' @param scale Optional. Resolution in meters for the final map. Defaults to 10.
#' @param background_method Optional. Method to generate background points if presence-only data is provided. Defaults to "sample_extent". Options: "sample_extent", "buffer".
#' @param background_buffer Optional. Numeric vector of length 2: c(min_dist, max_dist) in meters for buffer-based sampling.
#' @param python_path Optional. Path to Python executable. Auto-detected if not provided.
#' @param gee_project Optional. Google Cloud Project ID for Earth Engine. Required for newer API versions.
#' @param cv Optional. Boolean whether to run 5-fold Spatial Block Cross-Validation. Defaults to FALSE.
#' @param predict_coords Optional. Data frame of coordinates to predict at.
#' @return A list containing model metadata, performance metrics, and (if aoi is provided) paths to the generated maps.

#' @export
autoSDM <- function(data, aoi = NULL, output_dir = getwd(), scale = 10, background_method = "sample_extent", background_buffer = NULL, python_path = NULL, gee_project = NULL, cv = FALSE, predict_coords = NULL) {
  # Validate: need at least aoi or predict_coords
  if (is.null(aoi) && is.null(predict_coords)) {
    stop("You must provide either 'aoi' (for map generation) or 'predict_coords' (for point predictions), or both.")
  }
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
  emb_cols <- sprintf("A%02d", 0:63)
  all_cols <- names(data)
  nuisance_vars <- setdiff(all_cols, c(standard_cols, emb_cols, "species")) # Exclude species from nuisance
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

  # 5. Extract Embeddings (Satellite Data)
  # Step 1: Extracting Alpha Earth Embeddings (Once for all unique coordinates)
  message("--- Step 1: Extracting Alpha Earth Embeddings (Coordinate-Centric) ---")
  embedded_data <- extract_embeddings(
    data,
    scale = scale,
    python_path = python_path,
    gee_project = gee_project,
    background_method = background_method,
    background_buffer = background_buffer
  )

  # 6. Determine if multi-species or single species
  is_multi_species <- "species" %in% names(embedded_data) && length(unique(embedded_data$species)) > 1

  # Path for shared training data
  # NOTE: Embeddings (A00-A63) are kept in the CSV because the centroid analyzer
  # reads them locally. Maxent samples them server-side and ignores these columns.
  message("--- Preparing training data ---")

  extract_csv <- file.path(output_dir, "autoSDM_training_data.csv")
  if (requireNamespace("vroom", quietly = TRUE)) {
    vroom::vroom_write(embedded_data, extract_csv, delim = ",")
  } else {
    write.csv(embedded_data, extract_csv, row.names = FALSE)
  }


  if (!is_multi_species) {
    # SINGLE SPECIES WORKFLOW (Legacy/Default)
    # -------------------------------------------------------------------------
    message("--- Single-Species Mode ---")

    # 7. Step 2 & 3: Analysis (Centroid + Maxent) via CLI
    nuisance_arg <- if (length(nuisance_vars) > 0) c("--nuisance-vars", paste(nuisance_vars, collapse = ",")) else NULL
    cv_arg <- if (cv) "--cv" else NULL
    proj_arg <- if (!is.null(gee_project)) c("--project", shQuote(gee_project)) else NULL

    message("--- Step 2: Running Centroid Analysis ---")
    system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(file.path(output_dir, "centroid.csv")), "--method", "centroid", "--scale", scale, cv_arg, proj_arg))

    message("--- Step 3: Running Maxent Analysis ---")
    system2(python_path, args = c("-m", "autoSDM.cli", "analyze", "--input", shQuote(extract_csv), "--output", shQuote(file.path(output_dir, "maxent.csv")), "--method", "maxent", "--scale", scale, nuisance_arg, cv_arg, proj_arg))

    # 8. Predict at specific coordinates (if provided)
    if (!is.null(predict_coords)) {
      message("--- Step 4: Predicting at specific coordinates ---")

      # Extract embeddings for prediction coords ONCE (shared by both models)
      message("  Extracting embeddings for prediction coordinates...")
      pred_embedded <- extract_embeddings(predict_coords, scale = scale, python_path = python_path, gee_project = gee_project)

      # Predict with each model using pre-embedded data (no re-extraction)
      preds_c <- predict_at_coords(pred_embedded, analysis_meta_path = file.path(output_dir, "centroid.json"), scale = scale, python_path = python_path, gee_project = gee_project)
      preds_m <- predict_at_coords(pred_embedded, analysis_meta_path = file.path(output_dir, "maxent.json"), scale = scale, python_path = python_path, gee_project = gee_project)
      point_preds <- preds_c
      point_preds$centroid <- preds_c$similarity
      point_preds$maxent <- preds_m$similarity
      point_preds$similarity <- preds_c$similarity * preds_m$similarity
    }

    # 9. Generate Ensemble Extrapolation Map (only if AOI provided)
    if (!is.null(aoi)) {
      message("--- Step 5: Generating Ensemble Extrapolation Map ---")
      ensemble_results_json <- file.path(output_dir, "ensemble_results.json")
      ensemble_args <- c("-m", "autoSDM.cli", "ensemble", "--input", shQuote(extract_csv), "--output", shQuote(ensemble_results_json), "--meta", shQuote(file.path(output_dir, "centroid.json")), "--meta2", shQuote(file.path(output_dir, "maxent.json")), "--scale", scale, "--prefix", "ensemble")
      if (!is.null(gee_project)) ensemble_args <- c(ensemble_args, "--project", shQuote(gee_project))

      # Handle AOI
      if (is.list(aoi) && !is.null(aoi$lat)) {
        ensemble_args <- c(ensemble_args, "--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
      } else if (is.character(aoi)) {
        ensemble_args <- c(ensemble_args, "--aoi-path", shQuote(aoi))
      }

      status <- system2(python_path, args = ensemble_args)
      if (status != 0) stop("Ensemble extrapolation failed.")

      final_results <- jsonlite::fromJSON(ensemble_results_json)
    } else {
      # No map — return analysis metadata only
      final_results <- list(
        centroid_meta = jsonlite::fromJSON(file.path(output_dir, "centroid.json")),
        maxent_meta   = jsonlite::fromJSON(file.path(output_dir, "maxent.json"))
      )
    }

    if (!is.null(predict_coords)) {
      final_results$point_predictions <- point_preds
    }

    message("autoSDM pipeline complete!")
    return(final_results)
  } else {
    # MULTI-SPECIES WORKFLOW
    # -------------------------------------------------------------------------
    message(sprintf("--- Multi-Species Mode: Processing %d species ---", length(unique(embedded_data$species))))
    message("Orchestrating GEE server-side parallelization...")

    nuisance_arg <- if (length(nuisance_vars) > 0) c("--nuisance-vars", paste(nuisance_vars, collapse = ",")) else NULL
    cv_arg <- if (cv) "--cv" else NULL
    proj_arg <- if (!is.null(gee_project)) c("--project", shQuote(gee_project)) else NULL

    aoi_arg <- NULL
    if (is.list(aoi) && !is.null(aoi$lat)) {
      aoi_arg <- c("--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
    } else if (is.character(aoi)) {
      aoi_arg <- c("--aoi-path", shQuote(aoi))
    }

    # Internal multi-species ensemble pipeline in Python
    # This invokes a single Python process that loops over species and sends requests to GEE.
    multi_args <- c(
      "-m", "autoSDM.cli", "analyze",
      "--input", shQuote(extract_csv),
      "--output", shQuote(output_dir),
      "--method", "ensemble",
      "--species-col", "species",
      "--scale", scale,
      nuisance_arg,
      cv_arg,
      proj_arg,
      aoi_arg
    )

    status <- system2(python_path, args = multi_args)
    if (status != 0) stop("Multi-species analysis failed.")

    # Load and combine results from species-specific directories
    species_list <- unique(embedded_data$species)
    results_list <- list()
    for (sp in species_list) {
      res_path <- file.path(output_dir, sp, "ensemble_results.json")
      if (file.exists(res_path)) {
        results_list[[sp]] <- jsonlite::fromJSON(res_path)
      }
    }

    class(results_list) <- "autoSDM_multi"
    return(results_list)
  }
}
