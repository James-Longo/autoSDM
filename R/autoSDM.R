#' autoSDM: Automated Species Distribution Modeling
#'
#' This is the main entry point for the autoSDM pipeline. It performs embedding extraction,
#' multi-model analysis (Centroid + Ridge), and generates high-resolution distribution maps.
#'
#' @param data A data frame formatted via `format_data()`. Must have standardized lowercase columns
#'   (longitude, latitude, year, present). Any additional columns are ignored.
#' @param aoi Optional. Either a list with `lat`, `lon`, and `radius` (in meters), or a character string path to a polygon file (GeoJSON or Shapefile). Required for map generation. If NULL and `predict_coords` is provided, only point predictions are returned (no map).
#' @param output_dir Optional. Directory to save results. Defaults to the current working directory.
#' @param scale Optional. Resolution in meters for the final map. Defaults to 10.
#' @param python_path Optional. Path to Python executable. Auto-detected if not provided.
#' @param gee_project Optional. Google Cloud Project ID for Earth Engine. Required for newer API versions.
#' @param cv Optional. Boolean whether to run 5-fold Spatial Block Cross-Validation. Defaults to FALSE.
#' @param predict_coords Optional. Data frame of coordinates to predict at.
#' @param year Optional. Alpha Earth Mosaic year for mapping. Defaults to 2025.
#' @return A list containing training data, models, and prediction results.
#' @export
autoSDM <- function(data, aoi = NULL, output_dir = getwd(), scale = 10, python_path = NULL, gee_project = NULL, cv = FALSE, predict_coords = NULL, methods = NULL, year = 2025) {
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



  # 2b. Check if data is Presence-Only (no absences)
  has_absences <- "present" %in% names(data) && any(data$present == 0)
  
  if (is.null(methods)) {
    methods <- if (has_absences) "ridge" else "centroid"
  }

  # Support "ensemble" as a method keyword
  requested_ensemble <- "ensemble" %in% methods
  if (requested_ensemble) {
    # Remove "ensemble" from individual methods and ensure at least centroid/ridge
    methods <- setdiff(methods, "ensemble")
    methods <- unique(c(methods, "centroid", "ridge"))
  }
  
  if (length(methods) > 1) {
    requested_ensemble <- TRUE
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
    gee_project = gee_project
  )

  # 4. Decide on Multi-Species
  is_multi_species <- "species" %in% names(embedded_data) && length(unique(embedded_data$species)) > 1
  
  # 5. Prepare AOI arguments
  aoi_arg <- NULL
  if (is.list(aoi) && !is.null(aoi$lat)) {
    aoi_arg <- c("--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
  } else if (is.character(aoi)) {
    aoi_arg <- c("--aoi-path", shQuote(aoi))
  }

  # 6. Extraction/Preparation
  # (Standardizing CSV output for Python)
  extract_csv <- tempfile(fileext = ".csv")
  # embedded_data is already extracted from step 5
  if (requireNamespace("vroom", quietly = TRUE)) {
    vroom::vroom_write(embedded_data, extract_csv, delim = ",")
  } else {
    write.csv(embedded_data, extract_csv, row.names = FALSE)
  }

  if (!is_multi_species) {
    message("--- Single-Species Mode ---")
    cv_arg <- if (cv) "--cv" else NULL
    proj_arg <- if (!is.null(gee_project)) c("--project", shQuote(gee_project)) else NULL

    meta_files <- list()

    # Loop through requested methods
    for (m in methods) {
      m_clean <- gsub("[^a-zA-Z0-9]", "", m)
      message(sprintf("--- Step: Running %s Analysis ---", m))
      
      out_csv <- file.path(output_dir, paste0(m_clean, ".csv"))
      out_json <- file.path(output_dir, paste0(m_clean, ".json"))
      
      py_method <- m
      
      system2(python_path, args = c(
        "-m", "autoSDM.cli", "analyze", 
        "--input", shQuote(extract_csv), 
        "--output", shQuote(out_csv), 
        "--method", py_method, 
        "--scale", scale, 
        "--year", year, 
        cv_arg, proj_arg, aoi_arg
      ))
      
      if (file.exists(out_json)) {
        meta_files[[m]] <- out_json
        
        # 7. Generate Individual Map for this method
        if (!is.null(aoi)) {
          message(sprintf("--- Step: Generating Individual Map (%s) ---", m))
          system2(python_path, args = c(
            "-m", "autoSDM.cli", "ensemble", 
            "--input", shQuote(extract_csv), 
            "--output", shQuote(file.path(output_dir, paste0(m_clean, "_results.json"))), 
            "--meta", shQuote(out_json), 
            "--scale", scale, 
            "--prefix", m_clean, 
            "--year", year, 
            proj_arg, aoi_arg
          ))
        }
      }
    }

    # 8. Predict at specific coordinates (if provided)
    if (!is.null(predict_coords)) {
      message("--- Step: Predicting at specific coordinates ---")
      pred_embedded <- extract_embeddings(predict_coords, scale = scale, python_path = python_path, gee_project = gee_project)
      point_preds <- pred_embedded

      # Combined product similarity
      point_preds$similarity <- 1.0

      for (m in methods) {
        meta_p <- file.path(output_dir, paste0(m, ".json"))
        if (file.exists(meta_p)) {
          preds <- predict_at_coords(pred_embedded, analysis_meta_path = meta_p, scale = scale, python_path = python_path, gee_project = gee_project)
          point_preds[[m]] <- preds$similarity
          point_preds$similarity <- point_preds$similarity * preds$similarity
        }
      }
    }

    # 9. Generate Ensemble Map (if requested or multiple methods)
    if (!is.null(aoi) && requested_ensemble) {
      message("--- Step: Generating Ensemble Map (consensus of all methods) ---")
      
      ensemble_results_json <- file.path(output_dir, "ensemble_results.json")

      # Build args with all meta files
      ensemble_args <- c("-m", "autoSDM.cli", "ensemble", "--input", shQuote(extract_csv), "--output", shQuote(ensemble_results_json))
      for (mf in meta_files) {
        if (file.exists(mf)) ensemble_args <- c(ensemble_args, "--meta", shQuote(mf))
      }
      ensemble_args <- c(ensemble_args, "--scale", scale, "--prefix", "ensemble", "--year", year)
      if (!is.null(gee_project)) ensemble_args <- c(ensemble_args, "--project", shQuote(gee_project))

      if (is.list(aoi) && !is.null(aoi$lat)) {
        ensemble_args <- c(ensemble_args, "--lat", aoi$lat, "--lon", aoi$lon, "--radius", aoi$radius)
      } else if (is.character(aoi)) {
        ensemble_args <- c(ensemble_args, "--aoi-path", shQuote(aoi))
      }

      status <- system2(python_path, args = ensemble_args)
      if (status != 0) stop("Ensemble extrapolation failed.")
      final_results <- jsonlite::fromJSON(ensemble_results_json)
    } else {
      # For single results, we return the last json generated or empty list
      # The individual maps are already handled in the loop.
      final_results <- list()
      if (length(meta_files) > 0) {
          final_results <- jsonlite::fromJSON(meta_files[[1]])
      }
    }

    if (!is.null(predict_coords)) final_results$point_predictions <- point_preds
    message("autoSDM pipeline complete!")
    return(final_results)
  } else {
    # MULTI-SPECIES WORKFLOW
    # -------------------------------------------------------------------------
    message(sprintf("--- Multi-Species Mode: Processing %d species ---", length(unique(embedded_data$species))))
    message("Orchestrating GEE server-side parallelization...")


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
      "--year", year,
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
      res_path <- file.path(output_dir, sp, "results.json")
      if (file.exists(res_path)) {
        results_list[[sp]] <- jsonlite::fromJSON(res_path)
      }
    }

    class(results_list) <- "autoSDM_multi"
    return(results_list)
  }
}
