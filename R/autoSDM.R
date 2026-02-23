#' autoSDM: Automated Species Distribution Modeling
#'
#' This is the main entry point for the autoSDM pipeline.
#'
#' @param data A data frame formatted via `format_data()`. Must have standardized lowercase columns
#'   (longitude, latitude, year, present).
#' @param aoi Optional. Either a list with `lat`, `lon`, and `radius` (in meters), or a character string path to a polygon file.
#' @param output_dir Optional. Directory to save results. Defaults to the current working directory.
#' @param scale Optional. Resolution in meters for the final map. Defaults to 10.
#' @param python_path Optional. Path to Python executable. Auto-detected if not provided.
#' @param gee_project Optional. Google Cloud Project ID for Earth Engine.
#' @param cv Optional. Boolean whether to run spatial cross-validation. Defaults to FALSE.
#' @param predict_coords Optional. Data frame of coordinates to predict at.
#' @param year Optional. Alpha Earth Mosaic year for mapping. Defaults to 2025.
#' @param count Optional. Number of background points. Defaults to 10x presence points.
#' @param methods Optional character vector of method names. Supported:
#'   Classifiers: `"centroid"`, `"rf"`, `"gbt"`, `"cart"`, `"svm"`, `"maxent"`.
#'   Reducers: `"ridge"`, `"linear"`, `"robust_linear"`.
#'   Defaults to `c("centroid", "ridge")`.
#' @param ensemble Optional. Combine all methods into an ensemble map. Defaults to TRUE.
#' @param n_trees Optional. Number of trees for rf/gbt methods. Defaults to 100.
#' @param svm_kernel Optional. Kernel for SVM (`"LINEAR"`, `"POLY"`, `"RBF"`, `"SIGMOID"`). Defaults to `"RBF"`.
#' @param lambda_ Optional. Regularisation strength for ridge/linear reducers. Defaults to 0.1.
#' @return A list containing training data, models, and prediction results.
#' @export
autoSDM <- function(data, aoi = NULL, output_dir = getwd(), scale = NULL, python_path = NULL,
                    gee_project = NULL, cv = FALSE, predict_coords = NULL,
                    methods = NULL, ensemble = TRUE, year = NULL, count = NULL,
                    n_trees = 100L, svm_kernel = "RBF", lambda_ = 0.1) {
  # 1. Input Validation
  if (missing(data)) stop("Argument 'data' is required.")

  # Default AOI if not provided: Bounding box of data + buffer
  if (is.null(aoi) && is.null(predict_coords)) {
    message("No AOI provided. Calculating default bounding box from input data...")
    coords <- if (inherits(data, "sf")) sf::st_coordinates(data) else data[, c("longitude", "latitude")]

    min_lon <- min(coords[, 1], na.rm = TRUE)
    max_lon <- max(coords[, 1], na.rm = TRUE)
    min_lat <- min(coords[, 2], na.rm = TRUE)
    max_lat <- max(coords[, 2], na.rm = TRUE)

    lon_range <- max_lon - min_lon
    lat_range <- max_lat - min_lat
    buffer <- 0

    # Center and radius approximation for CLI
    center_lon <- (min_lon + max_lon) / 2
    center_lat <- (min_lat + max_lat) / 2
    # Radius in meters approx (1 deg ~ 111km)
    # We use the larger dimension to cover the rectangle with a circle
    radius_m <- max(lon_range, lat_range) / 2 * 111000

    aoi <- list(lat = center_lat, lon = center_lon, radius = radius_m)
    message(sprintf("  Default AOI: %0.1f km radius around %0.4f, %0.4f", radius_m / 1000, center_lat, center_lon))
  }

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
    methods <- if (has_absences) c("ridge") else c("centroid")
  }

  # Ensure "ensemble" is not accidentally listed in methods list
  methods <- setdiff(methods, "ensemble")

  if (length(methods) == 0) {
    stop("No valid matching methods found.")
  }

  # If more than one method is specified, ensemble is typically expected, but we respect the explicit argument.

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

  # If project was not provided, try to load it from the config that ensure_gee_authenticated might have just used/created
  if (is.null(gee_project)) {
    config_file <- file.path(Sys.getenv("HOME"), ".config", "autoSDM", "config.json")
    if (file.exists(config_file)) {
      try(
        {
          conf <- jsonlite::fromJSON(config_file)
          if (!is.null(conf$gee_project)) {
            gee_project <- conf$gee_project
            message("Loaded GEE Project from config: ", gee_project)
          }
        },
        silent = TRUE
      )
    }
  }

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Temporary directory for inter-process CV cache (never written to output_dir)
  cv_cache_dir <- tempfile(pattern = "autoSDM_cv_cache_")
  dir.create(cv_cache_dir, showWarnings = FALSE)
  cv_cache_arg <- c("--cv-cache-dir", shQuote(cv_cache_dir))

  # 5. Decide on Multi-Species
  is_multi_species <- "species" %in% names(data) && length(unique(data$species)) > 1

  # Use the raw input data as our base; analyze/predict CLI will sample GEE as needed.
  working_data <- data

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
  # working_data is our raw input (coords only)
  if (requireNamespace("vroom", quietly = TRUE)) {
    vroom::vroom_write(working_data, extract_csv, delim = ",")
  } else {
    write.csv(working_data, extract_csv, row.names = FALSE)
  }

  if (!is_multi_species) {
    message("--- Single-Species Mode ---")
    proj_arg <- if (!is.null(gee_project)) c("--project", shQuote(gee_project)) else NULL

    meta_files <- list()

    # Shared Background logic for Presence-Only
    # If any model needs background (centroid/ridge) and we don't have absences yet.
    needs_bg <- any(methods %in% c("centroid", "ridge")) &&
      !("present" %in% names(working_data) && any(working_data$present == 0))

    if (needs_bg && (!is.null(aoi) || !is.null(aoi_arg))) {
      message("--- Step: Generating Shared Background Points (10:1) ---")
      bg_csv <- tempfile(fileext = ".csv")
      n_bg <- nrow(working_data) * 10

      system2(python_path, args = c(
        "-m", "autoSDM.cli", "background",
        "--output", shQuote(bg_csv),
        "--count", if (!is.null(count)) count else n_bg,
        "--scale", scale,
        "--year", year,
        proj_arg, aoi_arg
      ))

      if (file.exists(bg_csv)) {
        bg_data <- read.csv(bg_csv)
        if (!"present" %in% names(bg_data)) bg_data$present <- 0
        if (!"present" %in% names(working_data)) working_data$present <- 1

        # Ensure column alignment
        common_cols <- intersect(names(working_data), names(bg_data))
        working_data <- rbind(working_data[, common_cols], bg_data[, common_cols])

        # Update standardized CSV with background points
        if (requireNamespace("vroom", quietly = TRUE)) {
          vroom::vroom_write(working_data, extract_csv, delim = ",")
        } else {
          write.csv(working_data, extract_csv, row.names = FALSE)
        }
      }
    }

    # Track execution time for each method
    execution_times <- list()

    # Use parallel::mclapply for concurrency on Unix platforms
    cores_to_use <- if (.Platform$OS.type == "unix") length(methods) else 1

    message(sprintf("--- Dispatching %d Method(s) in Parallel ---", length(methods)))

    parallel_results <- parallel::mclapply(methods, function(m) {
      m_clean <- gsub("[^a-zA-Z0-9]", "", m)
      message(sprintf("--- Step: Running %s Analysis ---", m))

      out_csv <- file.path(output_dir, paste0(m_clean, ".csv"))
      out_json <- file.path(output_dir, paste0(m_clean, ".json"))

      py_method <- m

      # Determine if this step should trigger CV evaluation for this submodel
      # If ensemble=TRUE, the submodels are just trained, CV evaluation happens at the end.
      cv_arg_analyze <- if (cv && !ensemble) {
        c("--cv", "--train-methods", m, "--eval-methods", m)
      } else {
        NULL
      }

      # Tuning args for this method
      tuning_args <- c(
        if (m %in% c("rf", "gbt")) c("--n-trees", as.character(as.integer(n_trees))),
        if (m == "svm") c("--svm-kernel", svm_kernel),
        if (m %in% c("ridge", "linear", "robust_linear")) c("--lambda", as.character(lambda_))
      )

      ret <- list(method = m, meta_file = NULL, training_seconds = NA)

      start_time <- Sys.time()
      system2(python_path, args = c(
        "-m", "autoSDM.cli", "analyze",
        "--input", shQuote(extract_csv),
        "--output", shQuote(out_csv),
        "--method", py_method,
        "--scale", scale,
        "--year", year,
        if (!is.null(count)) c("--count", count) else NULL,
        tuning_args, cv_arg_analyze, cv_cache_arg, proj_arg, aoi_arg
      ))
      end_time <- Sys.time()

      ret$training_seconds <- as.numeric(difftime(end_time, start_time, units = "secs"))

      if (file.exists(out_json)) {
        ret$meta_file <- out_json

        # 7. Generate Individual Map for this method (only when ensemble=FALSE)
        # When ensemble=TRUE, submodel metadata is still needed for the ensemble step,
        # but we skip the raster download — only the final ensemble map is output.
        if (!is.null(aoi) && !ensemble) {
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
      return(ret)
    }, mc.cores = cores_to_use)

    # Reconstruct meta_files and execution_times
    for (res in parallel_results) {
      if (!is.null(res$meta_file)) meta_files[[res$method]] <- res$meta_file
      if (is.null(execution_times[[res$method]])) execution_times[[res$method]] <- list()
      execution_times[[res$method]]$training_seconds <- res$training_seconds
    }

    # 8. Predict at specific coordinates (if provided)
    if (!is.null(predict_coords)) {
      message("--- Step: Predicting at specific coordinates (Parallel) ---")

      pred_results <- parallel::mclapply(methods, function(m) {
        meta_p <- file.path(output_dir, paste0(m, ".json"))
        if (!file.exists(meta_p)) {
          return(NULL)
        }

        m_meta <- jsonlite::fromJSON(meta_p)
        s_range <- m_meta$similarity_range

        pred_start_time <- Sys.time()
        preds <- predict_at_coords(predict_coords, analysis_meta_path = meta_p, scale = scale, year = year, python_path = python_path, gee_project = gee_project)
        pred_end_time <- Sys.time()

        if (!is.null(s_range) && (s_range[2] - s_range[1] > 1e-9)) {
          norm_sim <- (preds$similarity - s_range[1]) / (s_range[2] - s_range[1])
        } else {
          norm_sim <- preds$similarity
        }

        return(list(
          method = m,
          norm_sim = norm_sim,
          prediction_seconds = as.numeric(difftime(pred_end_time, pred_start_time, units = "secs"))
        ))
      }, mc.cores = cores_to_use)

      point_preds <- predict_coords
      point_preds$similarity <- 1.0

      for (res in pred_results) {
        if (!is.null(res)) {
          m <- res$method
          point_preds[[m]] <- res$norm_sim
          point_preds$similarity <- point_preds$similarity * res$norm_sim

          if (is.null(execution_times[[m]])) execution_times[[m]] <- list()
          execution_times[[m]]$prediction_seconds <- res$prediction_seconds
        }
      }

      # Final normalization of the ensemble product to 0-1
      e_min <- min(point_preds$similarity, na.rm = TRUE)
      e_max <- max(point_preds$similarity, na.rm = TRUE)
      if (e_max - e_min > 1e-9) {
        point_preds$similarity <- (point_preds$similarity - e_min) / (e_max - e_min)
      }
    }

    # 9. Generate Ensemble Map (if requested)
    if (!is.null(aoi) && ensemble) {
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

      # Run CV on ensemble formulation
      cv_arg_ensemble <- if (cv) {
        c("--cv", "--train-methods", paste(methods, collapse = ","), "--eval-methods", "ensemble")
      } else {
        NULL
      }

      if (!is.null(cv_arg_ensemble)) ensemble_args <- c(ensemble_args, cv_arg_ensemble, cv_cache_arg)

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
    if (length(execution_times) > 0) final_results$execution_times <- execution_times

    message("autoSDM pipeline complete!")
    return(final_results)
  } else {
    # MULTI-SPECIES WORKFLOW
    # -------------------------------------------------------------------------
    message(sprintf("--- Multi-Species Mode: Processing %d species ---", length(unique(working_data$species))))
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
    species_list <- unique(working_data$species)
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
