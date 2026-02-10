#' Initialize Earth Engine with Service Account
#'
#' @param json_path Path to the service account JSON key. Defaults to GEE_SERVICE_ACCOUNT_KEY env var.
#' @param venv_path Path to the Python virtual environment.
#' @keywords internal
ee_auth_service <- function(json_path = Sys.getenv("GEE_SERVICE_ACCOUNT_KEY"), venv_path = NULL) {
  # Store paths for later use
  sa_json_key <<- json_path

  if (!is.null(venv_path)) {
    py_venv_path <<- venv_path

    py_exe <- file.path(venv_path, "Scripts", "python.exe")
    if (!file.exists(py_exe)) py_exe <- file.path(venv_path, "bin", "python")

    if (!file.exists(py_exe)) {
      stop(sprintf("Python venv not found at %s. Please run setup first.", venv_path))
    }
  }

  message("GEE credentials set for system calls.")
}

#' @param background_method Optional. Method to generate background points ("sample_extent" or "buffer").
#' @param background_buffer Optional. Numeric vector of length 2: c(min_dist, max_dist).
#' @return A data frame with added A00-A63 embedding columns.
#' @keywords internal
extract_embeddings <- function(df, scale = 10, python_path = NULL, gee_project = NULL, background_method = NULL, background_buffer = NULL) {
  python_path <- resolve_python_path(python_path)

  if (is.null(python_path)) {
    stop("python_path could not be resolved. Please configure reticulate or pass python_path explicitly.")
  }

  # Create temp files
  tmp_in <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")

  write.csv(df, tmp_in, row.names = FALSE)

  message("Calling Python GEE extractor (via system)...")
  args <- c(
    "-m", "autoSDM.cli", "extract",
    "--input", shQuote(tmp_in),
    "--output", shQuote(tmp_out),
    "--scale", scale
  )

  if (!is.null(gee_project) && gee_project != "") {
    args <- c(args, "--project", shQuote(gee_project))
  }


  if (!is.null(background_method)) {
    args <- c(args, "--background-method", background_method)
  }

  if (!is.null(background_buffer)) {
    if (length(background_buffer) != 2) {
      stop("background_buffer must be a numeric vector of length 2: c(min_dist, max_dist)")
    }
    args <- c(args, "--background-buffer", background_buffer[1], background_buffer[2])
  }

  if (exists("sa_json_key") && !is.null(sa_json_key) && sa_json_key != "") {
    args <- c(args, "--key", shQuote(sa_json_key))
  }

  status <- system2(python_path, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Python extraction failed.")
  }

  res <- read.csv(tmp_out)
  # Clean up
  unlink(tmp_in)
  unlink(tmp_out)

  return(res)
}
