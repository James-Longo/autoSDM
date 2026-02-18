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

#' @return A data frame with added A00-A63 embedding columns.
#' @keywords internal
extract_embeddings <- function(df, scale = 10, python_path = NULL, gee_project = NULL) {
  python_path <- resolve_python_path(python_path)

  if (is.null(python_path)) {
    stop("python_path could not be resolved. Please configure reticulate or pass python_path explicitly.")
  }

  # Check if embeddings are already present
  emb_cols <- sprintf("A%02d", 0:63)
  if (all(emb_cols %in% names(df))) {
    message("Embeddings already present. Skipping extraction.")
    return(df)
  }

  # 1. Identify unique coordinate-year combinations to avoid redundant GEE calls
  # This is crucial for multi-species datasets like SatBird where many species share locations.
  dedup_cols <- c("longitude", "latitude", "year")
  keep_cols <- dedup_cols
  df_unique <- df[!duplicated(df[, dedup_cols]), keep_cols]

  # Create temp files
  tmp_in <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")

  write.csv(df_unique, tmp_in, row.names = FALSE)

  message(sprintf("Calling Python GEE extractor for %d unique locations...", nrow(df_unique)))
  args <- c(
    "-m", "autoSDM.cli", "extract",
    "--input", shQuote(tmp_in),
    "--output", shQuote(tmp_out),
    "--scale", scale
  )

  if (!is.null(gee_project) && gee_project != "") {
    args <- c(args, "--project", shQuote(gee_project))
  }

  if (exists("sa_json_key") && !is.null(sa_json_key) && sa_json_key != "") {
    args <- c(args, "--key", shQuote(sa_json_key))
  }

  status <- system2(python_path, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Python extraction failed.")
  }

  res_unique <- read.csv(tmp_out)
  # Clean up
  unlink(tmp_in)
  unlink(tmp_out)

  # 2. Join embeddings back to the original (multi-species) data frame
  # Standardize column types for robust joining
  res_unique$longitude <- as.numeric(res_unique$longitude)
  res_unique$latitude <- as.numeric(res_unique$latitude)
  res_unique$year <- as.numeric(res_unique$year)

  # Drop 'present' from extracted results to avoid duplicate columns on merge
  # (the original df already has the authoritative 'present' column)
  if ("present" %in% names(res_unique)) {
    res_unique$present <- NULL
  }

  df$longitude <- as.numeric(df$longitude)
  df$latitude <- as.numeric(df$latitude)
  df$year <- as.numeric(df$year)

  res <- merge(df, res_unique, by = c("longitude", "latitude", "year"), all.x = TRUE)

  return(res)
}
