#' Initialize Earth Engine with Service Account
#'
#' @param json_path Path to the service account JSON key. Defaults to GEE_SERVICE_ACCOUNT_KEY env var.
#' @param venv_path Path to the Python virtual environment.
#' @keywords internal
ee_auth_service <- function(json_path = Sys.getenv("GEE_SERVICE_ACCOUNT_KEY"), venv_path = "venv") {
  # Store paths for later use
  sa_json_key <<- json_path
  py_venv_path <<- venv_path

  py_exe <- file.path(venv_path, "Scripts", "python.exe")
  if (!file.exists(py_exe)) {
    stop(sprintf("Python venv not found at %s. Please run setup first.", py_exe))
  }

  message("GEE credentials set for system calls.")
}

#' Extract Alpha Earth Embeddings
#'
#' @param df A data frame with Latitude, Longitude, and Year.
#' @param scale Resolution in meters. Defaults to 10.
#' @return A data frame with added A00-A63 embedding columns.
#' @keywords internal
extract_embeddings <- function(df, scale = 10) {
  if (!exists("py_venv_path")) {
    stop("Please run ee_auth_service() first.")
  }

  # Create temp files
  tmp_in <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")

  write.csv(df, tmp_in, row.names = FALSE)

  py_exe <- file.path(py_venv_path, "Scripts", "python.exe")

  message("Calling Python GEE extractor (via system)...")
  args <- c(
    "-m", "autoSDM.cli", "extract",
    "--input", shQuote(tmp_in),
    "--output", shQuote(tmp_out),
    "--scale", scale
  )

  if (exists("sa_json_key") && !is.null(sa_json_key) && sa_json_key != "") {
    args <- c(args, "--key", shQuote(sa_json_key))
  }

  status <- system2(py_exe, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Python extraction failed.")
  }

  res <- read.csv(tmp_out)
  # Clean up
  unlink(tmp_in)
  unlink(tmp_out)

  return(res)
}
