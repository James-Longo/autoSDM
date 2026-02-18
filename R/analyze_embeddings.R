#' Analyze Embeddings
#'
#' This function calculates the similarity between each location's embedding and the species centroid (mean embedding).
#'
#' @param df Data frame containing columns A00 to A63.
#' @param method Mapping method: 'centroid' or 'ridge'.
#' @return A list containing the results.
#' @keywords internal
analyze_embeddings <- function(df, method = "centroid", python_path = NULL, gee_project = NULL, cv = FALSE) {
  python_path <- resolve_python_path(python_path)

  if (is.null(python_path)) {
    stop("python_path could not be resolved. Please configure reticulate or pass python_path explicitly.")
  }

  # Create temp files
  tmp_in <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")

  write.csv(df, tmp_in, row.names = FALSE)

  message("Calling Python analyzer (via system)...")
  args <- c(
    "-m", "autoSDM.cli", "analyze",
    "--input", shQuote(tmp_in),
    "--output", shQuote(tmp_out),
    "--method", method
  )

  if (!is.null(gee_project) && gee_project != "") {
    args <- c(args, "--project", shQuote(gee_project))
  }

  if (cv) {
    args <- c(args, "--cv")
  }

  status <- system2(python_path, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Python analysis failed.")
  }

  # Load results
  df_clean <- read.csv(tmp_out)

  # Load metadata
  meta_path <- sub("\\.csv$", ".json", tmp_out)
  if (!file.exists(meta_path)) meta_path <- paste0(tmp_out, ".json") # Fallback
  meta <- jsonlite::fromJSON(meta_path)

  # Clean up
  unlink(tmp_in)
  unlink(tmp_out)
  unlink(meta_path)

  dot_products <- df_clean$similarity
  # Centroid extraction handling
  mean_emb <- if (!is.null(meta$centroids)) meta$centroids[[1]] else meta$centroid

  metrics <- meta$metrics

  return(list(
    mean_embedding = mean_emb,
    dot_products = dot_products,
    metrics = metrics,
    data = df_clean,
    method = method
  ))
}
