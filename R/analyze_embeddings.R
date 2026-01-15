#' Analyze Embeddings
#'
#' This function calculates the similarity between each location's embedding and the species centroid (mean embedding).
#'
#' @param df Data frame containing columns A00 to A63.
#' @param method Mapping method: 'centroid' or 'standardized'.
#' @param nuisance_vars Character vector of columns to treat as nuisance variables.
#' @return A list containing the results.
#' @keywords internal
analyze_embeddings <- function(df, method = "centroid", nuisance_vars = NULL, python_path = NULL) {
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

  if (!is.null(nuisance_vars)) {
    args <- c(args, "--nuisance-vars", paste(nuisance_vars, collapse = ","))
  }

  status <- system2(python_path, args = args, stdout = "", stderr = "")

  if (status != 0) {
    stop("Python analysis failed.")
  }

  # Load results
  df_clean <- read.csv(tmp_out)

  # Load metadata
  meta_path <- paste0(tmp_out, ".json")
  meta <- jsonlite::fromJSON(meta_path)

  # Clean up
  unlink(tmp_in)
  unlink(tmp_out)
  unlink(meta_path)

  if (method == "centroid") {
    dot_products <- df_clean$similarity
    mean_emb <- meta$centroid

    # Get thresholds from new structure (may be nested under 'thresholds')
    thresholds <- meta$thresholds
    threshold_95tpr <- if (!is.null(thresholds$`95tpr`)) thresholds$`95tpr` else NULL
    threshold_balanced <- if (!is.null(thresholds$balanced)) thresholds$balanced else NULL

    # Create histogram (skip threshold line if not available)
    library(ggplot2)
    p <- ggplot(data.frame(dot_product = dot_products), aes(x = dot_product)) +
      geom_histogram(bins = 30, fill = "#69b3a2", color = "#e9ecef", alpha = 0.9) +
      theme_minimal() +
      labs(
        title = "Cosine Similarity to Species Centroid",
        subtitle = "Distribution of embedding similarities",
        x = "Cosine Similarity (Dot Product)",
        y = "Frequency"
      )

    # Add threshold line if available
    if (!is.null(threshold_95tpr) && is.numeric(threshold_95tpr)) {
      p <- p + geom_vline(xintercept = threshold_95tpr, color = "blue", linetype = "dotted", linewidth = 1)
    }

    return(list(
      mean_embedding = mean_emb,
      dot_products = dot_products,
      thresholds = thresholds,
      plot = p,
      data = df_clean,
      method = "centroid"
    ))
  } else {
    message("Maxent/RF model trained.")
    return(list(
      data = df_clean,
      method = method,
      thresholds = meta$thresholds,
      meta = meta
    ))
  }
}
