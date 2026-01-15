#' Analyze Embeddings
#'
#' This function calculates the similarity between each location's embedding and the species centroid (mean embedding).
#'
#' @param df Data frame containing columns A00 to A63.
#' @param method Mapping method: 'centroid' or 'standardized'.
#' @param nuisance_vars Character vector of columns to treat as nuisance variables.
#' @return A list containing the results.
#' @keywords internal
analyze_embeddings <- function(df, method = "centroid", nuisance_vars = NULL) {
  if (!exists("py_venv_path")) {
    stop("Please run ee_auth_service() first.")
  }

  # Create temp files
  tmp_in <- tempfile(fileext = ".csv")
  tmp_out <- tempfile(fileext = ".csv")

  write.csv(df, tmp_in, row.names = FALSE)

  py_exe <- file.path(py_venv_path, "Scripts", "python.exe")

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

  status <- system2(py_exe, args = args, stdout = "", stderr = "")

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
    threshold_5pct <- meta$threshold_5pct
    threshold_50pct <- meta$threshold_50pct

    # Create histogram
    library(ggplot2)
    p <- ggplot(data.frame(dot_product = dot_products), aes(x = dot_product)) +
      geom_histogram(bins = 30, fill = "#69b3a2", color = "#e9ecef", alpha = 0.9) +
      geom_vline(aes(xintercept = threshold_5pct, color = "5th Percentile"), linetype = "dotted", linewidth = 1) +
      scale_color_manual(name = "Threshold", values = c("5th Percentile" = "blue")) +
      theme_minimal() +
      labs(
        title = "Cosine Similarity to Species Centroid",
        subtitle = paste0("Core Niche Analysis (5th Pct Threshold: ", round(threshold_5pct, 3), ")"),
        x = "Cosine Similarity (Dot Product)",
        y = "Frequency"
      )

    return(list(
      mean_embedding = mean_emb,
      dot_products = dot_products,
      threshold_5pct = threshold_5pct,
      threshold_50pct = threshold_50pct,
      plot = p,
      data = df_clean,
      method = "centroid"
    ))
  } else {
    message("Random Forest model trained and nuisance optima determined.")
    return(list(
      data = df_clean,
      method = "standardized",
      nuisance_optima = meta$nuisance_optima,
      ecological_vars = meta$ecological_vars,
      nuisance_vars = meta$nuisance_vars,
      meta = meta
    ))
  }
}
