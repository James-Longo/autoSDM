#' Format Data for autoSDM
#'
#' This function standardizes input data frames for use in the autoSDM pipeline.
#' It ensures coordinate columns are named 'Latitude' and 'Longitude',
#' extracts the year from date columns if necessary, and renames the
#' presence column to 'present?'.
#'
#' @param data A data frame containing your survey data.
#' @param coords A character vector of length 2 specifying the longitude and latitude columns (e.g., c("lon", "lat")).
#' @param year A character string specifying the year or date column.
#' @param presence Optional. A character string specifying the presence/absence column (values should be 0 or 1).
#' @return A standardized data frame ready for `autoSDM()`.
#' @export
format_data <- function(data, coords, year, presence = NULL) {
    if (!is.data.frame(data)) {
        stop("Input 'data' must be a data frame.")
    }

    # 1. Coordinate Standardization
    if (length(coords) != 2) {
        stop("'coords' must be a character vector of length 2: c(longitude_col, latitude_col)")
    }

    # Check if columns exist
    missing_cols <- setdiff(coords, names(data))
    if (length(missing_cols) > 0) {
        stop(paste("Coordinate columns not found in data:", paste(missing_cols, collapse = ", ")))
    }

    # Rename to Longitude/Latitude
    names(data)[names(data) == coords[1]] <- "Longitude"
    names(data)[names(data) == coords[2]] <- "Latitude"

    # 2. Year Standardization
    if (!year %in% names(data)) {
        stop(paste("Year column not found:", year))
    }

    # Try to extract year if it's a date
    if (inherits(data[[year]], c("Date", "POSIXt"))) {
        data$Year <- as.numeric(format(data[[year]], "%Y"))
    } else {
        # Try converting to numeric year first
        val <- suppressWarnings(as.numeric(data[[year]]))

        # Check if straight numeric conversion worked (ignoring NAs in original data)
        # We consider it "worked" if non-NA inputs became numbers, or if all were NA.
        # But here we want to handle mixed cases or string dates.

        if (all(is.na(val) & !is.na(data[[year]]))) {
            # All non-NA values failed to convert to numeric, so likely a date string

            # Attempt various formats
            formats <- c("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d")
            parsed <- as.Date(rep(NA, nrow(data)))
            success <- FALSE

            for (fmt in formats) {
                try_date <- as.Date(data[[year]], format = fmt)
                if (any(!is.na(try_date))) {
                    # If we successfully parsed at least some dates that weren't NA
                    # Ideally we check if we parsed *all* non-NA strings, but data might be messy.
                    # Let's assume if we get reasonable number of dates, it's the right format.
                    parsed <- try_date
                    success <- TRUE
                    break
                }
            }

            # Fallback: try standard as.Date (ISO)
            if (!success) {
                try_date <- try(as.Date(data[[year]]), silent = TRUE)
                if (!inherits(try_date, "try-error")) {
                    parsed <- try_date
                }
            }

            data$Year <- as.numeric(format(parsed, "%Y"))

            # If we still have NAs where we had data, might warn?
            # For now, we proceed.
        } else {
            # Numeric conversion mostly worked, or it was mixed.
            # Use the numeric values.
            data$Year <- val
        }
    }

    # 3. Presence Standardization
    if (!is.null(presence)) {
        if (!presence %in% names(data)) {
            stop(paste("Presence column not found:", presence))
        }
        names(data)[names(data) == presence] <- "present?"
    }

    message("Data formatted successfully. Columns: ", paste(names(data), collapse = ", "))
    return(data)
}
