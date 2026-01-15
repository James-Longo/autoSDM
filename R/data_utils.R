#' Format Data for autoSDM
#'
#' This function standardizes input data frames for use in the autoSDM pipeline.
#' It renames coordinate/year/presence columns to standard lowercase names,
#' optionally keeps nuisance variables, and removes all other columns.
#'
#' @param data A data frame containing your survey data.
#' @param coords A character vector of length 2 specifying the longitude and latitude columns IN ORDER: c(longitude_col, latitude_col). Note: Longitude first, then Latitude!
#' @param year A character string specifying the year or date column.
#' @param presence Optional. A character string specifying the presence/absence column (values should be 0 or 1).
#' @param nuisance_vars Optional. A character vector of column names to keep as nuisance variables (e.g., c("elevation", "aspect")).
#' @return A standardized data frame ready for `autoSDM()` with lowercase column names.
#' @export
format_data <- function(data, coords, year, presence = NULL, nuisance_vars = NULL) {
    if (!is.data.frame(data)) {
        stop("Input 'data' must be a data frame.")
    }

    # 1. Coordinate Validation
    if (length(coords) != 2) {
        stop("'coords' must be a character vector of length 2: c(longitude_col, latitude_col)")
    }

    missing_cols <- setdiff(coords, names(data))
    if (length(missing_cols) > 0) {
        stop(paste("Coordinate columns not found in data:", paste(missing_cols, collapse = ", ")))
    }

    # 2. Year Validation
    if (!year %in% names(data)) {
        stop(paste("Year column not found:", year))
    }

    # 3. Presence Validation
    if (!is.null(presence) && !presence %in% names(data)) {
        stop(paste("Presence column not found:", presence))
    }

    # 4. Nuisance Variables Validation
    if (!is.null(nuisance_vars)) {
        missing_nuisance <- setdiff(nuisance_vars, names(data))
        if (length(missing_nuisance) > 0) {
            stop(paste("Nuisance columns not found in data:", paste(missing_nuisance, collapse = ", ")))
        }
    }

    # 5. Build the output data frame with only required columns
    result <- data.frame(
        longitude = data[[coords[1]]],
        latitude = data[[coords[2]]],
        stringsAsFactors = FALSE
    )

    # Sanity check: Warn if coordinates appear swapped
    sample_lat <- result$latitude[!is.na(result$latitude)][1]
    sample_lon <- result$longitude[!is.na(result$longitude)][1]

    if (!is.null(sample_lat) && !is.null(sample_lon)) {
        lat_in_range <- sample_lat >= -90 && sample_lat <= 90
        lon_in_range <- sample_lon >= -180 && sample_lon <= 180

        if (!lat_in_range && lon_in_range) {
            warning(sprintf(
                "Coordinates may be SWAPPED! Latitude=%s is outside valid range [-90, 90].\n  Did you pass coords in the correct order? It should be: coords = c(longitude_col, latitude_col)\n  Your call: coords = c('%s', '%s')",
                sample_lat, coords[1], coords[2]
            ))
        }
    }

    # 6. Process Year (handle dates)
    year_data <- data[[year]]
    if (inherits(year_data, c("Date", "POSIXt"))) {
        result$year <- as.numeric(format(year_data, "%Y"))
    } else {
        # Try numeric conversion
        val <- suppressWarnings(as.numeric(year_data))

        if (all(is.na(val) & !is.na(year_data))) {
            # Likely date strings - try common formats
            formats <- c("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d")
            parsed <- as.Date(rep(NA, nrow(data)))

            for (fmt in formats) {
                try_date <- as.Date(year_data, format = fmt)
                if (any(!is.na(try_date))) {
                    parsed <- try_date
                    break
                }
            }

            # Fallback to standard as.Date
            if (all(is.na(parsed))) {
                try_date <- try(as.Date(year_data), silent = TRUE)
                if (!inherits(try_date, "try-error")) {
                    parsed <- try_date
                }
            }

            result$year <- as.numeric(format(parsed, "%Y"))
        } else {
            result$year <- val
        }
    }

    # 7. Add presence column (lowercase "present")
    if (!is.null(presence)) {
        result$present <- as.numeric(data[[presence]])
    }

    # 8. Add nuisance variables (lowercase names)
    if (!is.null(nuisance_vars)) {
        for (nv in nuisance_vars) {
            # Create lowercase column name
            col_name <- tolower(gsub("[^a-zA-Z0-9]", "_", nv))
            result[[col_name]] <- data[[nv]]
        }
    }

    # 9. Filter to years with Alpha Earth data (2017+)
    rows_before <- nrow(result)
    result <- result[result$year >= 2017 & result$year <= 2024, ]
    rows_after <- nrow(result)

    if (rows_before != rows_after) {
        message(sprintf("Removed %d rows outside Alpha Earth coverage (2017-2024).", rows_before - rows_after))
    }

    # 10. Remove rows with NA values
    rows_before <- nrow(result)
    result <- na.omit(result)
    rows_after <- nrow(result)

    if (rows_before != rows_after) {
        message(sprintf("Removed %d rows with missing values.", rows_before - rows_after))
    }

    # 11. Remove duplicate rows
    rows_before <- nrow(result)
    result <- unique(result)
    rows_after <- nrow(result)

    if (rows_before != rows_after) {
        message(sprintf("Removed %d duplicate rows.", rows_before - rows_after))
    }

    # List final columns
    cols_desc <- paste(names(result), collapse = ", ")
    message(sprintf("Data formatted: %d rows, %d columns (%s)", nrow(result), ncol(result), cols_desc))

    return(result)
}
