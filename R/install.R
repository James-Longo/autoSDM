#' Install autoSDM Python Dependencies
#'
#' This function creates a dedicated Python environment for autoSDM
#' and installs the required packages (earthengine-api, pandas, etc.).
#'
#' @param envname Name of the environment. Defaults to "r-autoSDM".
#' @param method Installation method. "auto", "virtualenv", or "conda".
#' @param ... Additional arguments passed to reticulate::py_install.
#' @export
install_autoSDM <- function(envname = "r-autoSDM", method = "auto", ...) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("Package 'reticulate' is required. Please install it.")
    }

    message("Installing Python dependencies for autoSDM...")

    pkgs <- c(
        "earthengine-api",
        "pandas",
        "numpy",
        "geopandas",
        "shapely"
    )

    reticulate::py_install(
        packages = pkgs,
        envname = envname,
        method = method,
        ...
    )

    message("Installation complete. To use this environment, restart R and run:")
    message(sprintf('reticulate::use_virtualenv("%s", required = TRUE)', envname))
}
