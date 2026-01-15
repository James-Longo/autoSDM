#' Ensure autoSDM Python Dependencies
#'
#' Checks if required Python packages are installed in the given Python environment.
#' If missing, attempts to install them automatically.
#'
#' @param python_path Path to the python executable.
#' @keywords internal
ensure_autoSDM_dependencies <- function(python_path) {
    if (is.null(python_path) || !file.exists(python_path)) {
        stop("Invalid python path provided to dependency checker.")
    }

    required_pkgs <- c("earthengine-api", "pandas", "numpy", "geopandas", "shapely")

    # Check import
    # specific check script
    check_code <- paste0(
        "try:\n",
        paste0("    import ", required_pkgs, collapse = "\n"),
        "\n    print('OK')\n",
        "except Exception as e: exit(1)"
    )

    status <- system2(python_path, args = c("-c", shQuote(check_code)), stdout = FALSE, stderr = FALSE)

    if (status != 0) {
        message("Missing required Python dependencies. Installing now... (This may take a moment)")

        # We assume pip is available as -m pip.
        # This is standard for most python envs.
        install_args <- c("-m", "pip", "install", required_pkgs, "--quiet", "--upgrade")

        install_status <- system2(python_path, args = install_args)

        if (install_status != 0) {
            stop("Failed to automatically install dependencies. Please install 'earthengine-api', 'pandas', 'geopandas', 'shapely' manually in your python environment.")
        }
        message("Dependencies installed successfully.")
    }

#' Resolve Python Path
#'
#' Helper to find the Python executable using:
#' 1. Explicit argument
#' 2. Reticulate active configuration
#' 3. Legacy global variable (py_venv_path)
#'
#' @param path Optional explicit path.
#' @return Path to python executable or NULL.
#' @keywords internal
resolve_python_path <- function(path = NULL) {
    if (!is.null(path)) return(path)

    # Try reticulate
    path <- tryCatch(
        {
            reticulate::py_config()$python
        },
        error = function(e) {
            NULL
        }
    )
    if (!is.null(path)) return(path)

    # Legacy global
    if (exists("py_venv_path")) {
        candidate <- file.path(py_venv_path, "Scripts", "python.exe")
        if (file.exists(candidate)) return(candidate)
        
        candidate <- file.path(py_venv_path, "bin", "python")
        if (file.exists(candidate)) return(candidate)
    }

    return(NULL)
}
