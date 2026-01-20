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

    required_pkgs <- c("earthengine-api", "pandas", "numpy", "geopandas", "shapely", "scipy")

    # Check import - use simple import test for each package
    # earthengine-api imports as 'ee', others import as their package name
    import_names <- c("ee", "pandas", "numpy", "geopandas", "shapely", "scipy")

    check_code <- paste0(
        "try:\n",
        paste0("    import ", import_names, collapse = "\n"),
        "\n    print('OK')\nexcept: exit(1)"
    )

    status <- system2(python_path, args = c("-c", shQuote(check_code)), stdout = FALSE, stderr = FALSE)

    if (status != 0) {
        message("Missing required Python dependencies. Installing now... (This may take a moment)")

        # We assume pip is available as -m pip.
        install_args <- c("-m", "pip", "install", required_pkgs, "--quiet", "--upgrade")

        install_status <- system2(python_path, args = install_args)

        if (install_status != 0) {
            stop("Failed to automatically install dependencies. Please install 'earthengine-api', 'pandas', 'geopandas', 'shapely' manually in your python environment.")
        }
        message("Dependencies installed successfully.")
    }
}

#' Resolve Python Path
#'
#' Helper to find the Python executable using multiple fallback methods:
#' 1. Explicit argument
#' 2. Reticulate active/discovered configuration
#' 3. System PATH (python3, python)
#' 4. Legacy global variable (py_venv_path)
#'
#' @param path Optional explicit path.
#' @return Path to python executable or NULL.
#' @keywords internal
resolve_python_path <- function(path = NULL) {
    # 1. Explicit argument
    if (!is.null(path) && file.exists(path)) {
        return(path)
    }

    # 2. Try reticulate's py_config (active python)
    path <- tryCatch(
        {
            cfg <- reticulate::py_config()
            if (!is.null(cfg$python) && file.exists(cfg$python)) cfg$python else NULL
        },
        error = function(e) NULL
    )
    if (!is.null(path)) {
        return(path)
    }

    # 3. Try reticulate's py_discover_config (find any python)
    path <- tryCatch(
        {
            cfg <- reticulate::py_discover_config()
            if (!is.null(cfg$python) && file.exists(cfg$python)) cfg$python else NULL
        },
        error = function(e) NULL
    )
    if (!is.null(path)) {
        return(path)
    }

    # 4. Try system PATH - look for python3 or python
    for (cmd in c("python3", "python")) {
        sys_python <- Sys.which(cmd)
        if (sys_python != "" && file.exists(sys_python)) {
            return(sys_python)
        }
    }

    # 5. Legacy global variable fallback
    if (exists("py_venv_path", envir = .GlobalEnv)) {
        venv <- get("py_venv_path", envir = .GlobalEnv)
        candidate <- file.path(venv, "Scripts", "python.exe")
        if (file.exists(candidate)) {
            return(candidate)
        }

        candidate <- file.path(venv, "bin", "python")
        if (file.exists(candidate)) {
            return(candidate)
        }
    }

    return(NULL)
}
