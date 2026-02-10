#' Install autoSDM Python Dependencies
#'
#' This function creates a dedicated Python virtual environment for autoSDM and
#' installs all required Python dependencies (earthengine-api, pandas, etc.).
#'
#' @param envname Name of the virtual environment. Defaults to "r-autoSDM".
#' @param restart_session Whether to restart the R session after installation. Defaults to TRUE in interactive sessions.
#' @export
install_autoSDM <- function(envname = "r-autoSDM", restart_session = interactive()) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("The 'reticulate' package is required to manage Python dependencies.")
    }

    required_pkgs <- c("earthengine-api", "pandas", "numpy", "geopandas", "shapely", "scipy")

    message("Creating dedicated virtual environment: ", envname)

    # Create virtualenv if it doesn't exist
    if (!reticulate::virtualenv_exists(envname)) {
        tryCatch(
            {
                reticulate::virtualenv_create(envname)
            },
            error = function(e) {
                # Fallback to system python if default fails
                system_python <- Sys.which("python3")
                if (system_python == "") system_python <- Sys.which("python")
                reticulate::virtualenv_create(envname, python = system_python)
            }
        )
    }

    message("Installing Python dependencies...")
    reticulate::virtualenv_install(envname, packages = required_pkgs)

    message("\nSuccess! Python dependencies installed in virtualenv: ", envname)

    if (restart_session) {
        message("Please restart your R session to use the new environment.")
    }
}

#' Ensure autoSDM Python Dependencies (Internal)
#' @keywords internal
ensure_autoSDM_dependencies <- function(python_path = NULL) {
    env_name <- "r-autoSDM"
    required_pkgs <- c("earthengine-api", "pandas", "numpy", "geopandas", "shapely", "scipy")
    import_names <- c("ee", "pandas", "numpy", "geopandas", "shapely", "scipy")

    # 1. If user provided a path, try to use it
    if (!is.null(python_path)) {
        reticulate::use_python(python_path, required = TRUE)
    } else if (reticulate::virtualenv_exists(env_name)) {
        # 2. Otherwise, prefer our dedicated venv
        reticulate::use_virtualenv(env_name, required = TRUE)
        python_path <- reticulate::virtualenv_python(env_name)
    }

    # 3. Check if we can import the core modules
    message("Checking Python dependencies...")
    status <- tryCatch(
        {
            for (pkg in import_names) {
                reticulate::import(pkg)
            }
            TRUE
        },
        error = function(e) FALSE
    )

    if (!status) {
        message("\n[autoSDM] Python dependencies are missing or the environment is broken.")

        if (interactive()) {
            confirm <- utils::askYesNo("Would you like to automatically create a dedicated Python environment for autoSDM?")
            if (!isTRUE(confirm)) {
                stop("Python dependencies missing. Please run `autoSDM::install_autoSDM()` to set up your environment.")
            }
        } else {
            message("Non-interactive session detected. Automatically setting up dedicated Python environment...")
        }

        install_autoSDM(envname = env_name, restart_session = FALSE)
        reticulate::use_virtualenv(env_name, required = TRUE)
        return(reticulate::virtualenv_python(env_name))
    }

    return(python_path)
}

#' Resolve Python Path (Internal)
#' @keywords internal
resolve_python_path <- function(path = NULL) {
    env_name <- "r-autoSDM"

    # 0. Check for our dedicated venv first
    if (reticulate::virtualenv_exists(env_name)) {
        return(reticulate::virtualenv_python(env_name))
    }

    # 1. Explicit argument
    if (!is.null(path) && file.exists(path)) {
        return(path)
    }

    # 2. Try reticulate configuration
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

    # 3. System fallback
    for (cmd in c("python3", "python")) {
        sys_python <- Sys.which(cmd)
        if (sys_python != "" && file.exists(sys_python)) {
            return(sys_python)
        }
    }

    return(NULL)
}

#' Ensure Google Earth Engine is Authenticated
#' @keywords internal
ensure_gee_authenticated <- function(python_path) {
    message("Checking Google Earth Engine authentication...")

    ee <- reticulate::import("ee", delay_load = FALSE)

    # Try to initialize
    tryCatch(
        {
            ee$Initialize()
            return(TRUE)
        },
        error = function(e) {
            message("\n[autoSDM] Google Earth Engine authentication is missing or expired.")

            if (interactive()) {
                confirm <- utils::askYesNo("Would you like to open a browser to authenticate Google Earth Engine now?")
                if (!isTRUE(confirm)) {
                    stop("GEE authentication is required to extract satellite embeddings.")
                }
            }

            message("Opening browser for GEE authentication...")
            ee$Authenticate()

            # Try initializing again after authentication
            ee$Initialize()
            return(TRUE)
        }
    )
}
