#' Install autoSDM Python Dependencies
#'
#' This function installs the required Python dependencies for autoSDM
#' using reticulate.
#'
#' @param method Installation method. Pass "auto" to let reticulate decide.
#' @param envname The name, or full path, of the environment in which Python packages are to be installed.
#' @export
install_autoSDM <- function(method = "virtualenv", envname = "r-autoSDM") {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("The 'reticulate' package is required to manage Python dependencies.")
    }

    packages <- c("earthengine-api", "pandas", "numpy", "geopandas", "shapely", "scipy")

    message("Installing Python dependencies for autoSDM into environment: ", envname)

    # Force creation if it doesn't exist to avoid reticulate's uv search
    if (!reticulate::virtualenv_exists(envname)) {
        message("Creating virtual environment...")
        reticulate::virtualenv_create(envname)
    }

    reticulate::py_install(packages, envname = envname, method = method, pip = TRUE)

    message("\nSuccess! Python dependencies installed.")
}

#' Ensure Google Earth Engine is Authenticated (Internal)
#' @keywords internal
ensure_gee_authenticated <- function(project = NULL) {
    message("Checking Google Earth Engine authentication...")

    ee <- reticulate::import("ee", delay_load = FALSE)
    config_dir <- file.path(Sys.getenv("HOME"), ".config", "autoSDM")
    config_file <- file.path(config_dir, "config.json")

    # 1. Determine Project ID
    if (is.null(project) || project == "") {
        # Check local config
        if (file.exists(config_file)) {
            try(
                {
                    conf <- jsonlite::fromJSON(config_file)
                    project <- conf$gee_project
                },
                silent = TRUE
            )
        }
    }

    # 2. Try simple initialization
    initialized <- tryCatch(
        {
            if (!is.null(project) && project != "") {
                ee$Initialize(project = project)
            } else {
                ee$Initialize()
            }
            TRUE
        },
        error = function(e) FALSE
    )

    if (initialized) {
        return(TRUE)
    }

    # 3. Handle Authentication/Project Discovery
    message("\n[autoSDM] GEE initialization failed. Attempting auto-discovery...")

    # Check if we are authenticated at all
    tryCatch(
        {
            ee$data$getAuthorizedProjects() # This is a guess, let's use our helper
        },
        error = function(e) {
            # Suppress error, just checking if authenticated
            NULL
        }
    )

    # Use helper for discovery
    helper_path <- system.file("python/autoSDM/auth_helper.py", package = "autoSDM")
    if (helper_path == "" && file.exists("inst/python/autoSDM/auth_helper.py")) {
        helper_path <- "inst/python/autoSDM/auth_helper.py"
    }

    py_exe <- resolve_python_path()

    discover_projects <- function() {
        if (!file.exists(helper_path)) {
            return(character(0))
        }
        res <- system2(py_exe, args = shQuote(helper_path), stdout = TRUE, stderr = FALSE)
        if (length(res) == 0) {
            return(character(0))
        }
        tryCatch(jsonlite::fromJSON(res), error = function(e) character(0))
    }

    projs <- discover_projects()

    if (length(projs) == 0) {
        message("\n[autoSDM] No GEE credentials found. Opening browser for authentication...")
        ee$Authenticate()
        projs <- discover_projects()
    }

    if (length(projs) == 1) {
        project <- projs[1]
        message("Auto-selected GEE Project: ", project)
    } else if (length(projs) > 1) {
        message("\n[autoSDM] Multiple Google Cloud projects found.")
        idx <- utils::menu(projs, title = "Please select a project to use with Google Earth Engine:")
        if (idx == 0) stop("GEE Project selection required.")
        project <- projs[idx]
    } else {
        stop(
            "\n[autoSDM] No Google Cloud projects with Earth Engine access detected.\n",
            "Please create one here: https://code.earthengine.google.com/register"
        )
    }

    # 4. Initialize and Save
    tryCatch(
        {
            ee$Initialize(project = project)
            dir.create(config_dir, showWarnings = FALSE, recursive = TRUE)
            jsonlite::write_json(list(gee_project = project), config_file, auto_unbox = TRUE)
            message("GEE initialized and project saved to ", config_file)
            return(TRUE)
        },
        error = function(e) {
            stop("GEE Initialization failed for project '", project, "': ", e$message)
        }
    )
}


#' Ensure dependencies are met (Internal)
#' @keywords internal
ensure_autoSDM_dependencies <- function() {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("The 'reticulate' package is required.")
    }

    packages <- c("ee", "pandas", "numpy", "geopandas", "shapely", "scipy")
    env_name <- "r-autoSDM"

    # CRITICAL: Check for virtualenv BEFORE any imports to prevent initialization conflict
    if (reticulate::virtualenv_exists(env_name)) {
        reticulate::use_virtualenv(env_name, required = FALSE)
    }

    status <- tryCatch(
        {
            # Attempt to import. This will initialize Python.
            for (pkg in packages) {
                reticulate::import(pkg)
            }
            TRUE
        },
        error = function(e) FALSE
    )

    if (!status) {
        message("\n[autoSDM] Python dependencies missing.")

        if (interactive()) {
            confirm <- utils::askYesNo("Would you like to automatically setup the Python environment?")
            if (isTRUE(confirm)) {
                install_autoSDM(envname = env_name)
                # We can't switch in this session if initialization happened,
                # but we can try to use it if it hasn't.
                reticulate::use_virtualenv(env_name, required = TRUE)
                return(reticulate::py_config()$python)
            }
        }

        stop("Please run `autoSDM::install_autoSDM()` and then restart your R session.")
    }

    return(reticulate::py_config()$python)
}

#' Resolve Python Path (Internal)
#' @keywords internal
resolve_python_path <- function(path = NULL) {
    if (!is.null(path)) {
        return(path)
    }

    # Preference for our dedicated env
    if (reticulate::virtualenv_exists("r-autoSDM")) {
        return(reticulate::virtualenv_python("r-autoSDM"))
    }

    if (reticulate::py_available()) {
        return(reticulate::py_config()$python)
    }

    return(NULL)
}
