# Source local R files
r_files <- list.files("/home/james-longo/Projects/autoSDM/R", full.names = TRUE)
for (f in r_files) source(f)

# Resolve python
py_exe <- resolve_python_path(NULL)
message("Using Python: ", py_exe)

# Test simple CLI call
status <- system2(py_exe, args = c("-c", "print('Bridge OK')"), stdout = TRUE, stderr = TRUE)
print(status)
