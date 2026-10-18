# config.R
# Settings and helper functions for the R scripts (01, 02). Scripts source it from the repo root,
# so they must be run from there.

# Paths. figure_outdir and data_dir are relative to the repo root.
figure_outdir <- "figures"
data_dir <- "data"     # the marker gene workbook (HBCA_Reed_Top_Marker_Genes.xlsx)

# Machine-specific paths live in local_paths.env in the repo root (gitignored).
# Copy local_paths.env.example to local_paths.env and fill it in.
local_paths_file <- "local_paths.env"
if (!file.exists(local_paths_file)) {
    stop(sprintf("%s not found in %s. Copy local_paths.env.example to local_paths.env and fill in your paths.",
                 local_paths_file, getwd()))
}
readRenviron(local_paths_file)

# Look up one path from local_paths.env by name, e.g. get_local_path("HBCA_DIR").
# Stops with a message if it isn't set.
get_local_path <- function (name) {
    value <- Sys.getenv(name)
    if (value == "") {
        stop(sprintf("%s is not set in %s", name, local_paths_file))
    }
    value
}

# The storage folder: every script reads and writes its data objects here.
OUTPUT_DIR <- get_local_path("STORAGE_DIR")

# Print an object to the log the way the console would show it.
# Used in scripts because cat() can't print tables or data frames.
catObj <- function (x) {
    cat(paste0(capture.output(x), '\n'))
}

# Plot settings: point size for cells on UMAPs, and the larger point size used in legends
point_size <- 0.001
legend_point_size <- 3

# Seed for random steps (set.seed and BiocParallel's RNGseed)
rng.seed <- 0
