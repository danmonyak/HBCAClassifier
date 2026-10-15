# config.R
# Config file for the Reed_Explore scripts


# Paths. figure_outdir is relative, so scripts must be run from inside this folder.
RNAseq_SingleCell_dir = file.path(repo_dir, "RNAseq_SingleCell")
figure_outdir = "figures"

# Shared config: OUTPUT_DIR, data_dir, get_local_path() and helper functions
source(file.path(RNAseq_SingleCell_dir, "config.R"))

# Plot settings: point size for cells on UMAPs, and the larger point size used in legends
point_size <- 0.001
legend_point_size <- 3

# Seed for random steps (set.seed and BiocParallel's RNGseed)
rng.seed <- 0
