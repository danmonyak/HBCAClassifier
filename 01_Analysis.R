# 01_Analysis.R
#
# Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.), the reference used
# for level-2 label transfer in Cell_Type_Assignment/02_SingleR_Reed_transfer.R.
# Loads the atlas, ready for the analysis below.
#
# Needs:    hbca-global.h5ad, in the folder set as HBCA_DIR in local_paths.env
# Produces: nothing yet. It prints the atlas's size and main labels.
#
# Usage: Rscript 01_Analysis.R [cellranger_run_name]

library(SingleCellExperiment)
library(dplyr)
library(tidySingleCellExperiment)
library(tibble)
library(ggplot2)
library(patchwork)
library(scuttle)
library(scater)
library(anndataR)

# Standard header: load the folder config, read the run name (default 90),
# and set up the storage folder and this script's figure folder.
source(file.path(repo_dir, "RNAseq_SingleCell", "Reed_Explore", "config.R"))
cellranger_run_name <- get_cellranger_run_name_cmd_arg()
output_dir <- file.path(OUTPUT_DIR, cellranger_run_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

figure_outdir_run = file.path(figure_outdir, cellranger_run_name)
figure_outdir_cur = file.path(figure_outdir_run, '01')
dir.create(figure_outdir_cur, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\nCellranger run: %s\n", cellranger_run_name))
cat(sprintf("%s\n\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

# Make random steps (UMAP, clustering, subsampling) reproducible
set.seed(rng.seed)

# ------------------------------------------------------------------------

# ---------------------- Read the HBCA reference
# Same as Cell_Type_Assignment/02_SingleR_Reed_transfer.R: the atlas, read as a SingleCellExperiment
cat("Loading reference dataset...")
reed_dir <- get_local_path("HBCA_DIR")
reed <- read_h5ad(file.path(reed_dir, "hbca-global.h5ad"), as = "SingleCellExperiment")
cat("done\n")

# Quick look at the atlas: size, and cells per broad compartment (level1) and subtype (level2)
cat(sprintf("%d genes x %d cells\n", nrow(reed), ncol(reed)))
catObj(sort(table(reed$level1), decreasing = TRUE))
catObj(sort(table(reed$level2), decreasing = TRUE))

# ---------------------- Analysis
# reed (the atlas): useful colData columns are level0, level1 and level2 (the label hierarchy;
#   level2 is what SingleR transferred to our cells), cell_type, donor_id, risk_status,
#   brca_status and sample_type. Row names are Ensembl IDs, like ours (gene symbols are in
#   rowData(reed)$feature_name).
#
# Save figures to figure_outdir_cur, e.g.
#   ggsave(file.path(figure_outdir_cur, "my_plot.png"), plot = p, width = 8, height = 6, dpi = 200)
# Save objects that a later script needs to output_dir, e.g.
#   saveRDS(result, file.path(output_dir, "my_result.rds"))

# ---------------------- Level-2 subtype make-up of each donor
# A donors x subtypes table: each value is the fraction of that donor's cells that
# have that level2 subtype, so every row sums to 1.
# as.character() drops unused factor levels, so donors or subtypes with no cells
# don't appear as empty (NaN) rows or all-zero columns.
donor.level2.counts <- table(
  donor_id = as.character(reed$donor_id),
  level2 = as.character(reed$level2)
)
donor.level2.props <- prop.table(donor.level2.counts, margin = 1)

# Check that every donor's proportions add up to 1
stopifnot(all(abs(rowSums(donor.level2.props) - 1) < 1e-8))

# The same table as a data frame, with donor_id as the first column
donor.level2.props.df <- as.data.frame.matrix(donor.level2.props) %>%
  rownames_to_column("donor_id")

# The table is large, so print its size and the first few rows
cat(sprintf("\n%d donors x %d level2 subtypes\n", nrow(donor.level2.props), ncol(donor.level2.props)))
catObj(head(round(donor.level2.props, 3)))

# Record the package versions in the log
sessionInfo()
