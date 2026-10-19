# 01_atlas_exploration.R
#
# Exploration of the Human Breast Cell Atlas (HBCA, Reed et al.).
# Loads the atlas, ready for the analysis below.
#
# Needs:    hbca-global.h5ad, in the folder set as HBCA_DIR in local_paths.env
# Produces: nothing yet. It prints the atlas's size and main labels.
#
# Usage: Rscript scripts/01_atlas_exploration.R

library(SingleCellExperiment)
library(dplyr)
library(tidySingleCellExperiment)
library(tibble)
library(ggplot2)
library(patchwork)
library(scuttle)
library(scater)
library(anndataR)

# Header: load the config (from the repo root, where the script must be run), and set up
# the storage folder and this script's figure folder.
source("config.R")
output_dir <- OUTPUT_DIR
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

figure_outdir_cur = file.path(figure_outdir, '01')
dir.create(figure_outdir_cur, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("%s\n\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

# Make random steps (UMAP, clustering, subsampling) reproducible
set.seed(rng.seed)

# ------------------------------------------------------------------------

# ---------------------- Read the HBCA reference
# The atlas, read as a SingleCellExperiment
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
#   level2 is what the classifier predicts), cell_type, donor_id, risk_status,
#   brca_status and sample_type. Row names are Ensembl IDs (gene symbols are in
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

# ---------------------- Subtype make-up of each donor, one cell type at a time
# For one cell type: a donors x subtypes table of that cell type's subtypes, as a
# fraction of each donor's cells of that type (not all their cells), so every row
# sums to 1. Prints the table and each donor's cell count (proportions are less
# reliable for donors with few cells), and saves a stacked bar plot with one bar
# per donor to figures/01/<name>_subtypes_per_donor.png.
# Returns a list: counts (table), props (table), props.df (data frame with donor_id first).
get_donor_subtype_props <- function(subtypes, name) {
  is.subtype <- reed$level2 %in% subtypes

  counts <- table(
    donor_id = as.character(reed$donor_id[is.subtype]),
    level2 = factor(as.character(reed$level2[is.subtype]), levels = subtypes)
  )
  props <- prop.table(counts, margin = 1)

  # Check that every donor's proportions add up to 1
  stopifnot(all(abs(rowSums(props) - 1) < 1e-8))

  props.df <- as.data.frame.matrix(props) %>%
    rownames_to_column("donor_id")

  cat(sprintf("\n%s subtypes per donor:\n", name))
  catObj(round(props, 3))
  cat(sprintf("Number of %s cells per donor:\n", name))
  catObj(sort(rowSums(counts)))

  # Stacked bar plot: one bar per donor, split by subtype
  p <- as.data.frame(props) %>%
    rename(Proportion = Freq) %>%
    ggplot(aes(x = donor_id, y = Proportion, fill = level2)) +
    geom_col() +
    labs(
      x = "Donor",
      y = sprintf("Proportion of %s cells", name),
      fill = sprintf("%s subtype", name),
      title = sprintf("%s subtype make-up of each HBCA donor", name)
    ) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  ggsave(file.path(figure_outdir_cur, sprintf("%s_subtypes_per_donor.png", name)), plot = p, width = 16, height = 6, dpi = 200)

  list(counts = counts, props = props, props.df = props.df)
}

# LASP (luminal adaptive secretory precursor) subtypes
lasp.res <- get_donor_subtype_props(c("LASP1", "LASP2", "LASP3", "LASP4", "LASP5"), "LASP")
donor.lasp.props <- lasp.res$props
donor.lasp.props.df <- lasp.res$props.df

# LHS (luminal hormone sensing) subtypes
lhs.res <- get_donor_subtype_props(c("LHS1", "LHS2", "LHS3"), "LHS")
donor.lhs.props <- lhs.res$props
donor.lhs.props.df <- lhs.res$props.df

# BMYO (basal myoepithelial) subtypes
bmyo.res <- get_donor_subtype_props(c("BMYO1", "BMYO2"), "BMYO")
donor.bmyo.props <- bmyo.res$props
donor.bmyo.props.df <- bmyo.res$props.df

# Record the package versions in the log
sessionInfo()
