# 02_Marker_Gene_Matrices.R
#
# Builds two reduced expression matrices from the Human Breast Cell Atlas (HBCA,
# Reed et al.), keeping only the published marker genes:
#   - one for the cell type markers (e.g. Fibroblast, Luminal hormone sensing)
#   - one for the cell state markers (the level2 subtypes, e.g. LASP1, LHS2, FB3)
# The values are the atlas's log-normalized expression (its X matrix).
#
# Needs:    hbca-global.h5ad, in the folder set as HBCA_DIR in local_paths.env
#           data/HBCA_Reed_Top_Marker_Genes.xlsx (sheets "celltype" and "state")
# Produces: nothing saved yet. The two matrices are celltype.marker.expr and
#           state.marker.expr (genes x cells, gene symbols as row names).
#
# Usage: Rscript 02_Marker_Gene_Matrices.R [cellranger_run_name]

library(SingleCellExperiment)
library(dplyr)
library(tibble)
library(Matrix)
library(anndataR)
library(readxl)

# Standard header: load the folder config, read the run name (default 90),
# and set up the storage folder and this script's figure folder.
source(file.path(repo_dir, "RNAseq_SingleCell", "Reed_Explore", "config.R"))
cellranger_run_name <- get_cellranger_run_name_cmd_arg()
output_dir <- file.path(OUTPUT_DIR, cellranger_run_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

figure_outdir_run = file.path(figure_outdir, cellranger_run_name)
figure_outdir_cur = file.path(figure_outdir_run, '02')
dir.create(figure_outdir_cur, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\nCellranger run: %s\n", cellranger_run_name))
cat(sprintf("%s\n\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

# Make random steps (UMAP, clustering, subsampling) reproducible
set.seed(rng.seed)

# ------------------------------------------------------------------------

# ---------------------- Read the HBCA reference
# Same as Cell_Type_Assignment/02_SingleR_Reed_transfer.R: the atlas, read as a SingleCellExperiment.
# Its only assay, "X", is log-normalized expression. (The file's raw counts are in a
# separate "raw" slot, which read_h5ad doesn't load.)
cat("Loading reference dataset...")
reed_dir <- get_local_path("HBCA_DIR")
reed <- read_h5ad(file.path(reed_dir, "hbca-global.h5ad"), as = "SingleCellExperiment")
cat("done\n")
cat(sprintf("%d genes x %d cells\n", nrow(reed), ncol(reed)))

# ---------------------- Read the marker gene tables
# Each sheet is wide: one column per cell type (or cell state), holding its top marker
# genes as gene symbols, ranked from row 1. Columns can have fewer than 100 genes.
# The first column is the rank, so it is dropped.
marker_file <- file.path(data_dir, "HBCA_Reed_Top_Marker_Genes.xlsx")

# Returns a named list: cell type (or state) -> its marker gene symbols, in rank order
read_marker_sheet <- function(sheet) {
  tbl <- read_excel(marker_file, sheet = sheet)
  tbl <- tbl[, -1]
  lapply(tbl, function(genes) {
    genes <- as.character(genes)
    unique(genes[!is.na(genes) & genes != ""])
  })
}

celltype.markers <- read_marker_sheet("celltype")
state.markers <- read_marker_sheet("state")

cat(sprintf("\n%d cell types and %d cell states with markers\n", length(celltype.markers), length(state.markers)))

# ---------------------- Reduce the expression matrix to the marker genes
# The atlas's row names are Ensembl IDs, and the markers are gene symbols, so genes are
# matched through rowData(reed)$feature_name. Markers that aren't in the atlas (mostly
# clone-based names like "AC093001.1") are listed in the log and left out.
# Returns a genes x cells matrix of log-normalized expression, with gene symbols as row names.
get_marker_matrix <- function(markers, name) {
  marker.genes <- unique(unlist(markers))
  atlas.symbols <- as.character(rowData(reed)$feature_name)

  keep <- which(atlas.symbols %in% marker.genes)
  missing <- setdiff(marker.genes, atlas.symbols)
  cat(sprintf("\n%s markers: %d unique genes, %d found in the atlas, %d missing\n",
              name, length(marker.genes), length(keep), length(missing)))
  catObj(missing)

  expr <- assay(reed, "X")[keep, , drop = FALSE]
  rownames(expr) <- atlas.symbols[keep]

  # Symbols are only safe as row names if no marker symbol matches two atlas genes
  stopifnot(!anyDuplicated(rownames(expr)))

  cat(sprintf("%s marker matrix: %d genes x %d cells\n", name, nrow(expr), ncol(expr)))
  expr
}

celltype.marker.expr <- get_marker_matrix(celltype.markers, "Cell type")
state.marker.expr <- get_marker_matrix(state.markers, "Cell state")

# ---------------------- Analysis
# Save figures to figure_outdir_cur, e.g.
#   ggsave(file.path(figure_outdir_cur, "my_plot.png"), plot = p, width = 8, height = 6, dpi = 200)
# Save objects that a later script needs to output_dir, e.g.
#   saveRDS(celltype.marker.expr, file.path(output_dir, "reed_celltype_marker_expr.rds"))



# Record the package versions in the log
sessionInfo()
