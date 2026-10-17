# 08_Level2_UMAP.py
#
# One UMAP of the atlas cells on only the genes that the tier-1 model (05) selected, coloured by cell
# state: every cell in the 10 epithelial states (LASP1-5, LHS1-3, BMYO1-2) in its own colour, plus a
# random sample of UMAP_OTHER_SAMPLE other cells in grey, as "Other". It shows how well those genes
# alone separate the states. Every run's UMAP uses the same cells (the sample is seeded), whichever
# run selected the genes, so the UMAPs of different runs differ only in their genes. Genes are
# centred and scaled to unit variance over the plotted cells. UMAP itself isn't seeded, so it runs
# on all the job's cores; its layout differs slightly between runs.
#
# Needs:    model[_max<G>genes].npz from 05, in <STORAGE_DIR>/<run>/reed_level2_classifier[_cap<N>]/
#           X.npz and prep.npz from 03 run on the full data (no cap), in
#           <STORAGE_DIR>/<run>/reed_level2_classifier/ (they hold every cell)
# Produces: umap[_max<G>genes].csv.gz in that run's folder (cell, state, UMAP1, UMAP2), and
#           figures/<run>/08/ (or .../08/cap<N>/): umap[_max<G>genes].png
#
# Runs in the base conda environment (it has umap-learn; the torch environment doesn't).
#
# Usage: python 08_Level2_UMAP.py [cellranger_run_name] [max_cells_per_class] [max_genes]
#        e.g. sbatch Run_08.sh 90 20000 40

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap

from level2_classifier_common import (KEEP_STATES, CLASSES, RANDOM_STATE, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_OTHER_SAMPLE,
                                      get_args, get_classifier_dir, get_figure_dir, load_prepared)

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

cellranger_run_name, max_cells_per_class = get_args()
max_genes = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ('', 'none') else None
suffix = '' if max_genes is None else f'_max{max_genes}genes'     # matches 05's output names
clf_dir = get_classifier_dir(cellranger_run_name, max_cells_per_class)
figure_outdir_cur = get_figure_dir(cellranger_run_name, max_cells_per_class, '08')
print(f"Cellranger run: {cellranger_run_name}, max cells per class: {max_cells_per_class}, max genes: {max_genes}\n")

# ---------------------- The genes selected by the tier-1 model (05)
model_file = os.path.join(clf_dir, f'model{suffix}.npz')
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found: run 05 with the same arguments first")
with np.load(model_file) as f:
    selected = np.linalg.norm(f['W'], axis=1) > 0
    genes = f['genes'][selected]
    model_genes = f['genes']
print(f"{len(genes)} genes selected by 05: {', '.join(genes)}\n")

# ---------------------- The atlas cells, on the selected genes
# The full-data prep (03 with no cap) holds every cell; its genes are in the same order as every run's
full_dir = get_classifier_dir(cellranger_run_name, None)
if not os.path.exists(os.path.join(full_dir, 'X.npz')):
    raise FileNotFoundError(f"{full_dir}/X.npz not found: run 03 on the full data (no cap) first")
X, prep = load_prepared(full_dir)
assert (model_genes == prep['genes']).all(), "gene order differs between this run and the full-data prep"
# Every epithelial cell, plus a seeded random sample of the "Other" cells
other = np.flatnonzero(prep['y'] == 'Other')
other = np.random.default_rng(RANDOM_STATE).choice(other, size=min(UMAP_OTHER_SAMPLE, len(other)), replace=False)
keep = np.sort(np.concatenate([np.flatnonzero(prep['y'] != 'Other'), other]))
X = X[keep][:, np.flatnonzero(selected)].toarray()
state, cells = prep['y'][keep], prep['cells'][keep]        # the 10 epithelial states, or "Other"
X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
print(pd.Series(state).value_counts().reindex(CLASSES).to_frame('cells').to_string())
log(f"{len(state):,} cells x {len(genes)} genes")

# ---------------------- UMAP
# No random_state, so umap-learn runs on all the job's cores (a seed would force one core)
embedding = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST).fit_transform(X)
log("UMAP done")
pd.DataFrame({'cell': cells, 'state': state, 'UMAP1': embedding[:, 0], 'UMAP2': embedding[:, 1]}) \
    .to_csv(os.path.join(clf_dir, f'umap{suffix}.csv.gz'), index=False)

# ---------------------- Plot, coloured by cell state
# One colour family per epithelial cell type (LASP blues, LHS reds, BMYO greens); Other is light grey,
# drawn first so it stays underneath. The epithelial cells are drawn in random order, so no state hides another.
colors = dict(zip(KEEP_STATES, list(plt.cm.Blues(np.linspace(0.35, 0.95, 5))) +
                               list(plt.cm.Reds(np.linspace(0.4, 0.9, 3))) +
                               list(plt.cm.Greens(np.linspace(0.5, 0.85, 2)))))
colors['Other'] = (0.85, 0.85, 0.85, 1.0)
is_other = state == 'Other'
epi = np.flatnonzero(~is_other)
epi = epi[np.random.default_rng(RANDOM_STATE).permutation(len(epi))]
point_size = 0.3
fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(embedding[is_other, 0], embedding[is_other, 1], s=point_size, linewidths=0,
           color=colors['Other'], rasterized=True)
ax.scatter(embedding[epi, 0], embedding[epi, 1], s=point_size, linewidths=0,
           c=[colors[s] for s in state[epi]], rasterized=True)
counts = pd.Series(state).value_counts()
ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=colors[s],
                          label=f'{s} ({counts.get(s, 0):,}' + (' sampled)' if s == 'Other' else ')'))
                   for s in CLASSES], title='Cell state', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
run_label = 'genes from the full run' if max_cells_per_class is None else f'genes from the {max_cells_per_class:,}-per-class run'
ax.set(xlabel='UMAP1', ylabel='UMAP2', xticks=[], yticks=[],
       title=f'Atlas cells on {len(genes)} tier-1 genes ({run_label})')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, f'umap{suffix}.png'), dpi=200)
log("done")
