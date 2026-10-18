# 11_transplant_thresholds.py
#
# TRANSPLANT, step 3 (optional): other decision rules for the TRANSPLANT networks from 09, one per
# trained gamma (class weights (1/class frequency)^gamma in the loss). No retraining: it reloads the
# saved networks and recomputes their class probabilities on 09's validation and test cells.
#
# Part A, post-hoc reweighting: does training with class weights help, or does it only shift the
#   decision rule? Each network's probabilities are multiplied by (1/class frequency)^gamma_post,
#   renormalized, and the top class is the prediction. gamma_post (POSTHOC_GAMMAS; negative values undo
#   a weighted network's weights) is chosen per network by validation balanced accuracy, and the test
#   set is scored once. If the unweighted network (gamma 0) plus its best gamma_post matches the
#   weighted ones, the weights only shift the decision rule.
# Part B, thresholds: a cell is called "Other" if P(Other) >= other_threshold, or one of the 10 states
#   if that state's probability >= state_threshold. If several classes pass, the most probable one
#   wins (this can only happen if two thresholds sum to 1 or less). Otherwise it's "Unclassified".
#   Scored on the test set (and validation) for the thresholds given, and on validation only for a grid
#   of threshold pairs (THRESHOLD_GRID; exploratory: choose thresholds from it, not from the test set).
#   Precision = correct / cells called that class. Recall = correct / all true cells of that class,
#   so unclassified cells count as misses. Macro recall is then balanced accuracy with unclassified
#   cells counted as wrong.
# Part C, family fallback: as Part B, but a cell left unclassified is called LASP, LHS or BMYO if that
#   family's summed probability (e.g. LASP1 + ... + LASP5) is at least the state threshold. A family call
#   counts as right if the cell's true state is in that family.
#
# The unweighted network (trained gamma 0) is the tier-2 model; the others are kept for comparison.
# The baseline and KNN aren't included: 09 didn't save their validation scores.
#
# Needs:    from 09, in <STORAGE_DIR>/reed_level2_transplant/: nn_gamma<g>.pt and gamma<g>.npz for every
#           TRANSPLANT_GAMMAS value, and test_cells.csv.gz; 09's inputs (the full-data X.npz and prep.npz,
#           and the source run's prep.npz and model_max<G>genes.npz), to rebuild its cells
# Produces: in <STORAGE_DIR>/reed_level2_transplant/thresholds/:
#           posthoc_validation.csv   validation balanced accuracy and macro F1 per trained gamma x gamma_post
#           posthoc_test.csv         per trained gamma: test scores at gamma_post 0 and at its best gamma_post
#           threshold_report<sfx>.csv  per trained gamma, split and class: cells, cells called, correct,
#                                      precision, recall, fraction unclassified
#           threshold_summary<sfx>.csv per trained gamma and split: cells unclassified, accuracy of the
#                                      classified cells, macro precision and recall
#           family_report<sfx>.csv     per trained gamma, split and true class: fraction called the state,
#                                      its family, something wrong, or left unclassified
#           family_summary<sfx>.csv    per trained gamma and split: fractions unclassified and called at
#                                      state / family level, accuracy of the calls, each family call's precision
#           threshold_grid.csv       validation macro precision, macro recall and fraction unclassified per
#                                    trained gamma x threshold pair
#           (<sfx> = _other<a>_state<b>, the thresholds given)
#           in figures/11/: posthoc_reweighting.png, threshold_grid_gamma<g>.png and
#           threshold_confusion_gamma<g><sfx>.png and
#           family_confusion_gamma<g><sfx>.png (test set)
#
# Usage: sbatch slurm/Run_11.sh [other_threshold] [state_threshold]     (defaults 0.6 and 0.8)

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from hbca_classifier.settings import (OUTPUT_DIR, CLASSES, RANDOM_STATE, NN_VAL_SIZE, NN_DROPOUT,
                                      TRANSPLANT_SOURCE_CAP, TRANSPLANT_SOURCE_MAX_GENES, TRANSPLANT_GAMMAS,
                                      POSTHOC_GAMMAS, THRESHOLD_GRID, KEEP_STATES, FAMILIES, get_classifier_dir, get_figure_dir,
                                      load_prepared, set_threads)
from hbca_classifier.tier2_models import network_proba

t_start = time.time()
log = lambda s: print(f"[{time.time() - t_start:7.1f} s] {s}", flush=True)

other_t = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '' else 0.6
state_t = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '' else 0.8
sfx = f'_other{other_t:g}_state{state_t:g}'
transplant_dir = os.path.join(OUTPUT_DIR, 'reed_level2_transplant')
out_dir = os.path.join(transplant_dir, 'thresholds')
os.makedirs(out_dir, exist_ok=True)
figure_outdir_cur = get_figure_dir(None, '11')
set_threads()
print(f"Thresholds: Other {other_t:g}, states {state_t:g}\n")

# ---------------------- 09's cells (same code as 09, so the same split)
src_dir = get_classifier_dir(TRANSPLANT_SOURCE_CAP)
with np.load(os.path.join(src_dir, f'model_max{TRANSPLANT_SOURCE_MAX_GENES}genes.npz')) as f:
    selected = np.linalg.norm(f['W'], axis=1) > 0
    src_genes = f['genes']
with np.load(os.path.join(src_dir, 'prep.npz')) as f:
    src_train_cells = set(f['cells'][f['idx_train']])
X, prep = load_prepared(get_classifier_dir(None))
assert (prep['genes'] == src_genes).all(), "gene order differs between the source run and the full-data prep"
X = X[:, np.flatnonzero(selected)].toarray()
y, cells = prep['y'], prep['cells']
idx_fit, idx_val = train_test_split(prep['idx_train'], test_size=NN_VAL_SIZE,
                                    stratify=y[prep['idx_train']], random_state=RANDOM_STATE)
idx_test = np.array([i for i in prep['idx_test'] if cells[i] not in src_train_cells])
y_fit, y_val, y_test = y[idx_fit], y[idx_val], y[idx_test]
saved_test = pd.read_csv(os.path.join(transplant_dir, 'test_cells.csv.gz'))
assert (saved_test['cell'].to_numpy() == cells[idx_test]).all(), "test cells differ from 09's test_cells.csv.gz"
log(f"validation: {len(idx_val):,} cells, test: {len(idx_test):,}")

# Training-set class frequencies: the prior that post-hoc reweighting divides out
freq = pd.Series(y_fit).value_counts(normalize=True).reindex(CLASSES).to_numpy()


# ---------------------- Reload each network and recompute its probabilities
def load_network(path):
    """Rebuild 09's network (the layout of tier2_models.train_network) and load its weights."""
    saved = torch.load(path, weights_only=False)
    layers, n_in = [], len(saved['genes'])
    for n_out in saved['hidden']:
        layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(NN_DROPOUT)]
        n_in = n_out
    layers.append(nn.Linear(n_in, len(CLASSES)))
    net = nn.Sequential(*layers)
    net.load_state_dict(saved['state_dict'])
    net.eval()
    return net, saved


P = {}     # trained gamma -> {'val': ..., 'test': ...}
for g in TRANSPLANT_GAMMAS:
    net, saved = load_network(os.path.join(transplant_dir, f'nn_gamma{g}.pt'))
    assert list(saved['genes']) == list(src_genes[selected]) and list(saved['classes']) == list(CLASSES)
    scale = lambda idx: (X[idx] - saved['scaler_mean']) / saved['scaler_scale']
    P[g] = {'val': network_proba(net, scale(idx_val)), 'test': network_proba(net, scale(idx_test))}
    # Check: the recomputed probabilities must match what 09 saved
    with np.load(os.path.join(transplant_dir, f'gamma{g}.npz')) as f:
        diff = np.abs(P[g]['test'] - f['nn_P_test']).max()
        val_bacc_09 = float(f['nn_val_bacc'])
    val_bacc = balanced_accuracy_score(y_val, CLASSES[P[g]['val'].argmax(axis=1)])
    if diff > 1e-5 or abs(val_bacc - val_bacc_09) > 1e-6:
        raise RuntimeError(f"gamma {g}: recomputed network differs from 09's (test probabilities up to {diff:.2g} "
                           f"apart; validation balanced accuracy {val_bacc:.6f} vs {val_bacc_09:.6f})")
    log(f"network gamma {g}: reloaded, matches 09 (test probabilities within {diff:.1g})")


def scores(y_true, pred):
    return dict(balanced_accuracy=balanced_accuracy_score(y_true, pred),
                f1_macro=f1_score(y_true, pred, labels=CLASSES, average='macro', zero_division=0),
                accuracy=accuracy_score(y_true, pred))


def reweight(P_, gamma_post):
    return CLASSES[(P_ * (1 / freq) ** gamma_post).argmax(axis=1)]


# ---------------------- Part A: post-hoc reweighting
rows = [dict(gamma_trained=g, gamma_post=gp, **scores(y_val, reweight(P[g]['val'], gp)))
        for g in TRANSPLANT_GAMMAS for gp in POSTHOC_GAMMAS]
posthoc_val = pd.DataFrame(rows)
posthoc_val.to_csv(os.path.join(out_dir, 'posthoc_validation.csv'), index=False)

rows = []
for g in TRANSPLANT_GAMMAS:
    v = posthoc_val[posthoc_val['gamma_trained'] == g]
    best_gp = float(v.loc[v['balanced_accuracy'].idxmax(), 'gamma_post'])
    for gp, label in [(0.0, 'none (top class)'), (best_gp, 'best on validation')]:
        rows.append(dict(gamma_trained=g, gamma_post=gp, rule=label,
                         val_balanced_accuracy=float(v.loc[np.isclose(v['gamma_post'], gp), 'balanced_accuracy'].iloc[0]),
                         **{f'test_{k}': s for k, s in scores(y_test, reweight(P[g]['test'], gp)).items()}))
posthoc_test = pd.DataFrame(rows)
posthoc_test.to_csv(os.path.join(out_dir, 'posthoc_test.csv'), index=False)
print("\nPart A, post-hoc reweighting (gamma_post chosen by validation balanced accuracy):")
print(posthoc_test.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for g in TRANSPLANT_GAMMAS:
    v = posthoc_val[posthoc_val['gamma_trained'] == g]
    for ax, col in zip(axes, ['balanced_accuracy', 'f1_macro']):
        ax.plot(v['gamma_post'], v[col], marker='.', label=f'trained gamma {g}')
for ax, lab in zip(axes, ['Validation balanced accuracy', 'Validation macro F1']):
    ax.axvline(0, color='grey', lw=0.8, ls=':')
    ax.set(xlabel='gamma_post (probabilities x (1 / class frequency)^gamma_post)', ylabel=lab)
axes[0].legend()
fig.suptitle('TRANSPLANT networks: post-hoc reweighting')
fig.tight_layout()
fig.savefig(os.path.join(figure_outdir_cur, 'posthoc_reweighting.png'), dpi=200)
plt.close(fig)


# ---------------------- Part B: thresholds
def threshold_predict(P_, o_t, s_t):
    """Class per cell: the most probable class among those over their threshold, else 'Unclassified'."""
    thr = np.where(CLASSES == 'Other', o_t, s_t)
    passes = P_ >= thr
    pred = CLASSES[np.where(passes, P_, -np.inf).argmax(axis=1)].astype(object)
    pred[~passes.any(axis=1)] = 'Unclassified'
    return pred


def threshold_scores(y_true, pred):
    """Per-class table and overall summary for a prediction that may include 'Unclassified'."""
    rows = []
    for c in CLASSES:
        is_c = y_true == c
        n_pred, n_correct = (pred == c).sum(), (is_c & (pred == c)).sum()
        rows.append(dict(state=c, n_true=is_c.sum(), n_called=n_pred, n_correct=n_correct,
                         precision=n_correct / n_pred if n_pred else np.nan,
                         recall=n_correct / is_c.sum(),
                         frac_unclassified=(pred[is_c] == 'Unclassified').mean()))
    report = pd.DataFrame(rows)
    classified = pred != 'Unclassified'
    summary = dict(n_cells=len(y_true), n_unclassified=(~classified).sum(), frac_unclassified=(~classified).mean(),
                   accuracy_classified=(pred[classified] == y_true[classified]).mean() if classified.any() else np.nan,
                   macro_precision=report['precision'].mean(), macro_recall=report['recall'].mean())
    return report, summary


reports, summaries = [], []
for g in TRANSPLANT_GAMMAS:
    for split, key, y_true in [('test', 'test', y_test), ('validation', 'val', y_val)]:
        pred = threshold_predict(P[g][key], other_t, state_t)
        report, summary = threshold_scores(y_true, pred)
        reports.append(report.assign(gamma_trained=g, split=split))
        summaries.append(dict(gamma_trained=g, split=split, **summary))
        if split == 'test':
            print(f"\nPart B, network trained with gamma {g}, test set, thresholds Other {other_t:g} / states {state_t:g}: "
                  f"{summary['n_unclassified']:,} of {summary['n_cells']:,} cells unclassified "
                  f"({100 * summary['frac_unclassified']:.1f}%), accuracy of the rest {summary['accuracy_classified']:.4f}")
            print(report.round(3).to_string(index=False))
            # Confusion matrix with an Unclassified column, as a fraction of each true class
            cols = list(CLASSES) + ['Unclassified']
            cm = pd.crosstab(pd.Series(y_true, name='True'), pd.Series(pred, name='Predicted'), normalize='index') \
                .reindex(index=CLASSES, columns=cols, fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 7.5))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                        cbar_kws={'label': 'Fraction of true class'}, ax=ax)
            ax.set(title=f'TRANSPLANT network (trained gamma {g}), test set\n'
                         f'thresholds: Other {other_t:g}, states {state_t:g}; '
                         f"{100 * summary['frac_unclassified']:.1f}% unclassified")
            fig.tight_layout()
            fig.savefig(os.path.join(figure_outdir_cur, f'threshold_confusion_gamma{g}{sfx}.png'), dpi=200)
            plt.close(fig)
pd.concat(reports)[['gamma_trained', 'split'] + list(reports[0].columns[:-2])] \
    .to_csv(os.path.join(out_dir, f'threshold_report{sfx}.csv'), index=False)
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(os.path.join(out_dir, f'threshold_summary{sfx}.csv'), index=False)
print("\nPart B summary:")
print(summary_df.round(4).to_string(index=False))


# ---------------------- Part C: family fallback
def family_predict(P_, o_t, s_t):
    """Part B's calls, but a cell left unclassified is called LASP, LHS or BMYO if that family's summed
    probability is at least the state threshold."""
    pred = threshold_predict(P_, o_t, s_t)
    F = np.column_stack([P_[:, np.isin(CLASSES, members)].sum(axis=1) for members in FAMILIES.values()])
    fallback = (pred == 'Unclassified') & (F.max(axis=1) >= s_t)
    pred[fallback] = np.array(list(FAMILIES))[F.argmax(axis=1)[fallback]]
    return pred


family_of = {s: fam for fam, members in FAMILIES.items() for s in members}
fam_reports, fam_summaries = [], []
for g in TRANSPLANT_GAMMAS:
    for split, key, y_true in [('test', 'test', y_test), ('validation', 'val', y_val)]:
        pred = family_predict(P[g][key], other_t, state_t)
        true_family = np.array([family_of.get(s, 'Other') for s in y_true])
        classified = pred != 'Unclassified'
        # A call is right if it's the true state, or the true state's family
        right = (pred == y_true) | (pred == true_family)
        rows = []
        for c in CLASSES:
            is_c = y_true == c
            rows.append(dict(state=c, n_true=is_c.sum(), frac_state=(pred[is_c] == c).mean(),
                             frac_family=(pred[is_c] == family_of.get(c, '-')).mean(),
                             frac_wrong=(classified & ~right)[is_c].mean(),
                             frac_unclassified=(~classified[is_c]).mean()))
        report = pd.DataFrame(rows)
        fam_reports.append(report.assign(gamma_trained=g, split=split))
        epithelial = y_true != 'Other'
        summary = dict(gamma_trained=g, split=split, n_cells=len(y_true),
                       frac_unclassified=(~classified).mean(),
                       frac_epithelial_unclassified=(~classified[epithelial]).mean(),
                       frac_epithelial_state_call=np.isin(pred[epithelial], KEEP_STATES).mean(),
                       frac_epithelial_family_call=np.isin(pred[epithelial], list(FAMILIES)).mean(),
                       accuracy_classified=right[classified].mean())
        for fam in FAMILIES:     # precision of each family call: its cells truly in that family
            called = pred == fam
            summary[f'n_called_{fam}'] = called.sum()
            summary[f'precision_{fam}'] = (true_family[called] == fam).mean() if called.any() else np.nan
        fam_summaries.append(summary)
        if split == 'test':
            print(f"\nPart C, network trained with gamma {g}, test set, family fallback at {state_t:g}: "
                  f"{100 * summary['frac_unclassified']:.1f}% of cells unclassified "
                  f"({100 * summary['frac_epithelial_unclassified']:.1f}% of epithelial cells), "
                  f"accuracy of the rest {summary['accuracy_classified']:.4f} (a family call counts if it's the right family)")
            print(report.round(3).to_string(index=False))
            print("Family calls: " + ", ".join(f"{fam} {summary[f'n_called_{fam}']:,} (precision "
                                               f"{summary[f'precision_{fam}']:.3f})" for fam in FAMILIES))
            cols = list(CLASSES) + list(FAMILIES) + ['Unclassified']
            cm = pd.crosstab(pd.Series(y_true, name='True'), pd.Series(pred, name='Predicted'), normalize='index') \
                .reindex(index=CLASSES, columns=cols, fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 7.5))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                        cbar_kws={'label': 'Fraction of true class'}, ax=ax)
            ax.set(title=f'TRANSPLANT network (trained gamma {g}), test set, family fallback\n'
                         f'thresholds: Other {other_t:g}, states and families {state_t:g}; '
                         f"{100 * summary['frac_unclassified']:.1f}% unclassified")
            fig.tight_layout()
            fig.savefig(os.path.join(figure_outdir_cur, f'family_confusion_gamma{g}{sfx}.png'), dpi=200)
            plt.close(fig)
pd.concat(fam_reports)[['gamma_trained', 'split'] + list(fam_reports[0].columns[:-2])] \
    .to_csv(os.path.join(out_dir, f'family_report{sfx}.csv'), index=False)
fam_summary_df = pd.DataFrame(fam_summaries)
fam_summary_df.to_csv(os.path.join(out_dir, f'family_summary{sfx}.csv'), index=False)
print("\nPart C summary:")
print(fam_summary_df.round(4).to_string(index=False))

# Exploratory grid of threshold pairs, on validation only
rows = []
for g in TRANSPLANT_GAMMAS:
    for o in THRESHOLD_GRID:
        for s in THRESHOLD_GRID:
            _, summary = threshold_scores(y_val, threshold_predict(P[g]['val'], o, s))
            rows.append(dict(gamma_trained=g, other_threshold=o, state_threshold=s, **summary))
grid = pd.DataFrame(rows)
grid.to_csv(os.path.join(out_dir, 'threshold_grid.csv'), index=False)
for g in TRANSPLANT_GAMMAS:
    v = grid[grid['gamma_trained'] == g]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col, lab in zip(axes, ['macro_precision', 'macro_recall', 'frac_unclassified'],
                            ['Macro precision', 'Macro recall (unclassified = miss)', 'Fraction unclassified']):
        sns.heatmap(v.pivot(index='other_threshold', columns='state_threshold', values=col),
                    annot=True, fmt='.2f', cmap='viridis', annot_kws={'size': 6}, ax=ax)
        ax.invert_yaxis()
        ax.set(title=lab, xlabel='State threshold', ylabel='Other threshold')
    fig.suptitle(f'TRANSPLANT network (trained gamma {g}): threshold pairs, validation set (exploratory)')
    fig.tight_layout()
    fig.savefig(os.path.join(figure_outdir_cur, f'threshold_grid_gamma{g}.png'), dpi=200)
    plt.close(fig)
log("done")
