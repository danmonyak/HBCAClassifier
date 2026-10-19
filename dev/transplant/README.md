# Retired: TRANSPLANT (09, 10) and decision rules (11)

These scripts were part of the pipeline until 2026-10-19. They're kept for the record and **don't run as they are**: their settings were removed from `hbca_classifier/settings.py`, and they import `hbca_classifier.tier2_models`, which now lives here.

- `09_transplant.py`, `10_transplant_summary.py`: retrained the tier-2 models on the full data with the genes from a capped run, because the full-data run picked worse genes. Class-weighted gene scaling fixed that, so they're no longer needed.
- `11_transplant_thresholds.py`: tried other decision rules on the TRANSPLANT networks (post-hoc reweighting, probability thresholds with an "Unclassified" call, a family fallback).
- `tier2_models.py`: the network training and KNN votes as functions, used by `09` and `11`.

All of them used the old cell-level split. Their outputs are in `<STORAGE_DIR>/old_cell_split/reed_level2_transplant/` and `figures/old_cell_split/09` to `11`.
