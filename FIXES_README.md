# FedUCA Code Fixes — Summary

## Critical bugs corrected

| File | Bug | Fix |
|------|-----|-----|
| `06_calibration_FIXED.ipynb` | `find_T` minimised ECE, not NLL | Now minimises NLL via golden-section search; unimodality verified on 200-point grid |
| `06_calibration_FIXED.ipynb` | `melanoma_sensitivity` used `max` class prob | Now uses melanoma-class probability `probs[:, MELANOMA]` |
| `05_fedproto_ham10000_FIXED.ipynb` | Proto loss computed from fixed embeddings (zero gradient) | Loss now computed via head logits → valid gradient path |
| `05_fedproto_ham10000_FIXED.ipynb` | `fedproto_aggregate` denominator used total sizes | Now uses per-class sample counts `n_k^c` |
| `10_isic2019_fedproto_FIXED.ipynb` | Same FedProto bugs as above | Same fixes applied to 8-class ISIC version |
| `09_isic2019_fedavg_pga_FIXED.ipynb` | `compute_prototypes` / `prototype_guided_aggregation` undefined | Both functions defined with 8-class support |
| `09_isic2019_fedavg_pga_FIXED.ipynb` | `lesion_id = image` (image-level split) | Loads real lesion_id from metadata CSV; HAM10000 overlaps removed |
| `03b_pga_seeds5to9_FIXED.ipynb` | CV evaluated fixed test set every fold | Now evaluates held-out fold; fixed test set retained as reference only |
| `03b_pga_seeds5to9_FIXED.ipynb` | Class weights reassigned inconsistently inside method loop | Weights computed once per fold, shared across both methods |
| `12_figures_tables_FIXED.ipynb` | Loaded `isic19_pga_all.json` (wrong filename) | Now loads `isic2019_results.json` (actual NB09 output) |
| `12_figures_tables_FIXED.ipynb` | Hard-coded seeds 5-9 (`pga_59 = [0.538, ...]`) | Loads from actual saved JSON files in results_extra_a2 |
| `12_figures_tables_FIXED.ipynb` | Communication axis still at 24.4 MB/round | Corrected to 3.581 MB (head only, FP32) |
| `12_figures_tables_FIXED.ipynb` | ISIC results loaded without method filter | Now filters by `method` key; FedProto loaded from corrected file |

## Re-run order after applying fixes

1. **05_fedproto_ham10000_FIXED** — rerun FedProto baseline (saves `fedproto_corrected_results.json`)
2. **09_isic2019_fedavg_pga_FIXED** — rerun ISIC FedAvg + PGA with deduplication and correct split
3. **10_isic2019_fedproto_FIXED** — rerun ISIC FedProto baseline
4. **06_calibration_FIXED** — rerun calibration with NLL objective and correct melanoma metrics
5. **03b_pga_seeds5to9_FIXED** — rerun seeds 5-9 and CV with correct fold evaluation
6. **12_figures_tables_FIXED** — regenerate all figures and tables from corrected results

## Reproducibility note

Add `from SEED_FIX_PATCH import set_all_seeds; set_all_seeds(seed)` at the
start of each seed loop to control model initialisation and DataLoader shuffling.
