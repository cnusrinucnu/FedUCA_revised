# FedUCA — Reproducibility Package

**Title:** FedUCA: Prototype-Guided Federated Learning with Uncertainty
Calibration for Data-Local Skin Lesion Classification

**Author:** Srinivas Arukonda, SRM University-AP
**Email:** srinivas.a@srmap.edu.in
**ORCID:** 0000-0002-8974-844X
**Platform:** Kaggle (all experiments run on Kaggle cloud — free)
**Code repository:** https://github.com/cnusrinucnu/FedUCA-/

---

## Important: Use the Fixed Notebooks

Six notebooks in the original package contained implementation bugs that
affect experimental validity. **Replace the originals with the `*_FIXED`
versions before running any experiment.** The bugs and their corrections
are documented in `FIXES_README.md`.

| Original | Replace with | Bug fixed |
|----------|-------------|-----------|
| `05_fedproto_ham10000.ipynb` | `05_fedproto_ham10000_FIXED.ipynb` | FedProto had zero gradient path; wrong prototype denominator |
| `06_calibration.ipynb` | `06_calibration_FIXED.ipynb` | `find_T` minimised ECE not NLL; melanoma metric used wrong class probability |
| `09_isic2019_fedavg_pga.ipynb` | `09_isic2019_fedavg_pga_FIXED.ipynb` | Missing PGA functions; image-level split; no HAM10000 deduplication |
| `10_isic2019_fedproto.ipynb` | `10_isic2019_fedproto_FIXED.ipynb` | Same FedProto bugs as notebook 05 |
| `03b_pga_seeds5to9.ipynb` | `03b_pga_seeds5to9_FIXED.ipynb` | CV evaluated fixed test set every fold; inconsistent class weights |
| `12_figures_tables.ipynb` | `12_figures_tables_FIXED.ipynb` | Wrong ISIC filename; hard-coded seed values; 24.4 MB axis error |

Notebooks 01, 02, 03, 04, 07, 08, 11 are unchanged and can be used as-is.

---

## Overview

This package contains all Jupyter notebooks required to reproduce every
experiment in the paper. All experiments run on Kaggle using free GPU
(NVIDIA T4) and free CPU sessions. No local GPU is required.

**Total GPU time to reproduce all experiments: ~15 hours**
**All datasets are free and publicly available on Kaggle**

Privacy note: FedUCA achieves data locality — no raw patient images leave
any client — but does not implement formal differential privacy.

---

## Step 0 — One-Time Setup

1. Create a free account at kaggle.com
2. Go to Settings → Phone verification (required for GPU access)
3. Upload all notebooks to Kaggle

For reproducibility, add the following to the top of every seed loop
(a `set_all_seeds` utility is provided in `SEED_FIX_PATCH.py`):

```python
from SEED_FIX_PATCH import set_all_seeds
set_all_seeds(seed)
```

This sets `random`, `numpy`, `torch`, and `torch.cuda` seeds before each
model initialisation to ensure paired statistical comparisons are valid.

---

## Datasets — What To Add and Where

| Dataset | Kaggle URL | Size | Used In |
|---------|-----------|------|---------|
| HAM10000 images | kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 | 1.8 GB | Notebook 1 |
| ISIC 2019 images + metadata | kaggle.com/datasets/andrewmvd/isic-2019 | 10 GB | Notebook 9 |

All other datasets are **outputs you save yourself** as you run each notebook.

---

## Execution Order

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 3b →  Phase 4
Embedding   FedAvg      PGA         PGA seeds    FedProx
(required   (Phase 1)   (Phase 2)   5–9          (Phase 1)
 first)

Phase 5* →  Phase 6  →  Phase 7  →  Phase 8*  →  Phase 9*
FedProto*   Calibration* Scalability 5-fold CV    ISIC FedAvg
(Phase 1)   (Phase 2)   (Phase 1)   (Phase 1)    + PGA*

Phase 10* →  Phase 11  →  Phase 12*
ISIC         Centralised  Figures +
FedProto*    Baseline     Tables*

* = use the FIXED notebook version
```

---

## Notebook 1 — HAM10000 Embedding Extraction

**File:** `01_embedding_cache_ham10000.ipynb`
**Datasets to add:** `kmader/skin-cancer-mnist-ham10000`
**Accelerator:** GPU T4
**Runtime:** ~30 min
**Output — save as dataset:** `ham10000-vit-embeddings-orig`

Extracts frozen ViT-B/16 SWAG class-token embeddings (768-dim) for all
HAM10000 images. Applies lesion-disjoint GroupShuffleSplit (seed=42):
Train 7,984 / Val 1,005 / Test 1,026. No lesion overlap between splits.

---

## Notebook 2 — FedAvg Baseline (Head Ablation)

**File:** `02_fedavg_head_ablation.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~3 hours
**Output — save as dataset:** `feduca-phase2-results`

Runs FedAvg with 4 head architectures × 3 Dirichlet alpha levels × 5 seeds.
K=4 clients, T=60 rounds, E=5 local epochs, Adam lr=5×10⁻³.
Produces Tables 2 and 3 of the paper.

Expected key result:
- Bottleneck N/4, α=1000: macro-F1 = 0.643 ± 0.010
- Bottleneck N/4, α=0.1:  macro-F1 = 0.363 ± 0.116

Validation macro-F1 for model selection is computed by pooling per-class
confusion-matrix counts (TP, FP, FN) reported by each client. This is exact
and avoids the inaccuracy of averaging local macro-F1 scores.

---

## Notebook 3 — Prototype-Guided Aggregation (PGA)

**File:** `03_pga.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig` + `feduca-phase2-results`
**Accelerator:** GPU T4
**Runtime:** ~90 min (seeds 0–4)
**Output — save as dataset:** `feduca-phase3-results`

Runs PGA with Bottleneck N/4, 3 alpha levels, seeds 0–4. For the primary
α=0.1 comparison, run seeds 5–9 separately (Notebook 3b).

Expected key result:
- PGA, α=0.1, 10 seeds: macro-F1 = 0.483 ± 0.090, Δ=+0.074 vs FedAvg, p=0.001

PGA uses the active client set K_c = {k : n_k^c > 0} to exclude clients
with zero samples for class c from prototype computation and aggregation.
Both the final-layer weight matrix and bias vector are aggregated row-by-row
using cosine-distance-based weights.

---

## Notebook 3b — PGA Extended Seeds (Seeds 5–9)

**File:** `03b_pga_seeds5to9_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~45 min
**Output — save as dataset:** `feduca-extra-a2-results`

Runs PGA/FedProx/FedAvg seeds 5–9 at α=0.1 for the 10-seed primary
comparison. Also runs 5-fold lesion-grouped cross-validation.

**CV protocol:** Each fold uses the held-out fold as the test estimate
(out-of-fold score). The fixed primary test set is recorded separately
under `fixed_test` but is not used as the CV estimate.

---

## Notebook 4 — FedProx Baseline

**File:** `04_fedprox.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~90 min
**Output — save as dataset:** `feduca-extra-a-results`

Runs FedProx (μ=0.01) with Bottleneck N/4, 3 alpha × 5 seeds.

Expected key result:
- FedProx, α=0.1: macro-F1 = 0.414 ± 0.109, p=0.874 vs FedAvg (ns)

---

## Notebook 5 — FedProto Baseline (HAM10000)

**File:** `05_fedproto_ham10000_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~90 min
**Output — save as dataset:** `feduca-fedproto-results`
**Output filename:** `fedproto_corrected_results.json`

Runs FedProto (Tan et al., AAAI 2022) with Bottleneck N/4, 3 alpha × 5 seeds.

**Implementation note:** The prototype regularisation loss is computed from
the head's logit outputs for class-c samples (not from fixed cached embeddings).
This ensures that the loss contributes valid gradients to the head's parameters.
Prototype aggregation uses per-class sample counts (n_k^c) in the denominator.

Expected key result:
- FedProto, α=0.1: macro-F1 ≈ 0.365 (worse than FedAvg — this finding is
  the paper's key point; exact value may differ from original due to bug fix)

---

## Notebook 6 — Federated Temperature Scaling (Calibration)

**File:** `06_calibration_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** `ham10000-vit-embeddings-orig` + `feduca-phase2-results`
**Accelerator:** CPU (no GPU needed)
**Runtime:** ~30 min
**Output — save as dataset:** `feduca-phase5-results`

Computes federated temperature scaling using NLL minimisation (not ECE).
Each client independently finds T_k* = argmin_T NLL(h_θ, D_k^val) via
golden-section search over T ∈ [0.5, 5.0] after verifying unimodality on a
200-point dense grid. The server aggregates temperatures proportionally by
validation set size.

**Melanoma metrics:** Computed using the melanoma-class probability
(probs[:, MELANOMA]), not the maximum predicted class probability.

Calibration metrics reported: ECE (top-label, 15 equal-width bins, bins with
<5 samples excluded), Brier score, NLL.

Expected key result:
- ECE: 0.1660 → 0.0663 (−60.0%, p<0.001) at α=1000

---

## Notebook 7 — Scalability (K=10, K=20)

**File:** `07_scalability.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~2 hours
**Output — save as dataset:** `feduca-extra-a3-results`

Runs FedAvg and PGA with K=10 and K=20 clients, 3 alpha × 5 seeds.

**Note on K=20 results:** A safe-partitioning procedure ensures minimum client
sizes by redistributing samples. This reduces effective heterogeneity at K=20,
explaining the near-zero PGA advantage (Δ=+0.002, ns) at that scale.
K=10 uses the same distribution as K=4 results (from extra-a dataset);
K=20 uses the modified partition. These are not directly comparable as a
controlled scalability experiment; interpret K=10 and K=20 results separately.

---

## Notebook 8 — 5-Fold Cross-Validation

**File:** `08_cross_validation.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** GPU T4
**Runtime:** ~90 min
**Output — save as dataset:** `feduca-cv-results`

Runs 5-fold lesion-grouped GroupKFold, FedAvg and PGA, α=1000.

**CV protocol:** Each fold evaluates the model on the held-out fold (not the
fixed primary test set). The result labelled `test` in the output JSON is the
out-of-fold macro-F1. The fixed primary test set result is stored under
`fixed_test` for reference only and is not part of the CV estimate.

---

## Notebook 9 — ISIC 2019 Embeddings + FedAvg + PGA

**File:** `09_isic2019_fedavg_pga_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** `andrewmvd/isic-2019` (images + both CSV files)
**Accelerator:** GPU T4
**Runtime:** ~3 hours (embedding extraction ~1h + experiments ~2h)
**Output — save as dataset:** `feduca-isic-results`
**Output filename:** `isic2019_results.json`

**Deduplication protocol:** ISIC 2019 incorporates HAM10000-derived images.
This notebook loads `ISIC_2019_Training_Metadata.csv`, identifies HAM10000
lesion IDs (format HAM_XXXXXXX) in the metadata, and removes those images
before splitting. This yields a deduplicated set for independent evaluation.
Lesion IDs are taken from the metadata `patient_id` or `lesion_id` column;
images without a metadata match use their image ID as the group key.

**Split:** The GroupShuffleSplit uses real lesion identifiers (from metadata),
not image IDs, so the split is lesion-disjoint, not merely image-disjoint.

Expected key result:
- PGA, α=0.1: macro-F1 = 0.524 ± 0.058 (not statistically significant at
  5 seeds, p=0.153; additional seeds required to establish robustness)

---

## Notebook 10 — ISIC 2019 FedProto

**File:** `10_isic2019_fedproto_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** `andrewmvd/isic-2019` (embeddings from Notebook 9)
**Accelerator:** GPU T4
**Runtime:** ~90 min
**Output — save as dataset:** `feduca-isic-fedproto-results`
**Output filename:** `isic_fedproto_corrected_results.json`

Runs the corrected FedProto on ISIC 2019. Same implementation fix as
Notebook 5 (logit-based prototype loss, class-specific count denominator).

---

## Notebook 11 — Centralised Baseline

**File:** `11_centralised_baseline.ipynb`
**Datasets to add:** `ham10000-vit-embeddings-orig`
**Accelerator:** CPU (no GPU needed)
**Runtime:** ~10 min
**Output:** Prints results only

Trains Bottleneck N/4 on all training data with no federation.
Establishes the upper bound for comparison.

Expected key result:
- Centralised: macro-F1 = 0.639 ± 0.010

---

## Notebook 12 — All Figures and Tables

**File:** `12_figures_tables_FIXED.ipynb` ← **use FIXED version**
**Datasets to add:** All result datasets from Notebooks 2–11
**Accelerator:** CPU
**Runtime:** ~5 min

Generates all publication-quality figures (PDF + PNG, 300 DPI) and LaTeX
tables from the saved result JSON files.

**Key corrections in the fixed version:**
- Loads ISIC PGA results from `isic2019_results.json` (notebook 9 output)
  and ISIC FedProto from `isic_fedproto_corrected_results.json`
- Loads seeds 5–9 from actual JSON files (not hard-coded values)
- Communication axis corrected to 3.581 MB/round per client (head only, FP32)
- Results loaded once per experiment, filtered by `method` key

---

## Expected Final Results

| Method | α=1000 | α=1 | α=0.1 |
|--------|--------|-----|--------|
| Centralised (upper bound) | 0.639±0.010 | — | — |
| FedAvg | 0.643±0.010 | 0.627±0.011 | 0.409±0.117 |
| FedProx | 0.607±0.006 | 0.614±0.022 | 0.414±0.109 |
| FedProto† | — | — | ≈0.365 (re-run required) |
| FedAvg-ClassFreq | 0.641±0.011 | 0.628±0.013 | 0.452±0.088 |
| **PGA (Ours)** | **0.639±0.013** | **0.631±0.014** | **0.483±0.090** |

† FedProto results will differ from the original paper after the gradient
  bug fix. The re-run is required to establish the correct baseline.

| Metric | Value | p-value |
|--------|-------|---------|
| PGA vs FedAvg Δ (α=0.1, 10 seeds) | +0.074 [+0.044, +0.103] | 0.001 |
| PGA vs FedAvg-ClassFreq Δ | +0.031 [+0.002, +0.060] | 0.038 |
| PGA vs FedProto Δ | +0.118 [+0.021, +0.215] | 0.021 |
| ECE reduction (federated T*, α=1000) | −60.0% | <0.001 |
| ECE: centralised vs federated T* | 0.0575 vs 0.0663 | Δ=0.0088 |

---

## Communication Cost

| Component | Cost | Frequency |
|-----------|------|-----------|
| Head parameters (per client) | 3.581 MB | Every round |
| Server broadcast (global head) | 3.581 MB | Every round |
| Validation TP/FP/FN counts | ≤168 bytes | Every round |
| Prototypes + class counts | 86 KB (K=4) / 430 KB (K=20) | **Once, before Round 1** |
| Temperature scalar | 4 bytes | Once, after convergence |

The 3.581 MB per-client head-uplink figure is the standard comparison unit
used in Figure 6. Prototype transmission is a one-time cost, not per-round.

---

## Validation Macro-F1 Computation

Global validation macro-F1 for model selection is computed by pooling
per-class confusion-matrix counts (TP, FP, FN for each of C=7 classes)
reported by each client. The server accumulates these counts and computes
macro-F1 from the pooled totals. This is equivalent to computing macro-F1
on the union of all client validation predictions and is exact — it avoids
the inaccuracy of averaging local macro-F1 scores, which are not additive.
No raw validation images, labels, or logits leave any client.

---

## Transmitted Information Summary

| Information | Sent by | Timing |
|-------------|---------|--------|
| Class-token prototypes {p_k^c} | Each client | Once, before Round 1 |
| Class-wise training counts {n_k^c} | Each client | Once, before Round 1 |
| Head parameters θ_k | Each client | Every round |
| Per-class validation TP/FP/FN | Each client | Every round |
| Temperature scalar T_k* | Each client | Once, after convergence |

**Privacy note:** FedUCA achieves data locality but not formal differential
privacy. Transmitted class counts reveal per-client class distributions.
Validation confusion-matrix counts additionally reveal prediction outcomes
per class. DP-SGD (Abadi et al., 2016, CCS) or secure aggregation would be
required for formal privacy guarantees.

---

## Environment

| Component | Version |
|-----------|---------|
| Platform | Kaggle free tier |
| GPU | NVIDIA T4 (15 GB VRAM) |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| scikit-learn | 1.3.0 |
| numpy | 1.26.0 |
| scipy | 1.11.0 |

No additional installation required on Kaggle — all packages are pre-installed.

---

## Contact

For questions about reproducibility, contact:
**Srinivas Arukonda** — srinivas.a@srmap.edu.in
Department of Computer Science and Engineering
SRM University-AP, Amaravati, Andhra Pradesh, India
