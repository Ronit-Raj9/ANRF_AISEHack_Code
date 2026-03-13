# Rank-1 Strategy Report (Deep Audit)

## 0) Objective and Current Position
- Current leaderboard score: **~24** (rank ~38), top score ~9.
- Required improvement is large, so we need **systematic jump steps**, not small tuning.
- This report is based on:
  - EDA notebook findings in `notebooks/eda_online.ipynb` and `notebooks/eda.ipynb`
  - EDA figures in `outputs/images/*`
  - Current training stack in `src/data.py`, `src/train.py`, `src/model.py`, `src/inference.py`
  - Current experiment config in `configs/config.yaml`

---

## 1) What EDA Clearly Says (High-Confidence Facts)

## 1.1 Distribution + season shift (biggest source of error)
From EDA sections and figures (`eda_cpm25_distributions.png`, `eda_temporal_cpm25.png`, `eda_extreme_events.png`, `eda_test_temporal_distribution.png`):
- PM2.5 is heavy-tailed, especially in winter.
- December is hardest (highest mean/variance/extremes).
- Test distribution is closer to **Oct/Dec** than Apr/Jul.

**Implication:** if model optimizes average behavior across all months equally, it underfits the most important test regime.

## 1.2 Physics drivers are not symmetric
From (`eda_correlations.png`, `eda_pblh_vs_cpm25.png`, `eda_q2_spatial_corr.png`, `eda_winds.png`):
- `pblh`, `t2`, `q2` carry strongest useful predictive signal.
- Wind still matters for transport but often via advection/stagnation structure, not only global correlation.

**Implication:** architecture should explicitly separate **state memory** vs **future forcing** and preserve spatial transport structure.

## 1.3 Redundancy and sparsity matter
From (`eda_inter_feature_corr.png`, `eda_sparsity.png`, `eda_emissions.png`, `eda_swdown_daynight_cycle.png`):
- Several emissions are highly collinear or very sparse.
- Some features are mostly static priors (`psfc`, inventory-like emissions).

**Implication:** too many noisy channels can hurt more than help; compact feature sets are often better first.

## 1.4 Leakage and validation realism
From (`eda_cv_contamination.png`):
- Random splits are optimistic.
- Time-aware blocked validation is required.

**Implication:** if validation is easy/optimistic, your tuning will fail on leaderboard.

---

## 2) Audit of Current Code + Hyperparameters

## 2.1 Good changes already present
- CPM25 training-test mismatch fix (masking beyond 10h) is implemented in `src/data.py`.
- Log-aware target path and cpm25-specific inverse transform are implemented.
- Grid-wise z-score option for cpm25 is implemented.
- 8-feature setup and Dec-holdout config are set in `configs/config.yaml`.

These are strong improvements over baseline.

## 2.2 Critical issues still limiting score

### Issue A: Oversampling config currently cannot oversample December
- In current config, `val_month: DEC_16`.
- Training months are derived as all except val month.
- So `DEC_16` is **not in training set**, but sampler weights include DEC=3.0.

**Effect:** your intended “December oversampling” is inactive.

### Issue B: Objective/selection mismatch risk
- Training objective may use `log_rmse`, but printed/selected validation metric still uses normalized RMSE path.
- This can still work, but best-checkpoint selection can be suboptimal for leaderboard objective.

### Issue C: “Bigger robust model” not yet active
- Config still uses `model.type: tfno2d`.
- `res_stunet` exists but is not the active model.
- No ensemble/TTA in current Kaggle run loop.

### Issue D: Robustness missing in runtime strategy
- Single-seed runs only.
- No uncertainty reduction via ensembling.
- No horizon-aware post-processing calibration.

---

## 3) Why 24 Happens (Root-Cause Hypothesis)
1. Model family/capacity and inductive bias are still too close to baseline for this target regime.
2. Training objective and data weighting are not yet fully aligned to the test-heavy winter tail.
3. No ensemble/TTA means high variance and unstable leaderboard outcomes.
4. Validation protocol may still not perfectly mirror leaderboard pressure.

---

## 4) What To Improve Next (Ordered by Impact)

## 4.1 Immediate Run Fixes (before any new architecture)

### Step 1 — make seasonal weighting actually work
Pick one of these two protocols:

**Protocol P1 (recommended for leaderboard push):**
- `val_month: OCT_16`
- Train on `APRIL + JULY + DEC` and oversample `DEC` strongly.
- This allows true DEC oversampling and still gives a hard-ish val month.

**Protocol P2 (robust model selection):**
- Keep `val_month: DEC_16` for realism.
- Disable DEC oversampling (cannot apply).
- Use weighted loss by sample intensity instead.

### Step 2 — align checkpoint metric with objective
- If `loss.type: log_rmse`, save best model by a validation metric closer to leaderboard target:
  - Either physical-space RMSE proxy,
  - or mixed metric: `0.7 * val_log_rmse + 0.3 * val_rmse_phys_proxy`.

### Step 3 — add mixed precision
- Use AMP (`autocast` + `GradScaler`) in training.
- Gains: higher batch or longer/deeper model under same GPU budget.

---

## 4.2 Recommended Advanced Model (bigger + robust)

## Model choice: **ResidualSTUNet++ (primary)**
Use your existing `res_stunet` as base and upgrade it.

### Architecture upgrades
1. **Dual temporal encoders (already conceptually present):**
   - Branch A: `cpm25` + dynamic context for first 10h
   - Branch B: exogenous forcing for 26h
2. **Cross-attention fusion block** at bottleneck (new)
3. **Multi-scale deep supervision** at decoder heads (new)
4. **Horizon-conditioned head** (new): separate projection per lead bucket (1–4, 5–8, 9–12, 13–16)
5. **Residual persistence head** (already present conceptually) retained

### Suggested “bigger” hyperparameters
- `model.type: res_stunet`
- `base_ch: 96` (start), then 128 if memory allows
- `stem_ch: 64`
- `dropout: 0.08`
- Add stochastic depth ~0.05 (if implemented)
- Keep GroupNorm (stable for small batch)

### Why this over pure TFNO scaling
- Your data is limited and nonstationary with sparse/extreme behavior.
- A residual U-Net with explicit state/forcing separation is more robust than blindly widening spectral blocks.
- It handles local hotspot sharpness + large-scale transport when fused properly.

---

## 4.3 Secondary Model Track (for ensemble diversity)
Use **TFNO2D-XL** as a second family model (not main model):
- `width: 96`
- `modes: 24`
- `depth: 6`
- Add residual skip across blocks
- Add lightweight channel attention after each block

Reason: architectural diversity improves ensemble gain.

---

## 5) Training Recipe for Top-Leaderboard Push

## 5.1 Loss stack (recommended)
Primary:
- `log_rmse` (already integrated directionally)

Add:
- Intensity-weighted factor from target mean:
  - `w = 1 + alpha * clamp(target_mean / ref, 0, cap)`
  - start `alpha=1.5`, `ref=59`, `cap=3`

Optional robustifier:
- Blend 10–20% Huber in physical/log space.

## 5.2 Schedule
- Epochs: 40–60 effective (with early stop)
- LR: warmup 3–5 epochs then cosine to 1e-5
- Gradient clip: 1.0 (keep)
- EMA of weights (0.999) for stabler validation

## 5.3 Batch and throughput
- Enable AMP first.
- Use largest stable batch for your architecture.
- If OOM: gradient accumulation before reducing model too much.

---

## 6) Inference Upgrades (low risk, high ROI)
1. Non-negativity clamp (already present).
2. **TTA flips** (H, W, both), inverse-flip and average.
3. **Seed ensemble** (at least 3 seeds): average predictions.
4. Optional horizon calibration (linear rescale per lead on validation).

Expected practical gain from ensemble+TTA alone can be substantial relative to single-run variance.

---

## 7) Concrete Next 6 Runs (to maximize chance of rank jump)

### Run A (stabilization)
- Keep current 8 features + log pipeline
- Fix protocol issue (P1 or P2)
- Add AMP only

### Run B (bigger robust model)
- Switch to `res_stunet`, `base_ch=96`, `stem_ch=64`
- Same preprocessing and loss

### Run C (loss emphasis)
- Add intensity-weighted log_rmse

### Run D (capacity push)
- `res_stunet base_ch=128` if memory allows

### Run E (TFNO diversity model)
- Train TFNO2D-XL variant

### Run F (submission ensemble)
- Ensemble top 2–4 checkpoints/models + TTA

Do not spend submissions on single-seed single-model after Run B.

---

## 8) KPI Dashboard You Must Track Per Run
For each run store:
- Val RMSE overall
- Val RMSE on high-pollution subset (top 20% target means)
- Per-horizon RMSE (1..16)
- Night vs day RMSE
- Dec-like subset RMSE (or Oct/Dec weighted)
- LB score

If a run improves only easy subsets but not high-pollution subset, reject it even if average val improves.

---

## 9) Priority Decision: Which Model To Use Next?
If your goal is rank-1 push with limited attempts:
- **Primary model to build bigger and robust:** `res_stunet++` path (not plain tfno2d)
- **Secondary support model:** widened TFNO for ensemble diversity
- **Submission strategy:** ensemble + TTA mandatory

---

## 10) Final Recommendation (Short Version)
1. Fix weighting/validation protocol mismatch first.
2. Move main training to bigger residual dual-branch U-Net family.
3. Add AMP + weighted log loss + strong tracking of tail metrics.
4. Submit only ensembled predictions.

This is the fastest route from ~24 toward competitive top-tier scores.
