# ANRF AISEHack Theme 2 — PM2.5 Forecasting
**Team member:** Ronit  
**Kaggle username:** ronitraj1  
**Dataset URL:** https://www.kaggle.com/datasets/ronitraj1/ronit-pm25-src  
**Competition:** https://www.kaggle.com/competitions/aisehack-theme-2  

**Current status:** LB ~24 → rank ~38 | First place ~9 | Target: rank 1  
**Active strategy:** See `outputs/rank1_strategy_report.md`

### Current Live Config (from `configs/config.yaml`)
- `model.type: tfno2d`
- `data.val_month: DEC_16`
- `preprocessing.cpm25_log1p: true`
- `preprocessing.cpm25_grid_zscore: true`
- `loss.type: log_rmse`
- `training.use_weighted_sampler: true`

---

## Folder Structure

```
Ronit/
├── dataset-metadata.json      ← Kaggle upload metadata (DO NOT edit)
├── requirements.txt           ← Python dependencies
├── pipeline.py                ← (legacy, ignore)
├── README.md                  ← This file
│
├── configs/
│   └── config.yaml            ← ONLY file you edit between experiments
│
├── src/                       ← Core logic — edit only to fix bugs or add features
│   ├── __init__.py
│   ├── config.py              ← Loads config.yaml, resolves paths
│   ├── data.py                ← Normalization, sample construction, DataLoader
│   │                            (log1p, grid z-score, cpm25 masking, weighted sampler)
│   ├── model.py               ← TFNO2D, UNet, ResidualSTUNet architectures
│   ├── train.py               ← Training loop, log-RMSE / weighted-RMSE loss, optimizer
│   ├── inference.py           ← Test inference, denormalize_cpm25, saves preds.npy
│   └── utils.py               ← Seeding, device info, param count
│
├── notebooks/
│   ├── eda.ipynb              ← EDA (run locally, CPU only)
│   ├── eda_online.ipynb       ← Extended EDA with all 33 analysis plots
│   └── exp_01_baseline.ipynb  ← Kaggle submission notebook
│
└── outputs/
    ├── images/                ← 33 EDA PNG figures
    ├── models/                ← Local model checkpoints (.pt files)
    ├── submissions/           ← Local copies of preds.npy per experiment
    └── rank1_strategy_report.md  ← Full deep-audit strategy report
```

---

## Key Design Decisions (Current State)

### CPM25 input masking
At test time only 10h of cpm25 history is available (hours 11–26 are zeroed in `data.py`).
Training matches this exactly so there is no train/test mismatch.

### Preprocessing pipeline
Located in `data.py`, controlled entirely from `config.yaml`:
```yaml
preprocessing:
  cpm25_log1p: true          # log1p(x) before min-max norm → compresses winter tail
  cpm25_grid_zscore: true    # per-pixel z-score on top of log1p norm
  cpm25_grid_eps: 1.0e-6     # epsilon to avoid div-by-zero
```
Inverse chain at inference: z-score undo → min-max denorm → expm1 (in `denormalize_cpm25`).

### Loss function
Controlled by `loss.type` in `config.yaml`:
- `log_rmse` — RMSE in log1p-physical space with per-horizon weighting (current)
- `weighted_rmse_mae` — weighted blend of RMSE + MAE in normalized space (fallback)

### Weighted sampler
`training.use_weighted_sampler: true` enables `WeightedRandomSampler`.  
Weights applied **only to training months** (val month receives no weight and is excluded):
```yaml
training:
  use_weighted_sampler: true
  month_sampling_weights:
    APRIL_16: 0.5
    JULY_16:  0.5
    OCT_16:   1.5
    DEC_16:   3.0   # ← only effective when val_month ≠ DEC_16
```
> **Known bug:** current `val_month: DEC_16` means DEC is excluded from training, so
> `DEC_16: 3.0` weight has no effect. Fix: change `val_month: OCT_16` to make DEC part
> of training and allow oversampling (Protocol P1 in strategy report).

### Available model architectures (`model.type`)
| Key | Class | Notes |
|-----|-------|-------|
| `tfno2d` | `TFNO2D` | Current baseline — fast but limited capacity |
| `unet` | `UNet` | Classic skip-connection U-Net |
| `res_stunet` | `ResidualSTUNet` | **Recommended next model** — dual-branch, persistence-residual head |

Switch model by editing one line in `config.yaml`:
```yaml
model:
  type: res_stunet   # switch from tfno2d
  base_ch: 96        # 64 → 96 for bigger model
  stem_ch: 64
  dropout: 0.08
```

---

## What to Write in Each File

### `configs/config.yaml` — Your experiment control panel
This is the **only file that changes between experiments**. Everything else adapts automatically.

> The block below is a **recommended next-run template (Protocol P1)**, not the current live config.

```yaml
# ── Features (8 total = cpm25 + 6 met + PM25 emis) ──
features:
  met:  ["u10", "v10", "pblh", "rain", "t2", "q2"]
  emis: ["PM25"]
  use_aux: false

# ── Preprocessing ──
preprocessing:
  cpm25_log1p: true
  cpm25_grid_zscore: true

# ── Validation / oversampling protocol ──
data:
  val_month: "OCT_16"   # P1: use OCT as val → DEC in training → oversampling active

training:
  epochs: 60
  stride_train: 2
  stride_val: 4
  use_weighted_sampler: true
  month_sampling_weights:
    APRIL_16: 0.5
    JULY_16:  0.5
    OCT_16:   1.0      # unused (val month)
    DEC_16:   3.0      # oversampled in training ✓

# ── Model ──
model:
  type: res_stunet
  base_ch: 96
  stem_ch: 64
  dropout: 0.08

# ── Loss ──
loss:
  type: log_rmse
  horizon_weight_min: 0.8
  horizon_weight_max: 1.4
```

**Rule:** Each experiment = different values here. Never copy-paste into the notebook.

---

### `src/model.py` — Architecture changes go here
Three architectures are defined: `TFNO2D`, `UNet`, `ResidualSTUNet`.
To add a new feature (e.g. cross-attention bottleneck), add it to `ResidualSTUNet` here.

---

### `src/data.py` — Data pipeline changes go here
- `normalize_feature()` / `denormalize_cpm25()` — feature-aware transform chain
- `_compute_cpm25_grid_stats()` / `_apply_cpm25_grid_zscore()` — per-pixel z-score
- `PM25Dataset` — sliding-window samples with cpm25 masking after hour 10
- Augmentations, feature engineering go here

---

### `src/train.py` — Training changes go here
- `log_rmse_loss()` — RMSE in log1p-physical space
- `objective_loss()` — dispatches to correct loss by `cfg['loss']['type']`
- AMP (`torch.cuda.amp.autocast` + `GradScaler`) should be added here

---

### `notebooks/exp_01_baseline.ipynb` — DO NOT add logic here
This notebook only **calls** functions from `src/`. It should stay ≤15 cells.  
The only variable to set in it:
```python
KAGGLE_SRC_DATASET = "ronit-pm25-src"   # your dataset slug — already set
```

---

## One-Time Setup (Already Done)

```bash
# Kaggle CLI installed in aisehack env
conda run -n aisehack pip install kaggle

# Dataset created on Kaggle (already done)
cd ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit
conda run -n aisehack kaggle datasets create -p . --dir-mode zip
```

---

## Daily Workflow

### When you edit `src/` Python files or `configs/config.yaml`

```bash
# 1. Navigate to Ronit folder
cd ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit

# 2. Upload new version to Kaggle (run this every time you change any file)
conda run -n aisehack kaggle datasets version -p . --dir-mode zip -m "describe what you changed"
# Examples:
# -m "add residual skip connections"
# -m "reduce to 10 features for speed"
# -m "increase epochs to 30"
```

### When you edit the notebook (`exp_01_baseline.ipynb`)

```bash
# Same command — uploads everything including the notebook
conda run -n aisehack kaggle datasets version -p . --dir-mode zip -m "update notebook cells"
```

### On Kaggle after uploading

1. Go to https://www.kaggle.com/datasets/ronitraj1/ronit-pm25-src
2. Confirm the new version is visible
3. Open your Kaggle notebook
4. In the **Input** panel (right side) → find `ronit-pm25-src` → click **Check for Updates** → accept
5. Click **Run All** (or Save & Run All for a full commit/submission)

---

## Full Experiment Cycle (Step by Step)

### Step 1 — Edit locally
```bash
# Example: edit model.py to add residuals
code ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit/src/model.py
```

### Step 2 — Push to Kaggle
```bash
cd ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit
conda run -n aisehack kaggle datasets version -p . --dir-mode zip -m "add residual skip connections in TFNO2D"
```

### Step 3 — Run on Kaggle
- Open: https://www.kaggle.com/code (your notebook)
- Input panel → `ronit-pm25-src` → Check for Updates
- **Session → Accelerator → GPU P100** (verify this is selected)
- **Run All**
- Wait ~4–9 hours depending on config

### Step 4 — Submit
- After run completes, `/kaggle/working/preds.npy` is auto-generated
- Click **Submit to Competition** → leaderboard score appears in minutes

### Step 5 — Save result locally (optional but recommended)
```bash
# Download the preds.npy from your Kaggle notebook output to local
conda run -n aisehack kaggle kernels output YOUR_NOTEBOOK_SLUG -p ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit/outputs/submissions/
```

---

## Kaggle Notebook Setup (First Time Only)

1. Go to https://www.kaggle.com/code → **New Notebook**
2. **File → Import Notebook** → upload `notebooks/exp_01_baseline.ipynb`
3. **Input** (right panel) → **Add Input**:
   - Search **aisehack-theme-2** → Add (competition data)
  - Search **ronit-pm25-src** → Add (your src code)
4. **Session → Accelerator → GPU P100**
5. **Settings → Internet → On** (needed for package installs if any)
6. Run once to verify everything works

---

## Kaggle Paths (What Goes Where on the Kaggle Server)

| What | Kaggle Path | Saved after run? |
|------|------------|-----------------|
| Competition data | `/kaggle/input/competitions/aisehack-theme-2/` | N/A (read-only) |
| Your src code | `/kaggle/input/ronit-pm25-src/src/` | N/A (read-only) |
| Your config | `/kaggle/input/ronit-pm25-src/configs/config.yaml` | N/A (read-only) |
| Norm stats | `/kaggle/temp/norm_stats.npy` | ❌ No |
| Model checkpoint | `/kaggle/temp/best_model.pt` | ❌ No |
| **Predictions** | `/kaggle/working/preds.npy` | ✅ **Yes — this is submitted** |
| Training curve | `/kaggle/working/training_curves.png` | ✅ Yes |

---

## Experiment Log

| # | Date | Model | Key Config Changes | Val RMSE | LB Score | Notes |
|---|------|-------|-------------------|----------|----------|-------|
| exp_01 | | tfno2d | 10 feat, epoch=20, stride=3, no log1p | — | ~24 | Baseline run, rank ~38 |
| exp_A | | tfno2d | Fix val protocol (OCT val), 8 feat, log1p, AMP only | | | Protocol stabilisation |
| exp_B | | res_stunet | base_ch=96, stem_ch=64, same preprocessing | | | **Primary upgrade** |
| exp_C | | res_stunet | + intensity-weighted log_rmse | | | Loss emphasis |
| exp_D | | res_stunet | base_ch=128 if GPU allows | | | Capacity push |
| exp_E | | tfno2d-XL | width=96, modes=24, depth=6 | | | Diversity model for ensemble |
| exp_F | | ensemble | Top 2–4 checkpoints + TTA flips | | | **Final submission** |

> **Rule:** Never submit a single-seed single-model after Run B. Always ensemble.

---

## 6-Run Roadmap to Rank 1

### Run A — Stabilisation
- Fix `val_month: OCT_16` so December oversampling actually activates
- Keep 8 features + log1p pipeline
- Add AMP (`autocast` + `GradScaler`) in `src/train.py`

### Run B — Bigger model (biggest expected jump)
- `model.type: res_stunet`, `base_ch: 96`, `stem_ch: 64`
- Same preprocessing and loss as Run A

### Run C — Emphasis loss
- Add intensity-weighted factor to `log_rmse`:
  `w = 1 + 1.5 * clamp(target_mean / 59, 0, 3)`

### Run D — Max capacity
- `base_ch: 128` if VRAM allows; else gradient accumulation

### Run E — Diversity model
- `model.type: tfno2d`, `width: 96`, `modes: 24`, `depth: 6`
- Train independently for ensemble diversity

### Run F — Submission ensemble
- Average preds from best Run B/C/D + Run E
- 4-fold TTA: identity, H-flip, V-flip, both-flip (invert before averaging)

> Optional robust alternative (Protocol P2): keep `val_month: DEC_16`, disable DEC oversampling,
> and rely on intensity-weighted loss for winter emphasis.

---

## Quick Reference — All Commands

```bash
# Activate environment
conda activate aisehack

# Upload updated files to Kaggle
cd ~/Documents/CODING/Hackathon/ANRF_AISEHack_Code/Ronit
conda run -n aisehack kaggle datasets version -p . --dir-mode zip -m "YOUR MESSAGE HERE"

# Check dataset versions on Kaggle
conda run -n aisehack kaggle datasets status ronitraj1/ronit-pm25-src

# List your Kaggle notebooks
conda run -n aisehack kaggle kernels list --mine

# Current baseline kernel slug
# ronitraj1/YOUR_NOTEBOOK_SLUG

# Check if a notebook run is complete
conda run -n aisehack kaggle kernels status YOUR_NOTEBOOK_SLUG

# Download notebook output (preds.npy) after run
conda run -n aisehack kaggle kernels output YOUR_NOTEBOOK_SLUG -p ./outputs/submissions/

# Check leaderboard
conda run -n aisehack kaggle competitions leaderboard aisehack-theme-2 --show
```

---

## Submission Checklist

Before every Kaggle run verify:
- [ ] `config.yaml` has the correct `model.type`, `base_ch`, features, epochs, stride
- [ ] Validation protocol chosen: P1 (`val_month: OCT_16` + DEC oversampling) or P2 (`val_month: DEC_16` + no DEC oversampling)
- [ ] `preprocessing.cpm25_log1p: true` and `cpm25_grid_zscore: true`
- [ ] `loss.type: log_rmse`
- [ ] Dataset version updated on Kaggle (`kaggle datasets version ...`)
- [ ] Kaggle notebook has latest dataset version (Check for Updates)
- [ ] Accelerator is set to **GPU P100**
- [ ] `preds.npy` shape will be `(996, 140, 124, 16)` (guaranteed by `inference.py` assert)
- [ ] After Run B: **always ensemble** before submitting — no single-model submissions
- [ ] 3 submissions/day limit — don't waste runs on untested changes

### KPIs to Track Per Run
After each run, record in the Experiment Log above:
- Val RMSE overall
- Val RMSE high-pollution subset (top 20% target means)
- Per-horizon RMSE at t+1, t+8, t+16
- LB score

> If a run improves easy subsets but not high-pollution subset, reject it even
> if average val improves.

---

## GPU Quota Warning

| Action | GPU Hours Used |
|--------|----------------|
| tfno2d, 8 features, 20 epochs | ~4–5h |
| tfno2d, 8 features, 60 epochs | ~8–9h |
| res_stunet base_ch=96, 60 epochs | ~9–12h |
| res_stunet base_ch=128, 60 epochs | ~14–18h |
| Weekly budget | ~30h |

**Maximum safe runs per week: ~3–4** (res_stunet) or **~5–6** (tfno2d). Budget carefully.
