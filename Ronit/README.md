# ANRF AISEHack Theme 2 — PM2.5 Forecasting
**Team member:** Ronit  
**Kaggle username:** ronitraj1  
**Dataset URL:** https://www.kaggle.com/datasets/ronitraj1/ronit-pm25-src  
**Competition:** https://www.kaggle.com/competitions/aisehack-theme-2  

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
│   ├── model.py               ← TFNO2D architecture
│   ├── train.py               ← Training loop, loss, optimizer
│   ├── inference.py           ← Test inference, saves preds.npy
│   └── utils.py               ← Seeding, device info, param count
│
├── notebooks/
│   ├── eda.ipynb              ← EDA (run locally, CPU only)
│   └── exp_01_baseline.ipynb  ← Kaggle submission notebook
│
└── outputs/
    ├── models/                ← Local model checkpoints (.pt files)
    └── submissions/           ← Local copies of preds.npy per experiment
```

---

## What to Write in Each File

### `configs/config.yaml` — Your experiment control panel
This is the **only file that changes between experiments**. Everything else adapts automatically.

```yaml
# Change features to experiment with subsets:
features:
  met: ["u10", "v10", "pblh", "rain", "t2", "q2"]      # 6 features = faster
  emis: ["PM25", "SO2", "NOx"]                           # 3 emis = faster
  # Full 16-feature run: add "swdown","psfc" to met and "NH3","NMVOC_e","NMVOC_finn","bio" to emis

# Change these for faster first run:
training:
  epochs: 20          # Start with 20, increase to 30 for final submission
  stride_train: 3     # Higher = fewer samples = faster epoch (start with 3)
  stride_val: 6

# Change these for better model:
model:
  width: 64           # Increase to 96/128 for more capacity (uses more memory)
  modes: 20           # Fourier modes — higher = finer spatial patterns
  depth: 4            # More blocks = deeper model (slower)
```

**Rule:** Each experiment = different values here. Never copy-paste into the notebook.

---

### `src/model.py` — Architecture changes go here
When adding residuals, changing depth, or trying a new architecture:
```python
# In TFNO2D.forward(), change:
for block in self.blocks:
    x = block(x)
# To (adds residual skip connections):
for block in self.blocks:
    x = block(x) + x    # ← residual
```

---

### `src/data.py` — Data pipeline changes go here
- Feature engineering (e.g. wind speed magnitude from u10+v10)
- Different masking strategies  
- Augmentations

---

### `src/train.py` — Training changes go here
- Different loss functions
- Different schedulers
- Mixed precision training (`torch.cuda.amp`)

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

## Experiment Log (Fill This In)

| # | Date | Config changes | Val RMSE | LB Score | Notes |
|---|------|---------------|----------|----------|-------|
| exp_01 | | 10 feat, epoch=20, stride=3 | | | Baseline |
| exp_02 | | + residual skip | | | |
| exp_03 | | 16 feat, epoch=30 | | | |

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
- [ ] `config.yaml` has the correct features/epochs/stride
- [ ] Dataset version updated on Kaggle (`kaggle datasets version ...`)
- [ ] Kaggle notebook has latest dataset version (Check for Updates)
- [ ] Accelerator is set to **GPU P100**
- [ ] `preds.npy` shape will be `(996, 140, 124, 16)` (guaranteed by `inference.py` assert)
- [ ] 3 submissions/day limit — don't waste runs on untested changes

---

## GPU Quota Warning

| Action | GPU Hours Used |
|--------|----------------|
| Full run, 10 features, 20 epochs | ~4–5h |
| Full run, 16 features, 30 epochs | ~8–9h |
| Weekly budget | ~30h |

**Maximum safe runs per week: ~5–6** (with 10 features). Budget carefully.
