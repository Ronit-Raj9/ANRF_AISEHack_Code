# 🌫️ ANRF AISEHack Theme 2 — PM2.5 Pollution Forecasting

> **National Climate AI Hackathon** | Country-Level PM2.5 Concentration Forecasting over India
> 
> Forecast 16 hours of surface PM2.5 concentration across a 140 × 124 grid of India using WRF-Chem simulation data.

---

## 📋 Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Repository Structure](#-repository-structure)
3. [Setup & Installation](#-setup--installation)
4. [Dataset Overview](#-dataset-overview)
5. [EDA Findings — Full Summary](#-eda-findings--full-summary)
   - [Domain & Geographic Scope](#1-domain--geographic-scope)
   - [Feature Taxonomy](#2-feature-taxonomy)
   - [Temporal Asymmetry](#3-temporal-asymmetry--the-10-16-26-rule)
   - [Target Distribution](#4-target-distribution-cpm25)
   - [April — Summer Season Deep Dive](#-april--summer-season-deep-dive-april_16)
   - [Primary Physical Forcings](#5-primary-physical-forcings)
   - [Transport & Scavenging](#6-transport--scavenging-dynamics)
   - [Emission Inventory](#7-emission-inventory-mechanics)
   - [Spatio-Temporal Rhythms](#8-spatio-temporal-rhythms)
   - [Geographic Non-Stationarity](#9-geographic-non-stationarity)
   - [Normalization Discrepancy](#10-normalization-discrepancy)
   - [Test Set Intelligence](#11-test-set-intelligence)
   - [Persistence Benchmark](#12-persistence-benchmark)
6. [Model Architecture & Rank-1 Strategy](#-model-architecture--rank-1-strategy)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Architecture: Tucker-FrNO](#architecture-tucker-frno)
   - [Data Strategy](#data-strategy)
   - [Loss Function](#loss-function)
   - [Inference & TTA](#inference--tta)
7. [Critical Missing Safeguards](#-critical-missing-safeguards)
8. [EDA Completion Checklist](#-eda-completion-checklist)
9. [Submission Format](#-submission-format)
10. [References](#-references)

---

## 🏆 Competition Overview

| Field | Details |
|---|---|
| **Competition** | ANRF AISEHack Theme 2 — PM2.5 Pollution Forecasting |
| **Platform** | Kaggle (Notebook-based submission) |
| **Task** | 16-hour PM2.5 forecast over India (140 × 124 spatial grid) |
| **Training Data** | WRF-Chem 2016 — April, July, October, December |
| **Test Data** | 996 samples from 2017 (≈75% December, ≈24% October) |
| **Metric** | Average Domain RMSE (lower is better) |
| **Baseline RMSE** | 30.83 µg/m³ (Persistence @ t+16) |
| **Rank-1 Target** | < 21.6 µg/m³ (beat baseline by ≥ 30%) |
| **Submission** | `preds.npy` with shape `(996, 140, 124, 16)` |

**Why this matters:** India's air quality frequently exceeds national safety standards by 8–10×. The Global Burden of Disease 2019 attributed ~1.67 million deaths and $36.8 billion in economic costs to air pollution in India. A reliable short-term forecasting system protects 1.4 billion citizens.

---

## 📁 Repository Structure

```
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── data/
│   └── raw/
│       ├── APRIL_16/             # April 2016 — Summer season
│       │   ├── cpm25.npy
│       │   ├── t2.npy
│       │   ├── pblh.npy
│       │   ├── u10.npy  v10.npy
│       │   ├── q2.npy   rain.npy
│       │   ├── swdown.npy  psfc.npy
│       │   ├── PM25.npy  NOx.npy  SO2.npy
│       │   ├── NH3.npy  NMVOC_e.npy
│       │   ├── NMVOC_finn.npy  bio.npy
│       │   └── time.npy
│       ├── JULY_16/              # July 2016 — Monsoon season
│       ├── OCT_16/               # October 2016 — Post-monsoon
│       ├── DEC_16/               # December 2016 — Winter season
│       └── lat_long.npy          # (140, 124, 2) lat/lon grid
│
├── test_in/                      # Test inputs (2017 data)
│   ├── cpm25.npy                 # Shape: (996, 10, 140, 124)
│   └── <all other features>.npy  # Shape: (996, 26, 140, 124)
│
├── stats/
│   └── feat_min_max.mat          # ⚠️ DO NOT USE — see Normalization section
│
├── notebooks/
│   ├── 01_EDA_seasonal.ipynb
│   ├── 02_EDA_features.ipynb
│   ├── 03_EDA_normalization.ipynb
│   └── 04_EDA_test_intelligence.ipynb
│
├── src/
│   ├── dataset.py                # Dataset class + sliding window sampler
│   ├── scaler.py                 # Grid-wise log-standardisation (17,360 scalers)
│   ├── model.py                  # Tucker-FrNO architecture
│   ├── loss.py                   # Spatially-weighted MSE + residual head
│   ├── train.py                  # Training loop with intensity sampling
│   └── inference.py              # TTA + ensemble inference
│
└── submission.ipynb              # End-to-end Kaggle notebook
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (NVIDIA P100 / T4 recommended)
- ~32 GB RAM (for loading all four months at stride=1)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/anrf-pm25-forecasting.git
cd anrf-pm25-forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
numpy>=1.24.0
torch>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
einops>=0.6.0
tqdm>=4.65.0
h5py>=3.8.0          # for reading .mat files (if needed)
jupyter>=1.0.0
```

### 4. Kaggle Environment Setup

If running on Kaggle notebooks, attach the competition dataset and baseline repo:

```python
import os
DATA_DIR  = "/kaggle/input/aisehack-theme-2"
TEST_DIR  = "/kaggle/input/aisehack-theme-2/test_in"
WORK_DIR  = "/kaggle/working"

# Verify data is accessible
for month in ["APRIL_16", "JULY_16", "OCT_16", "DEC_16"]:
    path = os.path.join(DATA_DIR, "raw", month)
    files = os.listdir(path)
    print(f"{month}: {len(files)} files")
```

### 5. Verify Data Shapes

```python
import numpy as np

# Raw training data — shape: (timesteps, 140, 124)
cpm25 = np.load(f"{DATA_DIR}/raw/DEC_16/cpm25.npy")
print(f"DEC cpm25 shape : {cpm25.shape}")   # (744, 140, 124) for Dec

# Test data — shape: (996, T, 140, 124)
test_cpm = np.load(f"{TEST_DIR}/cpm25.npy")
test_met = np.load(f"{TEST_DIR}/t2.npy")
print(f"Test cpm25 shape: {test_cpm.shape}") # (996, 10, 140, 124)
print(f"Test t2 shape   : {test_met.shape}") # (996, 26, 140, 124)
```

### 6. Quick Sanity Check

```python
# Confirm December max exceeds official .mat ceiling
print(f"Dec cpm25 max  : {cpm25.max():.1f}")  # Should be ~2849.7
print(f"Official ceiling: 1465.2")
print(f"→ DO NOT use feat_min_max.mat for normalisation!")
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| **Spatial Grid** | 140 rows × 124 columns (17,360 pixels) |
| **Resolution** | 25 km × 25 km per grid cell |
| **Temporal Resolution** | 1 hour |
| **Lat Range** | 7.06°N – 38.00°N |
| **Lon Range** | 67.79°E – 97.93°E |
| **Source Model** | WRF-Chem (Weather Research & Forecasting with Chemistry) |
| **Training Year** | 2016 (4 months) |
| **Test Year** | 2017 (2 inferred months) |
| **Total Features** | 16 (1 target + 8 meteorological + 7 emission) |
| **Input window** | 10 hrs of cpm25 + 26 hrs of all other features |
| **Output window** | 16 hrs of cpm25 |

### Feature Groups

**Target Variable:**
| Feature | Unit | Description |
|---|---|---|
| `cpm25` | µg/m³ | Surface PM2.5 — input (t0–t9) and forecast target (t10–t25) |

**Meteorological Variables (8):**
| Feature | Unit | Scale | Physical Role |
|---|---|---|---|
| `t2` | K | 260–330 | Temperature — controls inversions and atmospheric stability |
| `pblh` | m | 50–3,500 | Planetary Boundary Layer Height — atmospheric "lid" (r = −0.749) |
| `q2` | kg/kg | 0–0.03 | Specific humidity — secondary particle formation |
| `u10` | m/s | −15 to +15 | Zonal (E–W) wind — plume transport |
| `v10` | m/s | −15 to +15 | Meridional (N–S) wind — monsoon tracer |
| `swdown` | W/m² | 0–920 | Downward solar radiation — PBLH expansion driver |
| `psfc` | Pa | 50k–102k | Surface pressure — topography proxy (35,660 Pa range) |
| `rain` | mm/hr | 0–397 | Rainfall — wet deposition scavenging (70% zeros) |

**Emission Variables (7):**
| Feature | Unit | Scale | Sparsity | Physical Role |
|---|---|---|---|---|
| `PM25` | kg/m²/s | ~1e-8 | 5% | Direct PM2.5 source map |
| `NOx` | mol/km²/hr | ~1e-9 | 8% | Industrial/transport precursor |
| `SO2` | mol/km²/hr | ~1e-9 | 10% | Sulfate aerosol precursor (r=0.99 with NOx — redundant!) |
| `NH3` | mol/km²/hr | ~1e-10 | 12% | Agricultural ammonia — key for IGP winter haze |
| `NMVOC_e` | mol/km²/hr | ~1e-10 | 15% | Urban/anthropogenic organic aerosols |
| `bio` | mol/km²/hr | ~1e-11 | 20% | Biogenic isoprene from vegetation |
| `NMVOC_finn` | mol/km²/hr | ~1e-11 | **98.4%** | Biomass burning — active only during Oct/Nov fires |

---

## 🔬 EDA Findings — Full Summary

### 1. Domain & Geographic Scope

The 140 × 124 grid covers all of India at 25 km resolution. Key geographic features that dominate the physics:

- **The Indo-Gangetic Plain (IGP):** Stretches from Punjab through Delhi to West Bengal. Bounded by the Himalayas to the north, creating a "Pollution Dome" in winter where cold air and inversions trap emissions. Peak hotspot cell (Grid 93, 39) has a December mean of **466.2 µg/m³**.
- **The Himalayan Wall:** Acts as a physical reflection barrier. Pollution plumes hitting the mountains pool and concentrate rather than dispersing north.
- **Southern Coastal Regions:** Clean baseline (mean < 25 µg/m³). Sea breezes dominate. Feature correlations behave fundamentally differently here vs. the IGP (see Section 9).

> **Spatial correlation radius: ~275 km (≈ 11 grid cells).** This is the minimum receptive field any model architecture must capture to model plume transport correctly.

---

### 2. Feature Taxonomy

The 16 features span **9 orders of magnitude** — from surface pressure (~10⁵ Pa) to emission features (~10⁻¹¹ mol/km²/hr). This is the fundamental preprocessing challenge: a standard neural network initialised with small weights will treat emission values as exactly zero without explicit rescaling.

**Feature Tiers (recommended priority):**

| Tier | Features | Reason |
|---|---|---|
| Tier 1 (Critical) | `pblh`, `t2`, `u10`, `v10` | Highest correlations; dominate physical processes |
| Tier 2 (Important) | `q2`, `swdown`, `psfc`, `rain` | Secondary physical drivers |
| Tier 3 (Emission) | `PM25`, `NOx`, `NH3` | Key spatial priors; drop SO2 (redundant with NOx) |
| Tier 4 (Optional) | `NMVOC_e`, `bio` | Lower priority; highly correlated or low signal |
| Mask Only | `NMVOC_finn` | 98.4% zeros — use binary mask, not raw values |

---

### 3. Temporal Asymmetry — The "10-16-26" Rule

This is the core structural challenge of the task:

```
Time window:  |-- t0 -------- t9 --|-- t10 ------------- t25 --|
cpm25:        |  KNOWN (10 hrs)    |  PREDICT (16 hrs)         |
Met/Emission: |         KNOWN — Full 26 hours available        |
```

- **t+1 to t+4:** Autocorrelation > 0.90 — persistence is nearly unbeatable in this range.
- **t=10 (cliff):** Autocorrelation drops to ~0.65–0.75. This is where the PM2.5 input ends.
- **t+16 (horizon):** Autocorrelation ~0.55 — initial state has minimal predictive value.
- **t=24 (echo):** A distinct bump in the autocorrelation plot confirms a 24-hour diurnal echo — the atmosphere "remembers" what time of day it is.
- **Lags 6–18 (PACF):** Provide almost zero unique information. The model must stop relying on PM2.5 history and switch entirely to meteorological forcing during this window.

**Key implication:** The model architecture must separate *state memory* (what was pollution at t=9?) from *future forcing* (what is the wind doing at t=20?).

---

### 4. Target Distribution (cpm25)

The target variable is severely **right-skewed** with a heavy tail driven by winter pollution episodes.

| Statistic | April (Summer) | July (Monsoon) | October (Post-M) | December (Winter) |
|---|---|---|---|---|
| Mean (µg/m³) | 22.1 | 15.4 | 37.0 | 59.1 |
| Median (µg/m³) | ~14 | ~9 | ~22 | ~23.7 |
| 95th Percentile | ~75 | ~45 | ~120 | ~220 |
| Absolute Maximum | ~350 | ~180 | ~600 | **2,849.7** |
| Persistence RMSE t+16 | 22.1 | 14.4 | 31.5 | **47.8** |
| % of Total Variance | ~8% | ~5% | ~18% | **~27%** |
| Dominant Driver | Dust + pre-monsoon heat | Monsoon washout | Crop residue fires begin | Winter inversions |
| PBLH regime | High (1,500–3,000 m) | Moderate–high | Moderate collapsing | Very low (<500 m nights) |
| Diurnal Amplitude | Moderate (1.5×) | Weak (rain-dampened) | Increasing (2×) | Extreme (3×) |

> ⚠️ **December single-handedly determines leaderboard rank.** A model that trains well on July but fails in December will lose the competition. December accounts for 27% of total training variance and the persistence floor is 3.3× harder (47.8 vs 14.4 µg/m³).

**Why log-transformation is mandatory:**  
The mean (59.1) is 2.5× larger than the median (23.7) in December — confirming extreme right skew. A single 2,849 µg/m³ spike creates a gradient thousands of times larger than a 10 µg/m³ day, causing the model to completely ignore clean-air physics. Apply `log1p(x)` before any normalisation.

---

## 🌞 April — Summer Season Deep Dive (`APRIL_16`)

April represents the **pre-monsoon summer regime** — a transitionally complex season that is
often underestimated in importance. While it accounts for only ~1% of the test set, it is
a critical *training season* that teaches the model how the atmosphere behaves when neither
winter inversions nor monsoon washout dominate.

### Physical Regime: Pre-Monsoon Summer

| Property | April (Summer) Value | Contrast with December |
|---|---|---|
| Mean 2m Temperature | ~305 K (~32°C) | ~288 K (~15°C) in Dec |
| Mean PBLH | 1,500–3,000 m (afternoon) | 300–600 m (night) |
| Mean PM2.5 | ~22.1 µg/m³ | ~59.1 µg/m³ |
| Dominant pollution source | Dust storms + biomass burning | Industrial + vehicular + crop residue |
| Rain frequency | Very rare (<2% of hours) | Near zero |
| Wind pattern | South-westerlies building | Calm / north-easterly |
| Humidity | Low (dry, hot air) | Moderate–high (foggy mornings) |

### Key April-Specific EDA Findings

**1. Dust as the Primary PM2.5 Source**

Unlike December (where industrial and vehicular emissions dominate), April PM2.5 is heavily
driven by **wind-blown dust** from the Thar Desert (Rajasthan), Arabian Peninsula transport,
and pre-monsoon convective mixing that lifts surface soils.

- Dust episodes produce **spatially smooth, broad-scale PM2.5 plumes** — very different
  from the sharp point-source spikes of December.
- Emission features (NOx, SO2) are **weaker predictors** in April than any other month,
  because dust is not represented in the anthropogenic emission inventory.
- The model must learn that high April PM2.5 can occur even with **low emission inventory
  values** — purely driven by wind lifting natural dust.

**2. High PBLH — The "Clean-ish" Paradox**

April has the highest mean PBLH of all four months (often exceeding 2,500 m in the afternoon).
This creates a counterintuitive pattern:

- Emissions are **diluted** across a much larger atmospheric volume → lower PM2.5 despite
  similar emission rates to other months.
- The **diurnal amplitude is moderate** (~1.5×): PBLH rises strongly during the day but does
  not collapse as severely at night as in December.
- The model must learn the April PBLH regime separately — applying December's hyperbolic
  PBLH-PM2.5 relationship to April data will systematically over-predict daytime pollution.

**3. Pre-Monsoon Wind Shift**

April marks the transition from the dry north-easterly winter circulation to the building
south-westerly monsoon winds. This creates:

- **Higher wind variability** than any other month — the atmosphere is "in transition."
- Wind direction becomes less predictable over a 16-hour horizon, which **increases forecast
  difficulty** in the t+12 to t+16 range despite the cleaner baseline.
- u10/v10 future predictability drops to r ≈ 0.60 at t+16 (vs. r ≈ 0.72 in July when the
  monsoon is fully established and directionally consistent).

**4. Biomass Burning — Spring Agricultural Fires**

A secondary but notable April PM2.5 source is **spring agricultural burning** in parts of
central and eastern India (wheat stubble burning before the summer crop planting). This is
distinct from the October/November Punjab stubble burning:

- Fires are more scattered and less intense than the Oct/Nov events.
- NMVOC_finn shows **occasional non-zero spikes** in April (vs. sustained high values in Oct/Nov).
- The binary mask strategy for NMVOC_finn still applies, but the spatial pattern differs.

**5. April PM2.5 Spatial Distribution**

Unlike December's IGP-concentrated dome, April PM2.5 is more **spatially diffuse**:

- Elevated PM2.5 appears across **Rajasthan, Gujarat, and Central India** (dust belt).
- The IGP still shows elevated values but the north-south gradient is less sharp.
- Southern India shows relatively clean air (mean ~12 µg/m³) — similar to July conditions.

```
April Spatial Pattern:
  High (>60 µg/m³):   Rajasthan, Gujarat, West MP  ← Dust belt
  Moderate (20-60):   IGP, Central India
  Low (<20 µg/m³):    Southern India, Coastal areas
  
December Spatial Pattern (for contrast):
  Extreme (>200 µg/m³): IGP hotspot band          ← Inversion dome
  High (60-200):         NCR, Bihar, UP, West Bengal
  Moderate (20-60):      Central India
  Low (<20 µg/m³):       Southern India, Coastal
```

**6. April Autocorrelation Behaviour**

April sits between the two extremes:

- Autocorrelation decays faster than December (pollution events are less persistent —
  dust storms clear quickly with wind shifts) but slower than July.
- The 9-hour independence gap still holds, but the "plateau" in the persistence RMSE
  curve appears **earlier** (~step 10) than December (~step 12).
- This means the 16-hour forecast for April is actually slightly *harder* relative to
  its baseline than the numbers suggest — the atmosphere shuffles faster.

**7. Feature Importance Ranking in April (differs from global ranking)**

| Rank | Feature | April Role | Notes |
|---|---|---|---|
| 1 | `u10`, `v10` | **Wind is king** in April | Dust transport; direction determines PM2.5 more than any other season |
| 2 | `pblh` | High values → dilution | Weaker correlation than December (r ≈ −0.55 vs −0.75) |
| 3 | `swdown` | Very high in summer | Strong solar heating drives convective mixing |
| 4 | `t2` | Moderate correlation | Heat drives PBLH up, reducing PM2.5 |
| 5 | `psfc` | Terrain proxy | Less relevant than December; pressure gradients weaker |
| 6 | `PM25_emis` | Weaker than usual | Dust not in inventory; emission map less predictive |
| 7 | `rain` | Near zero | Almost no rain in April; ignore rain-washout mechanism |
| 8 | `q2` | Low humidity | Dry air; secondary aerosol formation suppressed |

### April vs All Seasons — Side-by-Side

| Dimension | April | July | October | December |
|---|---|---|---|---|
| **PM2.5 Driver** | Dust + spring fires | Monsoon washout | Crop fires begin | Inversions + industry |
| **PBLH** | Very high (2,500 m) | High (rain-mixed) | Moderate-collapsing | Very low (<400 m) |
| **Wind** | Transitional, variable | Strong SW monsoon | Weakening | Calm, stagnant |
| **Rain** | Near zero | Frequent (washout) | Rare | Near zero |
| **Diurnal Amplitude** | Moderate (1.5×) | Weak (0.8×) | Growing (2×) | Extreme (3×) |
| **Forecast Difficulty** | Moderate | Easy | Medium-Hard | Hardest |
| **Training Value** | Teaches dust physics | Teaches washout | Teaches fire events | Most important season |
| **Test Set %** | ~1% | 0% | ~24% | ~75% |
| **Model Strategy** | Wind-driven; ignore emissions | Use rain mask | Use NMVOC_finn mask | Weight heavily in loss |

### April EDA Action Items

- [x] Confirm April mean PM2.5 ≈ 22.1 µg/m³ — use in loss weighting (weight = 0.8× global mean)
- [x] Confirm PBLH is highest in April — apply separate PBLH normalisation check per season
- [x] Identify dust-belt spatial pattern (Rajasthan, Gujarat high; South India clean)
- [x] Verify NMVOC_finn occasional spikes in April — binary mask covers this correctly
- [x] Note wind variability is highest in April — validate that autocorrelation drops faster
- [ ] **Quick Win:** Plot April vs December PM2.5 spatial maps side-by-side to confirm
      dust belt vs IGP dome pattern — confirms the model needs spatially-aware architecture


---

### 5. Primary Physical Forcings

**PBLH (r = −0.749) — The Atmospheric "Lid"**

The Planetary Boundary Layer Height is the single strongest predictor. The relationship is **non-linear and hyperbolic** — halving the PBLH roughly doubles PM2.5:

- Normal hours: mean PBLH ≈ 768 m
- Extreme pollution events (> 300 µg/m³): mean PBLH drops to **384 m** (2× collapse)
- Winter nights: PBLH can drop below 200 m, creating "nighttime lid" events

**Temperature t2 (r = −0.724) — Stability Driver**

Cold temperatures → stable atmospheric conditions → temperature inversions → pollution trapping.

- Mean temperature during extreme episodes: **10.2°C** vs. normal mean of **18.5°C**
- In December, the nighttime valley and daytime peak in PM2.5 differ by nearly **3×**
- In July (monsoon), cloud cover and rain flatten this diurnal cycle significantly

**Feature Correlation Summary:**

| Feature | Global r | IGP r | South India r | Notes |
|---|---|---|---|---|
| `pblh` | −0.749 | −0.75 | −0.35 | Strongest; non-linear |
| `t2` | −0.724 | −0.72 | −0.20 | Seasonal phase shift |
| `q2` | +0.32 | **+0.45** | **−0.25** | Sign flips by region! |
| `u10/v10` | −0.14/−0.23 | Variable | Variable | Direction > speed |
| `rain` | +0.019 | ~0 | −0.55 | 70% zeros; threshold effect |
| `swdown` | −0.38 | −0.45 | −0.25 | Solar proxy for PBLH |

---

### 6. Transport & Scavenging Dynamics

**Wind (u10, v10):**

The low global correlation of wind with PM2.5 (−0.14 to −0.23) is **not** a sign wind is unimportant. It reflects the *directionality problem*: a strong wind from a clean ocean lowers PM2.5; the same speed from an industrial zone raises it. This is why a spatial model (FNO/CNN) is required — it must "see" what is upwind.

- **July:** Strong SW monsoon winds disperse pollution across the IGP — mean PM2.5 drops to 15.4 µg/m³.
- **December:** Weak, variable winds create stagnation zones — pollution pools against the Himalayan wall.

**Rainfall (rain) — The Washout Switch:**

Rain acts as a **threshold switch**, not a linear predictor:
- If `rain == 0`: no scavenging effect
- If `rain > 0`: PM2.5 drops precipitously (wet deposition)

This threshold behaviour is why the global correlation is near zero (+0.019) despite rain being a critical physical driver. Standard linear normalisation fails to capture this — use a **binary mask channel** alongside the log-transformed value.

**Spatial Correlation Decay:**

PM2.5 spatial autocorrelation decays to r = 0.5 at approximately **275 km (~11 grid cells)**. This physically reflects how far wind can transport a typical pollution plume in a few hours, and sets the **minimum receptive field** requirement for any model architecture.

---

### 7. Emission Inventory Mechanics

**The Scale Problem:**

Emission features range from 10⁻⁸ to 10⁻¹¹. With standard neural network initialisation, these values are effectively zero. If not explicitly rescaled, the model will never "see" the difference between a high-emission industrial zone and a clean rural forest.

> ⚠️ **Epsilon Trap:** `log(1 + 1e-11) ≈ 1e-11 ≈ 0` in 32-bit floating point. **Never use `log1p` for emission features.** Use `log(x + 1e-12)` instead and verify the resulting distribution has non-collapsed variance per grid cell.

**Emissions as Static Spatial Priors:**

Within any given month, emission arrays are **nearly constant** — they do not fluctuate hour-by-hour like wind or temperature. Physically, they act as a *static map* of where factories, highways, and farms are located. The model learns: "If the wind is blowing from the NW and there is high NOx at Grid (X, Y), expect a PM2.5 spike downwind."

**Multicollinearity:**

PM25, NOx, and SO2 have r > 0.97 because they are all mapped from the same industrial inventories. Using all three adds noise without signal. **Recommended approach: keep PM25 and NOx; drop SO2.**

| Feature Pair | Correlation | Action |
|---|---|---|
| NOx ↔ SO2 | r = 0.99 | Drop SO2 |
| PM25 ↔ NOx | r = 0.98 | Keep both (Tier 3) |
| NH3 ↔ NOx | r = 0.42 | Keep NH3 (unique signal) |
| NMVOC_finn | 98.4% zeros | Binary mask only |

**NMVOC_finn (Biomass Burning):**

This feature is 98.4% zeros across the year. It only carries signal during the October/November stubble burning season in Punjab and Haryana — but during those events it is an extremely important trigger. Use a **binary mask channel**: 1 if `NMVOC_finn > 0`, else 0.

---

### 8. Spatio-Temporal Rhythms

**The 24-Hour Diurnal Heartbeat:**

Every season shows a powerful cyclic pattern tied to the solar day:

1. Sunrise → ground heats up → PBLH expands upward → PM2.5 **diluted** (daytime valley)
2. Sunset → PBLH collapses → fresh emissions trapped in shallow "nighttime lid" → PM2.5 **spikes**

Seasonal amplitude: In December, the difference between nighttime peak and daytime valley is nearly **3×**. In July (monsoon), cloud cover and rain flatten this cycle significantly.

**Sinusoidal Time Encoding is Mandatory:**

Raw hour numbers (0–23) are dangerous because 23 and 0 are physically adjacent but numerically distant. Use:

```python
time_sin = np.sin(2 * np.pi * hour / 24)
time_cos = np.cos(2 * np.pi * hour / 24)
```

This maps time onto a circle, ensuring midnight and 1 AM are treated as adjacent states.

**Partial Autocorrelation (PACF) Key Findings:**

- Lags 1–3: Provide "plume momentum" — the near-term trajectory
- Lag 24: Provides the "diurnal anchor" — what the atmosphere looked like at the same time yesterday
- Lags 6–18: Provide nearly **zero unique information** — the model must rely on meteorological features here

---

### 9. Geographic Non-Stationarity

India is not a uniform block. Feature correlations with PM2.5 change sign depending on the region — a phenomenon called **Geographic Non-Stationarity**.

**Critical example — Specific Humidity (q2):**
- **IGP (North India):** Higher humidity → **MORE** PM2.5 (moisture helps NOx/SO2 form secondary particles) → **positive correlation (+0.45)**
- **South Indian Coast:** Higher humidity → **LESS** PM2.5 (associated with clean sea breezes) → **negative correlation (−0.25)**

A global model that averages these effects will produce systematically wrong predictions in both regions simultaneously.

**Hotspot Persistence:**

The top 10 most polluted grid cells are **geographically static** — Grid (93, 39) is consistently a peak across all months. These "hidden features" (power plants, urban density, industrial clusters) are not in the 16 variables but are encoded in the grid coordinates themselves.

> **Architecture requirement:** A U-Net (for local spatial detail) or FNO (for global IGP-scale modes) is needed to handle this geographic variability. A model that treats all pixels equally will fail.

---

### 10. Normalization Discrepancy

**⚠️ DO NOT USE `feat_min_max.mat`**

The official normalisation file contains pre-computed bounds that are **not representative** of the 2016/2017 data:

| Feature | Official Max | Actual Max | Excess |
|---|---|---|---|
| `cpm25` | 1,465.2 µg/m³ | **2,849.7 µg/m³** | +95% |
| `rain` | 96.6 mm | **397.0 mm** | +311% |
| `swdown` | 850 W/m² | ~920 W/m² | +8% |
| `q2` | 0.025 kg/kg | 0.028 kg/kg | +12% |
| `NOx/SO2` | 6–8e-9 | 1.1e-8 | ~38% |

**Consequence:** Using official bounds clips 2.45% of the test set's cpm25 values — the most extreme events, which are exactly what the competition judges. These events will be numerically erased.

**Correct Normalisation Pipeline:**

```python
import numpy as np

class GridWiseLogScaler:
    """
    Applies log(1+x) then per-pixel Z-score normalisation.
    Stores 17,360 individual (mean, std) pairs.
    """
    def __init__(self, epsilon=None):
        self.epsilon = epsilon   # None → use log1p; float → use log(x+epsilon)
        self.mean = None         # shape: (140, 124)
        self.std  = None         # shape: (140, 124)

    def fit(self, X):
        """X shape: (T, 140, 124)"""
        if self.epsilon is not None:
            X_log = np.log(X + self.epsilon)
        else:
            X_log = np.log1p(X)
        self.mean = X_log.mean(axis=0)              # (140, 124)
        self.std  = X_log.std(axis=0) + 1e-8        # (140, 124)
        return self

    def transform(self, X):
        if self.epsilon is not None:
            X_log = np.log(X + self.epsilon)
        else:
            X_log = np.log1p(X)
        return (X_log - self.mean) / self.std

    def inverse_transform(self, Z):
        X_log = Z * self.std + self.mean
        if self.epsilon is not None:
            return np.exp(X_log) - self.epsilon
        else:
            return np.expm1(X_log)

# Usage:
# - For meteorological features: GridWiseLogScaler(epsilon=None)
# - For emission features (1e-11 scale): GridWiseLogScaler(epsilon=1e-12)
```

---

### 11. Test Set Intelligence

The 2017 test set is not a random slice of the year — it is heavily concentrated in the most difficult meteorological regime.

**Inferred Seasonal Composition (via Temperature Fingerprinting):**

| Season | Samples | % of Test | Notes |
|---|---|---|---|
| December (Winter) | ~746 | **74.9%** | Dominant — highest RMSE difficulty |
| October (Post-M) | ~240 | **24.1%** | Secondary |
| April (Summer) | ~10 | 1.0% | Negligible |
| July (Monsoon) | 0 | **0.0%** | Completely absent from test! |

> **Critical implication:** If you validate on July data, you will get a false sense of security. Your model's true performance will be determined almost entirely by how well it handles December winter conditions.

**Statistical Shift in Test Set:**

- The test set's temperature (t2) shows a **−31.6% shift in standard deviation** vs. training data. The 2017 winter appears "steadier" than 2016.
- Global means are similar (~33 µg/m³) but variance is more unimodal in the test set.
- Q-Q plots confirm good distribution matching up to the 90th percentile.

**Out-of-Bounds Collision:**

- ~2.45% of test cpm25 input values fall **outside** the official `.mat` bounds — these are the cleanest air samples, and they will be numerically erased by min-max clipping.

---

### 12. Persistence Benchmark

The persistence model predicts that the next 16 hours will equal the PM2.5 concentration at Hour 10. This is the "zero-effort" baseline.

| Metric | Value |
|---|---|
| Global Persistence RMSE (t+16) | **30.83 µg/m³** |
| December Persistence (t+16) | **47.8 µg/m³** |
| July Persistence (t+16) | 14.4 µg/m³ |
| Normalised Persistence Floor | 0.0208 |
| Rank-1 Target RMSE | **< 21.6 µg/m³** |

**The Plateau Effect:**

RMSE grows rapidly from Step 1 (~6 µg/m³) to Step 10 (~28 µg/m³), then **plateaus** between steps 12–16 (30.1 → 30.8). This confirms that after ~12 hours, the "momentum" of the initial pollution state is almost entirely gone. The model must switch entirely to meteorological forcing.

> This is the primary motivation for the **Residual Head**: predict Δ(PM2.5) relative to the last known value at t=10, not the absolute concentration. The first 30 µg/m³ of error is "free" information from persistence — force the model to focus on the *change* caused by wind and chemistry.

---

## 🏗️ Model Architecture & Rank-1 Strategy

### Preprocessing Pipeline

```python
# Step 1 — NEVER use feat_min_max.mat
# Step 2 — Log-transform all features
# Step 3 — Grid-wise Z-score normalisation (17,360 scalers)
# Step 4 — Add binary masks for sparse features
# Step 5 — Add sinusoidal time embeddings

def preprocess(features: dict, scalers: dict, hour: np.ndarray):
    processed = {}
    
    for name, data in features.items():
        scaler = scalers[name]
        
        if name in ["NMVOC_finn", "rain"]:
            # Add binary mask channel
            mask = (data > 0).astype(np.float32)
            processed[f"{name}_mask"] = mask
        
        processed[name] = scaler.transform(data)
    
    # Sinusoidal time embeddings
    processed["time_sin"] = np.sin(2 * np.pi * hour / 24)
    processed["time_cos"] = np.cos(2 * np.pi * hour / 24)
    
    return processed
```

### Architecture: Tucker-FrNO

The backbone is a **Tucker-decomposed Fourier Neural Operator (FrNO)** with four horizon-specific output heads.

```
Input Channels:
  - cpm25: (B, 10, 140, 124)         ← 10-hr PM2.5 history
  - Met features: (B, 26, 140, 124)  ← Full 26-hr forcing (×8)
  - Emission features: (B, 1, 140, 124) ← Static spatial prior (×3)
  - Binary masks: (B, 26, 140, 124)  ← rain, NMVOC_finn
  - Time embeddings: (B, 26, 2)      ← sin/cos per timestep

Tucker-FrNO Backbone:
  → Lifting layer (channel projection)
  → 4× FNO Blocks with Tucker-decomposed spectral convolutions
     - n_modes: determined by 2D FFT 95% energy cutoff (~16-24)
  → Residual connections at each block
  
Residual Persistence Gate:
  → Subtract cpm25[t=9] from output (predict Δ, not absolute value)
  
4× Horizon-Specific Output Heads:
  → Head 1: t+1  to t+4   (high-frequency Fourier modes)
  → Head 2: t+5  to t+8   
  → Head 3: t+9  to t+12  
  → Head 4: t+13 to t+16  (low-frequency global modes)
  
Final Output:
  → Softplus activation (ensures PM2.5 ≥ 0)
  → Shape: (B, 140, 124, 16)
```

**Why Tucker decomposition?**  
Tucker factorisation reduces the spectral weight tensors from O(n_modes⁴) to O(n_modes²), making the FNO computationally feasible for the 140 × 124 grid within Kaggle's 12-hour runtime limit.

**Why 4 output heads?**  
The RMSE plateau analysis (Section 12) shows that the physics of hour t+1 (momentum-driven) is fundamentally different from hour t+16 (weather-driven). Separate heads allow the model to use high-frequency Fourier modes for near-term prediction and lower-frequency global modes for long-term forecasting.

### Data Strategy

**Temporal Firewall (prevents leakage):**

```python
# WRONG — adjacent windows create temporal leakage
train_months = ["APRIL_16", "JULY_16", "OCT_16"]
val_month    = "DEC_16"
# If train ends at hour H and val starts at H+1, the model memorises the transition

# CORRECT — enforce a 12-hour gap
def split_with_firewall(month_data, val_start_hour, gap_hours=12):
    train_end   = val_start_hour - gap_hours
    val_start   = val_start_hour
    return month_data[:train_end], month_data[val_start:]
```

**Intensity-Based Importance Sampling:**

```python
from torch.utils.data import WeightedRandomSampler

def compute_sample_weights(windows: np.ndarray) -> np.ndarray:
    """
    Upweight high-pollution windows so December episodes
    are seen proportionally more during training.
    windows shape: (N, 26, 140, 124)
    """
    global_mean = windows.mean()
    window_means = windows[:, :10].mean(axis=(1, 2, 3))  # mean over spatial + time
    weights = (window_means / global_mean) ** 2           # squared intensity
    return weights / weights.sum()

sampler = WeightedRandomSampler(
    weights=compute_sample_weights(train_windows),
    num_samples=len(train_windows),
    replacement=True,
)
```

**Recommended Train/Val Split:**

| Split | Months Used | Rationale |
|---|---|---|
| Training | APRIL_16, JULY_16, OCT_16 + first ~80% of DEC_16 | Diverse seasons |
| Validation | Last ~20% of DEC_16 (with 12-hr firewall) | Matches test distribution |
| Alternative Val | OCT_16 last 20% | Second-best alignment |
| **Never use** | JUL_16 as primary validation | 0% of test is July |

### Loss Function

```python
import torch
import torch.nn as nn

class SpatiallyWeightedResiduaLoss(nn.Module):
    def __init__(self, spatial_weights: torch.Tensor):
        """
        spatial_weights: (140, 124) tensor
        Higher weights on IGP hotspot cells, lower on clean southern regions.
        Computed from per-pixel persistence RMSE at t+16 on training data.
        """
        super().__init__()
        self.register_buffer("weights", spatial_weights)

    def forward(self, pred_delta, true_delta):
        """
        pred_delta: (B, 140, 124, 16) — predicted Δ from t=9
        true_delta: (B, 140, 124, 16) — actual Δ from t=9
        """
        sq_err = (pred_delta - true_delta) ** 2          # (B, 140, 124, 16)
        weighted = sq_err * self.weights.unsqueeze(0).unsqueeze(-1)
        return weighted.mean()
```

### Inference & TTA

```python
def predict_with_tta(model, test_inputs, scalers):
    """Test-Time Augmentation: average H-flip + V-flip predictions."""
    
    def run(inputs):
        with torch.no_grad():
            pred_delta = model(inputs)
        return pred_delta
    
    # Original
    p_orig  = run(test_inputs)
    
    # Horizontal flip (longitude axis)
    p_hflip = run(flip_inputs(test_inputs, axis="lon")).flip(dims=[-2])
    
    # Vertical flip (latitude axis)
    p_vflip = run(flip_inputs(test_inputs, axis="lat")).flip(dims=[-3])
    
    # Average
    pred_delta = (p_orig + p_hflip + p_vflip) / 3
    
    # Add back persistence (cpm25 at t=9) and inverse-transform
    cpm25_t9   = test_inputs["cpm25"][:, -1:, :, :]          # (B, 1, 140, 124)
    pred_norm  = pred_delta + cpm25_t9.unsqueeze(-1)
    pred_raw   = scalers["cpm25"].inverse_transform(pred_norm) # denormalise
    
    return pred_raw   # shape: (B, 140, 124, 16)
```

---

## ⚠️ Critical Missing Safeguards

These 8 components are **absent from the naive baseline** and will silently degrade model performance without throwing errors:

| # | Safeguard | Silent Risk if Absent | Implementation |
|---|---|---|---|
| 1 | **9-hr Temporal Firewall** | Temporal leakage — model memorises validation | 12-hr gap between all train/val windows |
| 2 | **Multi-Head Horizon Split** | Single head learns t+1 & t+16 physics mixed | 4 parallel heads for 4-hr blocks |
| 3 | **Intensity-Squared Sampling** | Model overfits July; catastrophically fails December | `W = (mean/global_mean)²` WeightedRandomSampler |
| 4 | **Binary Masks for Sparse Features** | Network treats `0` and `1e-11` identically | Mask channel: 1 if `rain > 0` / `NMVOC_finn > 0` |
| 5 | **Softplus Output Activation** | ReLU dead neurons near PM2.5 = 0 in South India | `log(1 + exp(x))` — smooth gradient everywhere |
| 6 | **2D FFT Spectral Check** | `n_modes` hyperparameter set blindly | Plot 2D power spectrum; find 95% energy cutoff |
| 7 | **Epsilon for Emissions** | `log(1+1e-11) = 0` in float32 — emission map invisible | `log(x + 1e-12)`; verify non-zero std per pixel |
| 8 | **Spatial Loss Weighting** | Equal penalty → under-training on IGP hotspot cells | Higher weight on top-10 most polluted grid cells |

---

## ✅ EDA Completion Checklist

```
DONE  [✓] Seasonal statistics — December dominates variance (27%)
DONE  [✓] Target distribution — Max 2,849 µg/m³; heavy right tail confirmed
DONE  [✓] Autocorrelation — 9-hr independence gap found
DONE  [✓] Test set fingerprinting — 74.9% December; 0% July
DONE  [✓] Normalization audit — Official .mat file invalid (2.45% OOB)
DONE  [✓] Emission collinearity — PM25/NOx/SO2 r > 0.97 (redundant trio)
DONE  [✓] Spatial hotspot analysis — Top-10 cells static in IGP
DONE  [✓] Feature importance — PBLH & t2 are Tier-1 features
DONE  [✓] Temporal autocorrelation — Persistence plateau at step 12
DONE  [✓] Geographic non-stationarity — q2 correlation flips sign by region

QUICK [⚡] 2D FFT spectral check — run on Dec PM2.5 map → pick n_modes (~30 min)
QUICK [⚡] Epsilon stability — compare log1p vs log(x+1e-12) for emissions (~30 min)
QUICK [⚡] Spatial RMSE heatmap — visualise per-pixel t+16 persistence error (~30 min)
```

> **Verdict:** EDA is 98% complete. 3 quick wins remain (≈ 90 min total). All findings converge on one strategy. **Move immediately to implementation.**

---

## 📤 Submission Format

Your final `preds.npy` must have **exactly** this shape:

```python
import numpy as np

preds = model.predict(test_inputs)   # your inference pipeline
assert preds.shape == (996, 140, 124, 16), f"Wrong shape: {preds.shape}"

np.save("/kaggle/working/preds.npy", preds)
print("Submission saved.")
```

| Dimension | Size | Meaning |
|---|---|---|
| 0 | 996 | Number of test samples |
| 1 | 140 | Latitude grid (rows) |
| 2 | 124 | Longitude grid (columns) |
| 3 | 16 | Forecast horizon (hours t+1 to t+16) |

> ⚠️ If file name, file location, or array shape do not match exactly, the submission will **not be evaluated** and no leaderboard score will be generated.

**Submission checklist:**
- [ ] File name: exactly `preds.npy`
- [ ] File location: `/kaggle/working/preds.npy`
- [ ] Array shape: `(996, 140, 124, 16)`
- [ ] Array dtype: `float32`
- [ ] No NaN or Inf values: `assert not np.isnan(preds).any()`
- [ ] Values physically reasonable: `assert preds.min() >= 0`
- [ ] Notebook link included in submission description
- [ ] Notebook shared with competition hosts (rahulsundar, sanchitbedi, siddharthandileep)

---

## 📚 References

- [Competition Page — ANRF AISEHack Theme 2](https://www.kaggle.com/competitions/aisehack-theme-2)
- [Baseline Code Repository](https://github.com/vaasew/baseline_anrf)
- [Baseline Notebook](https://www.kaggle.com/code/siddharthandileep/baseline-run-aisehack-test/)
- [Helper Notebook](https://www.kaggle.com/code/siddharthandileep/helper-notebook/)
- [Average Domain RMSE Definition](https://www.kaggle.com/code/siddharthandileep/average-domain-rmse)
- [Fourier Neural Operator Paper](https://arxiv.org/pdf/2512.01421)
- Li et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- WRF-Chem Model Documentation — [https://www2.mmm.ucar.edu/wrf/users/](https://www2.mmm.ucar.edu/wrf/users/)

---

<div align="center">

**Built for ANRF AISEHack Theme 2 | National Climate AI Hackathon**  
*Protecting 1.4 billion citizens through better air quality forecasting*

</div>
