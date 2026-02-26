# Predicting-Energy-Consumption
# ⚡ Energy Consumption Forecasting
### Machine Learning Pipeline for Hourly Electricity Demand Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2EC4B6?style=for-the-badge)

**R² = 0.963 &nbsp;|&nbsp; MAPE = 1.91% &nbsp;|&nbsp; Industry Benchmark < 5% ✅**

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Problem Matters](#-why-this-problem-matters)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Pipeline Blocks Explained](#-pipeline-blocks-explained)
6. [Why These Models?](#-why-these-models)
7. [Why These Features?](#-why-these-features)
8. [Audit: Leakage Prevention](#-audit-leakage-prevention)
9. [Results](#-results)
10. [Visualisations](#-visualisations)
11. [How to Run](#-how-to-run)
12. [Key Learnings](#-key-learnings)

---

## 🎯 Project Overview

This project builds an end-to-end machine learning pipeline to **forecast hourly electrical energy consumption** using 6 years of historical data from a transmission system operator.

The goal is not just to build a model — it is to build a **trustworthy, explainable, audit-proof** pipeline that a real energy company could use in production. Every single decision (model choice, feature design, preprocessing step, evaluation method) is justified with a clear reason.

---

## ⚡ Why This Problem Matters

Electricity is unique among commodities: **it cannot be stored at scale**. Supply must match demand in real time — every second. This makes accurate forecasting critical:

| Domain | Impact |
|--------|--------|
| **Grid Stability** | Operators must pre-schedule generation hours ahead. A 5% demand spike without warning can cause frequency deviations or blackouts. |
| **Cost Reduction** | Utilities trade power in spot markets 24–48 hours ahead. A 1% forecast improvement saves millions per year in balancing costs. |
| **Renewable Integration** | Wind and solar output is variable. Pairing accurate load forecasts with generation forecasts enables maximum clean energy usage. |
| **Sustainability** | Gas "peaking plants" idle on hot standby for unexpected spikes. Better forecasts reduce their runtime, directly lowering CO₂ emissions. |
| **Infrastructure Planning** | Long-term demand trends guide billion-dollar decisions on grid upgrades and new generation capacity. |

> **This project targets short-term (hourly) forecasting** — the most operationally critical horizon for grid operators.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Transmission system operator (anonymised) |
| **File** | `project_power.csv` |
| **Records** | 52,966 hourly rows |
| **Date Range** | Dec 31, 2015 → Dec 31, 2021 (6 years) |
| **Columns** | `Start time UTC`, `End time UTC`, `Electricity consumption (MWh)` |
| **Missing Values** | 0 |
| **Outliers** | 0 (seasonal peaks are genuine, not errors) |
| **Duplicates** | 358 removed (DST clock-back hours) |

### Dataset Quality Verdict

The dataset is **exceptionally clean for a real-world industrial source**. The only issue found was 358 duplicate timestamps caused by Daylight Saving Time (DST) clock-back events — these are handled by keeping the first occurrence.

### Why No Outlier Removal?

The IQR-based outlier check (3×IQR threshold) found zero anomalous points. All extreme values represent genuine physical phenomena:
- **Winter peaks**: Heating demand in cold months
- **Summer peaks**: Cooling/air-conditioning demand in hot months

Removing these would **degrade model performance** by hiding real seasonal patterns the model needs to learn.

---

## 📁 Project Structure

```
energy-consumption-forecasting/
│
├── 📄 README.md                          ← You are here
├── 📊 project_power.csv                  ← Raw dataset
│
├── 🐍 full_enhanced_solution.py          ← MAIN script (all 10 blocks)
├── 🐍 model_shootout.py                  ← 3-model comparison only
├── 🐍 energy_consumption_audit_proof.py  ← Leakage audit version
│
├── 📈 Visualisations/
│   ├── block3_baseline_comparison.png    ← MAE/RMSE/R²/MAPE bars
│   ├── block4_cross_validation.png       ← CV scores per fold
│   ├── block5_hyperparameter_tuning.png  ← Tuning search results
│   ├── block6_tuned_vs_baseline.png      ← Tuned vs baseline
│   ├── block7_permutation_importance.png ← Feature importance
│   ├── block8_error_analysis.png         ← Error deep dive
│   └── block9_future_forecast.png        ← 7-day forward forecast
│
└── 📋 7day_forecast.csv                  ← Exported forecast values
```

---

## 🧱 Pipeline Blocks Explained

The entire solution is organised into **10 sequential blocks**. Each block has a clear purpose:

### Block 1 — Data Load, Audit & Preprocessing
**Why?** Before any modelling, we must verify the data is trustworthy. A model trained on bad data produces confidently wrong predictions — *"garbage in, garbage out"*. This block runs 6 automated quality checks and documents every finding.

**What it does:**
- Loads CSV and parses timestamps
- Removes DST duplicate timestamps
- Checks for missing values, negatives, outliers, temporal gaps
- **Critically: splits BEFORE filling gaps** (see Audit section)

---

### Block 2 — Feature Engineering
**Why?** The raw dataset has only 1 signal column. ML models work on numbers, not timestamps. All temporal patterns must be manually extracted.

**What it does:** Creates 20 features across 4 categories (see [Why These Features?](#-why-these-features))

---

### Block 3 — 3-Model Baseline Shootout
**Why?** We never pick a model based on intuition alone. All 3 candidates are trained on **identical data** and compared empirically. This is the scientific approach to model selection.

---

### Block 4 — Cross-Validation (TimeSeriesSplit)
**Why?** A single train/test split gives **one** performance estimate. What if that split was lucky? Cross-validation evaluates the model on **5 different time windows** to verify consistent performance.

**Why `TimeSeriesSplit` and NOT regular `k-fold`?**

Regular k-fold shuffles data → temporal leakage (future data leaks into training set). TimeSeriesSplit always keeps the test window **after** the training window:

```
Fold 1: [━━━Train━━━] [Test]
Fold 2: [━━━━━━Train━━━━━━] [Test]
Fold 3: [━━━━━━━━━━Train━━━━━━━━━] [Test]
Fold 4: [━━━━━━━━━━━━━━Train━━━━━━━━━━━━] [Test]
Fold 5: [━━━━━━━━━━━━━━━━━━Train━━━━━━━━━━━━━━] [Test]
```

---

### Block 5 — Hyperparameter Tuning (RandomizedSearchCV)
**Why?** Our baseline used manually chosen hyperparameters. Are those the best? Systematic search finds better configurations automatically.

**Why `RandomizedSearchCV` and NOT `GridSearchCV`?**

GridSearchCV tries **every** combination — with 5 parameters × 5 values each = 3,125 model fits with 5-fold CV. Too slow.

RandomizedSearchCV samples **N random combinations** from the parameter space. With `n_iter=20` + 3-fold CV = only **60 model fits**. Finds ~95% of the optimal solution in ~5% of the time.

---

### Block 6 — Tuned Model Evaluation
**Why?** Compares the tuned model against all 3 baselines on the true held-out test set. Answers: *"Did tuning actually help?"*

---

### Block 7 — Permutation Importance
**Why?** Built-in tree feature importance can be misleading. A feature used on many splits may not actually improve predictions — it could just be correlated with a better feature.

**How Permutation Importance works:**
1. Take the trained model and test set
2. For feature X: randomly **shuffle** column X (breaking its relationship with the target)
3. Measure how much accuracy **drops**
4. Large drop → feature was critical. Small drop → feature was not actually useful.

This tells us what the model **actually relies on** vs what it happened to split on during training.

---

### Block 8 — Error Analysis & Confidence Bands
**Why?** A single MAE number hides a lot. We need to know:
- **When** does the model fail? (by hour, day)
- **How bad** are the worst errors? (tail risk for operations)
- **Is the model biased?** (consistently over/under-predicting)
- **Confidence bands** — how uncertain are predictions?

Confidence bands are built empirically: we estimate how errors grow over the forecast horizon and shade a ±1.5σ band around predictions.

---

### Block 9 — Future Forecast (7-Day Rolling)
**Why?** All previous blocks evaluate on historical data (where we know the answer). This block demonstrates real deployment: **predicting future hours that have never been seen**.

**How Rolling Forecast works:**
```
Hour 1: predict using last known actuals for all lags
Hour 2: use Hour 1 PREDICTION as lag input (no actual available)
Hour N: accumulate previous predictions as lag features
```
This is called an **autoregressive** (recursive) forecast. Error compounds over time — shorter horizons are more accurate. The uncertainty band widens proportional to √(hours ahead).

---

### Block 10 — Summary Scorecard
Final comparison table of all models + file manifest.

---

## 🤖 Why These Models?

All three models are **tree-based ensembles**. This choice is justified because:
- ✅ Handle non-linear relationships (energy is not linear with time)
- ✅ Capture feature interactions (hot summer + Monday morning ≠ cold winter + Monday)
- ✅ Work with mixed feature types (calendar + lag + rolling)
- ✅ No feature scaling required (trees are scale-invariant)
- ✅ Built-in feature importance for interpretability

---

### Model 1: XGBoost (HistGradientBoostingRegressor) 🥇

**Algorithm:** Sequential gradient boosting with **histogram-based splitting**

**The core idea:**
```
Round 1: Train a shallow tree → makes some errors
Round 2: Train a NEW tree focused ONLY on Round 1's errors
Round 3: Fix errors from Round 1+2 combined
... repeat 500 times with learning_rate=0.05
```
Each tree is a "specialist corrector". The final prediction is the **sum of all corrections**.

**Why histograms?** Instead of checking every possible split value exactly (O(n) per feature), XGBoost bins values into ~256 buckets (O(1) lookup). This gives **10-100× faster training** vs sklearn's `GradientBoostingRegressor` with near-identical accuracy.

**Key hyperparameters justified:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_iter` | 500 | More rounds → finer error correction. Early stopping prevents overfitting. |
| `learning_rate` | 0.05 | Small steps = careful improvement. High lr (0.3) reaches a local minimum fast and overfits. |
| `max_depth` | 6 | Medium depth captures complex seasonal interactions. Deeper = more overfitting risk. |
| `min_samples_leaf` | 20 | Each leaf needs ≥20 samples. Prevents overfitting on rare edge-case hours. |
| `l2_regularization` | 0.1 | Penalises large leaf weights. Unique to XGBoost-style models — key generalisation advantage. |

**L2 Regularisation (the secret weapon):**
```
Leaf weight formula: w = -gradient / (hessian + λ)
                                              ↑
                                    λ = l2_regularization
```
This dampens extreme predictions and is the primary reason XGBoost-style models dominate forecasting leaderboards.

---

### Model 2: Random Forest 🥈

**Algorithm:** Parallel **bagging** of 500 independent decision trees

**The core idea:** Train 500 trees independently on random subsets of data. Each tree votes → average the votes. No tree communicates with or learns from another.

**Why it's good but not the best:**
- ✅ Naturally parallelisable (`n_jobs=-1`)
- ✅ Very robust to noisy data
- ❌ **No directed error correction** — trees are blind to others' mistakes
- ❌ Higher variance than boosting on structured time-series patterns

**Key hyperparameters justified:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `n_estimators` | 500 | More trees = lower variance (diminishing returns above ~300-500) |
| `max_depth` | 15 | Deeper trees are safe here — averaging 500 trees cancels individual overfitting |
| `max_features` | `'sqrt'` | Each split sees √20 ≈ 4 features. Creates diverse, decorrelated trees. If all trees see same features, they all make same splits — defeating the purpose. |
| `min_samples_leaf` | 5 | Smaller leaves than boosting: each tree can be more expressive since averaging corrects mistakes |

---

### Model 3: Gradient Boosting (sklearn) 🥉

**Algorithm:** Sequential boosting with **exact split search**

Same conceptual approach as XGBoost but uses exact (non-histogram) split finding. This means:
- ✅ More interpretable (exact splits)
- ❌ ~10× slower than XGBoost
- ❌ Less regularisation options

Used as the **gold standard baseline** that XGBoost is compared against.

---

### Why NOT Neural Networks / LSTM?

| Reason | Detail |
|--------|--------|
| **Data size** | LSTMs need millions of rows to beat well-engineered tabular features. Our 52K rows is borderline. |
| **Interpretability** | LSTMs are black boxes. Tree models give feature importances, which is critical for energy management decisions. |
| **Training time** | LSTM training is 10-100× slower for no guaranteed accuracy gain. |
| **Feature engineering** | With strong lag features already capturing temporal structure, the LSTM has no remaining advantage to exploit. |

---

## 🔧 Why These Features?

The raw dataset has **only 1 signal column** (`consumption_mwh`). All 20 features are derived from the timestamp and the consumption history itself.

### Feature Group 1: Calendar Features
```python
hour, dayofweek, month, year, quarter, dayofyear, weekofyear, is_weekend
```
**Why:** Energy demand follows predictable calendar patterns. 8 AM Monday has a completely different demand profile than 8 AM Saturday or 8 AM in July vs January.

---

### Feature Group 2: Cyclic Encoding (sin/cos)
```python
hour_sin  = sin(2π × hour / 24)
hour_cos  = cos(2π × hour / 24)
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
dow_sin   = sin(2π × dayofweek / 7)
dow_cos   = cos(2π × dayofweek / 7)
```
**Why — the ordinal distance problem:**

If we feed `hour = 23` and `hour = 0` (midnight) directly to a tree model, it treats them as 23 units apart. But they are only **1 hour apart** in reality.

Sin/cos encoding maps the circle correctly:
- hour=23 → (sin≈-0.26, cos≈0.97)
- hour=0  → (sin=0.00, cos=1.00)

These vectors are close together. The model correctly learns that 11 PM and midnight are similar.

---

### Feature Group 3: Lag Features
```python
lag_24  = consumption at t-24h  (same hour yesterday)
lag_48  = consumption at t-48h  (same hour 2 days ago)
lag_168 = consumption at t-168h (same hour last week)
```
**Why:** Energy consumption has extremely strong 24-hour autocorrelation. If demand was high at 8 AM yesterday, it will likely be high at 8 AM today (same commute patterns, same industrial schedules). The `lag_168` captures the weekly cycle — Monday always resembles last Monday more than it resembles last Saturday.

**Verified empirically:** The autocorrelation plot shows spikes at exactly 24h and 48h, confirming this intuition.

**Why these are safe (no leakage):**
```python
lag_24 = df['consumption_mwh'].shift(24)
# At time t, lag_24 = value at t-24 (yesterday)
# We are predicting t — we DO have yesterday's data ✅
```

---

### Feature Group 4: Rolling Statistics
```python
rolling_mean_24  = mean of previous 24 hours
rolling_std_24   = std dev of previous 24 hours  
rolling_mean_168 = mean of previous 168 hours (1 week)
```
**Why:** Captures recent momentum and volatility. If consumption has been trending up for 24 hours, it's likely to remain elevated. The rolling std tells the model how stable or volatile recent demand has been.

**Critical: `.shift(1)` before every rolling operation:**
```python
# WRONG — includes current hour in its own window (leakage ❌)
rolling_mean = df['consumption_mwh'].rolling(24).mean()

# CORRECT — window ends 1 hour before current (✅)
rolling_mean = df['consumption_mwh'].shift(1).rolling(24).mean()
```

---

## 🔒 Audit: Leakage Prevention

Data leakage is the #1 silent killer of ML models. The model appears accurate in testing but fails in production because it cheated during training.

### Check 1 ✅ — No Random Split (Strict Time Cutoff)

**The trap:**
```python
# WRONG for time series (temporal leakage ❌)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data, test_size=0.2, shuffle=True)
# Model trains on Wednesday to predict Tuesday → sees the future!
```

**Our solution:**
```python
# CORRECT — strict temporal boundary ✅
SPLIT_DATE = pd.Timestamp('2021-10-01')
train = data[data['start_utc'] <  SPLIT_DATE]
test  = data[data['start_utc'] >= SPLIT_DATE]
# Every test prediction is truly "blind" — model never saw Oct-Dec 2021
assert train['start_utc'].max() < test['start_utc'].min()
```

---

### Check 2 ✅ — No Rolling/Lag Feature Leakage

**The trap:**
```python
# WRONG — current hour is inside its own rolling window (leakage ❌)
df['rolling_mean'] = df['consumption_mwh'].rolling(24).mean()
# At 10:00 AM, window includes 10:00 AM itself — we're predicting it!
```

**Our solution:**
```python
# CORRECT — shift(1) ensures window ends 1 hour before current ✅
df['rolling_mean'] = df['consumption_mwh'].shift(1).rolling(24).mean()
# At 10:00 AM, window covers 10:00 AM yesterday → 09:00 AM today
```

---

### Check 3 ✅ — No Interpolation Across the Train/Test Boundary

**The trap:**
```python
# WRONG — interpolate on full dataset before splitting (leakage ❌)
df['consumption_mwh'] = df['consumption_mwh'].interpolate()
# A gap at Dec 31 2020 (train) gets filled using Jan 1 2021 (test)
# Future data used to construct a training feature!
```

**Our solution:**
```python
# CORRECT — split FIRST, then fill within each half independently ✅
raw_train = df[df['start_utc'] <  SPLIT_DATE].copy()
raw_test  = df[df['start_utc'] >= SPLIT_DATE].copy()

# ffill only looks backwards — never uses future values
raw_train['consumption_mwh'] = raw_train['consumption_mwh'].ffill()
raw_test['consumption_mwh']  = raw_test['consumption_mwh'].ffill()
```

**Why `ffill` is safe:** Forward-fill copies the last known value forward. It never looks ahead. `bfill` (backward-fill) and `interpolate()` both use future values and must be avoided at the boundary.

---

### Check 4 ✅ — Target Variable Not in Feature Set

```python
assert TARGET not in FEATURES, "TARGET leaked into features!"
# Crashes immediately if someone accidentally adds 'consumption_mwh' to FEATURES
```

---

## 📈 Results

### Final Model Comparison

| Rank | Model | MAE (MWh) | RMSE (MWh) | R² | MAPE |
|------|-------|-----------|------------|-----|------|
| 🥇 | **Tuned XGBoost** | **204.40** | **268.65** | **0.9635** | **1.91%** |
| 🥈 | Gradient Boosting | 217.52 | 283.65 | 0.9593 | 2.04% |
| 🥉 | Random Forest | 320.74 | 417.67 | 0.9118 | 2.89% |

**Industry benchmark: MAPE < 5% ✅ — All models pass**

### What These Numbers Mean

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE = 204 MWh** | On a mean load of 9,489 MWh | Average error is just 2.1% of mean demand |
| **MAPE = 1.91%** | vs 5% industry benchmark | Operationally excellent — suitable for grid balancing |
| **R² = 0.9635** | 96.35% variance explained | Model captures nearly all systematic patterns |
| **RMSE > MAE** | RMSE = 268 vs MAE = 204 | A small number of harder hours (holidays, anomalies) have larger errors — but these are rare |

### Cross-Validation Stability
The model was validated on 5 different time folds — not just one lucky split. The consistent MAPE across folds confirms the model is **genuinely reliable**, not just lucky.

### Error Distribution
- **90% of hours** are predicted within ~400 MWh
- **95% of hours** are predicted within ~500 MWh
- The residuals are **centred near zero** — the model is unbiased

---

## 📊 Visualisations

| File | What it shows |
|------|---------------|
| `block3_baseline_comparison.png` | MAE, RMSE, R², MAPE bars for all 3 baseline models |
| `block4_cross_validation.png` | Performance on each of 5 CV folds — checks consistency |
| `block5_hyperparameter_tuning.png` | The 20 configurations searched, learning rate vs MAE |
| `block6_tuned_vs_baseline.png` | Tuned XGBoost vs all 3 baselines |
| `block7_permutation_importance.png` | What features the model ACTUALLY relies on |
| `block8_error_analysis.png` | Error by hour, error distribution, cumulative error, confidence bands |
| `block9_future_forecast.png` | 7-day ahead forecast with widening uncertainty bands |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/energy-consumption-forecasting.git
cd energy-consumption-forecasting
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

> **Note on XGBoost:** This project uses `sklearn.ensemble.HistGradientBoostingRegressor` which implements the identical histogram-based gradient boosting algorithm as XGBoost. If you prefer the original XGBoost library:
> ```bash
> pip install xgboost
> ```
> Then replace `HistGradientBoostingRegressor` with `xgboost.XGBRegressor` in the code.

### 3. Add the dataset
Place `project_power.csv` in the project root directory. The CSV should have columns:
```
Start time UTC, End time UTC, Electricity consumption (MWh)
```

### 4. Run the full pipeline
```bash
# Full 10-block pipeline (recommended)
python full_enhanced_solution.py

# Model comparison only (faster)
python model_shootout.py

# Audit-focused version
python energy_consumption_audit_proof.py
```

### 5. Outputs
After running, the following files will be generated:
- `block3_baseline_comparison.png` through `block9_future_forecast.png`
- `7day_forecast.csv` — the 7-day ahead point forecast

---

## 💡 Key Learnings

### 1. Data Quality > Model Complexity
This dataset required almost no cleaning. The 6 audit checks confirmed it is near-perfect. In real projects, cleaning usually takes 70% of the time. A simple model on clean data beats a complex model on dirty data every time.

### 2. Feature Engineering is the Real Work
The raw dataset has 1 column. The model uses 20 engineered features. The lag features alone explain ~85% of the model's predictive power. The quality of features determines the ceiling of what any model can achieve.

### 3. Temporal Leakage is Invisible but Deadly
All 4 leakage traps (random split, rolling without shift, interpolation across boundary, target in features) would produce a model that appears excellent in evaluation but collapses in production. Building checks and assertions for these is non-negotiable.

### 4. Cross-Validation Proves Reliability
A single test split is an anecdote. Five CV folds are evidence. Always validate across multiple time windows before claiming a model is reliable.

### 5. XGBoost Wins on Structured Tabular Data
For tabular data with strong auto-correlation structure and well-engineered features, XGBoost-style models consistently outperform both simpler (Linear Regression, Random Forest) and more complex (LSTM, Transformer) approaches. The L2 regularisation and histogram-based splitting are key.

---

## 🔮 Future Improvements

| Improvement | Expected Gain | Why |
|-------------|---------------|-----|
| Add temperature data | **High** (MAPE → ~1.0%) | Demand is strongly temperature-driven. The single biggest missing feature. |
| Add public holiday flag | **Medium** | Holidays behave like Sundays — very different from weekday patterns |
| Year-over-year trend feature | **Medium** | Long-term consumption growth/decline not captured by calendar features alone |
| LightGBM or native XGBoost | **Small** | Marginal accuracy gain, significant speed gain |
| Quantile regression for intervals | **Operational** | Point forecasts are less useful than forecasts with uncertainty ranges |
| Monthly retraining pipeline | **Operational** | Consumption patterns shift over time (COVID-19, new industrial loads, EVs) |

---

## 📄 Assignment Context

This project was completed as part of an ML assignment on time-series forecasting. The assignment objective was to demonstrate:

- ✅ Understanding of the problem domain
- ✅ Correct data preprocessing methodology
- ✅ Feature engineering for time series
- ✅ Model selection with justification
- ✅ Proper evaluation (no leakage, time-based split)
- ✅ Interpretation of results and business implications

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 1.3+ | Data loading, manipulation, time-series reindexing |
| `numpy` | 1.21+ | Numerical operations, cyclic encoding |
| `scikit-learn` | 1.0+ | All ML models, CV, hyperparameter search, metrics |
| `matplotlib` | 3.4+ | All visualisations |
| `seaborn` | 0.11+ | Statistical plot styling |
| `scipy` | 1.7+ | Probability distributions for hyperparameter search |

---

<div align="center">

**Built with rigour. Every decision justified. Every assumption tested.**

*If you found this useful, please ⭐ star the repo!*

</div>
