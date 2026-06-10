# Predicting Stellar Class

Solution for the Kaggle **Playground Series S6E6 — Predicting Stellar Class** competition.
Three-class classification (`GALAXY` / `QSO` / `STAR`) of SDSS photometric objects, scored on
**balanced accuracy**.

## Approach

A stacking ensemble: several diverse base models produce out-of-fold (OOF) class probabilities,
which a **TabPFN-3** meta-learner combines together with the raw engineered features.

```
Layer 0 (base models, 5-fold OOF):  XGBoost · LightGBM · CatBoost · HistGradientBoosting ·
                                    ExtraTrees · MLP  (+ KNN / LogReg for diversity)
        |  out-of-fold probabilities (leak-free)
Layer 1 (meta):                     TabPFN-3 over [base OOF logits + engineered features]
        |
Final prediction -> submission.csv
```

### Feature engineering
- Winsorized magnitudes (`u g r i z`) to the 0.1–99.9 percentile (removes a few broken rows).
- SDSS **colors** — all adjacent and cross band differences (`u_g`, `g_r`, …, `u_z`).
- **redshift** transforms (`log1p`, square, sqrt) and `redshift × color` interactions.
- Brightness aggregates (mean / std / range of magnitudes), color spread.
- One-hot `spectral_type` and `galaxy_population`.

### Key findings
- `redshift` is by far the strongest single signal (STAR ≈ 0, GALAXY low, QSO high).
- `class_weight='balanced'` (or balanced `sample_weight`) matters a lot for the balanced-accuracy
  metric — it lifts each tree base by ~0.008–0.010.
- TabPFN-3 works best as a **meta-learner** over model predictions, not on raw photometry.
- Residual error concentrates on **low-redshift galaxies** (z ≈ 0.13) that are photometrically
  degenerate with stars — hard to separate without morphology features the dataset doesn't include.

## Results (leaderboard, balanced accuracy)

| Version | LB     | Notes                                            |
|---------|--------|--------------------------------------------------|
| v1      | 0.96577| base stack                                       |
| v2      | 0.96662| + richer features                                |
| v3      | 0.96711| + LightGBM base                                  |
| **v4**  | **0.96758** | 6 bases multi-seed -> TabPFN meta (**best**) |
| v5      | 0.96716| balanced-weighted bases (more correlated)        |
| v8      | 0.96565 | AutoGluon `best_quality` + `balance_weight`, single model, no blending |

The v8 AutoGluon single model (one `.fit()`, internally multi-level-stacked, optimizing
`balanced_accuracy` directly) reaches OOF 0.9656 — essentially tied with the hand-built v4 stack.
The `sample_weight='balance_weight'` lever lifted every booster ~+0.010 (LightGBM 0.9538 -> 0.9642),
confirming balanced weighting is decisive for this metric. It is a genuinely different, fully
reproducible single model rather than a vote-average of public submissions.

## Repository layout

```
notebook/
  model.ipynb                  # EDA + interactive development
  pipeline.py                  # base models + TabPFN meta (v4, the main pipeline)
  pipeline_v5.py               # balanced-weighted base variants
  pipeline_v6.py               # diversity push (KNN / LogReg / wide MLP + extra features)
  pipeline_v8.py               # AutoGluon best_quality + balance_weight single model
  pipeline_v7.py               # meta-of-metas (stacks all accumulated OOF)
  blender-is-all-you-need.ipynb# reference: weighted submission blender
  tabpfn-3-stacker.ipynb       # reference: TabPFN stacking notebook
README.md
.gitignore
```

## Reproduce

1. Download competition data into `data/` (`train.csv`, `test.csv`, `sample_submission.csv`)
   from the Kaggle [playground-series-s6e6](https://www.kaggle.com/competitions/playground-series-s6e6) page.
2. Install deps: `pip install numpy pandas scikit-learn xgboost lightgbm catboost torch tabpfn`.
3. TabPFN needs a one-time license: set `TABPFN_TOKEN` in a local `.env`
   (get the key + accept the license at <https://ux.priorlabs.ai>).
4. Run the pipeline:

   ```bash
   cd notebook
   set -a && source ../.env && set +a
   python pipeline.py
   ```

   Base-model OOF/test probabilities are cached under `artifacts/` so reruns skip completed work.
   The final file is written to `submission.csv` at the repo root.

> A CUDA GPU is recommended (the pipeline was developed on an RTX 4050). Base trees and the
> TabPFN meta both use the GPU; LightGBM runs on CPU (faster here on this feature count).
