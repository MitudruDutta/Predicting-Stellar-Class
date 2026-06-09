"""
v8: AutoGluon best_quality as a single honest model (no public-submission blending).

One TabularPredictor.fit() trains + internally multi-level-stacks LightGBM, CatBoost, XGBoost,
RandomForest, ExtraTrees and neural nets, optimizing eval_metric='balanced_accuracy' directly.
Far more model diversity/tuning than our hand-rolled bases, from one script.

Features: build_features_v6 (colors, redshift transforms/interactions, mag aggregates) but keep
spectral_type / galaxy_population as NATIVE categoricals (AutoGluon handles them better than one-hot).
Winsorization via pipeline.load_and_clean().

Usage:
  python pipeline_v8.py smoke   # 100k subsample, medium_quality, 10min, holdout check
  python pipeline_v8.py full    # full train, best_quality, 4h, writes submission

Honesty gate: compare OOF balanced_accuracy to v4 (0.9660). If AutoGluon plateaus ~0.967 -> STOP,
keep best single model. Never fabricate a crossing of 0.97154.
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import ROOT, log, load_and_clean, C2I, I2C, N_CLASSES
from pipeline_v6 import build_features_v6
from sklearn.metrics import balanced_accuracy_score

ART8 = ROOT / 'artifacts_v8'
ART8.mkdir(exist_ok=True)
CLASSES = ['GALAXY', 'QSO', 'STAR']

# categoricals kept native (not one-hot) for AutoGluon
CAT_COLS = ['spectral_type', 'galaxy_population']


def make_frames():
    """Engineered features + native categoricals. Returns train_df (with 'class'), test_df, test_ids."""
    tr, te = load_and_clean()
    # build_features_v6 one-hots the cats; instead replicate its numeric features but keep cats raw.
    trf = build_features_v6_no_ohe(tr)
    tef = build_features_v6_no_ohe(te)
    train_df = trf.copy()
    train_df['class'] = tr['class'].values
    return train_df, tef, te['id'].values


def build_features_v6_no_ohe(df):
    """Same numeric features as build_features_v6 but leaves categoricals as raw category dtype."""
    out = df.copy()
    bands = ['u', 'g', 'r', 'i', 'z']
    for a in range(len(bands)):
        for b in range(a + 1, len(bands)):
            out[f'{bands[a]}_{bands[b]}'] = out[bands[a]] - out[bands[b]]
    z = out['redshift']
    out['redshift_log1p'] = np.log1p(z.clip(lower=0))
    out['redshift_sq'] = z ** 2
    out['redshift_sqrt'] = np.sqrt(z.clip(lower=0))
    for col in ['u_g', 'g_r', 'r_i', 'i_z', 'u_z']:
        out[f'z_x_{col}'] = z * out[col]
        out[f'zlog_x_{col}'] = out['redshift_log1p'] * out[col]
    out['mag_mean'] = out[bands].mean(axis=1)
    out['mag_std'] = out[bands].std(axis=1)
    out['mag_min'] = out[bands].min(axis=1)
    out['mag_max'] = out[bands].max(axis=1)
    out['mag_range'] = out['mag_max'] - out['mag_min']
    color_all = [f'{bands[a]}_{bands[b]}' for a in range(len(bands)) for b in range(a + 1, len(bands))]
    out['color_mean'] = out[color_all].mean(axis=1)
    out['color_std'] = out[color_all].std(axis=1)
    out['color_spread'] = out[color_all].max(axis=1) - out[color_all].min(axis=1)
    out['ug_minus_iz'] = out['u_g'] - out['i_z']
    out['gr_minus_ri'] = out['g_r'] - out['r_i']
    for c in CAT_COLS:
        out[c] = out[c].astype('category')
    if 'id' in out.columns:
        out = out.drop(columns=['id'])
    if 'class' in out.columns:
        out = out.drop(columns=['class'])
    return out


def write_sub(test_ids, pred_labels, path):
    sub = pd.DataFrame({'id': test_ids, 'class': pred_labels})
    sub.to_csv(path, index=False)
    log(f"wrote {Path(path).name}: {sub.shape}  dist={dict(sub['class'].value_counts())}")


def run_smoke():
    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import train_test_split
    log("=" * 60)
    log("V8 SMOKE — AutoGluon medium_quality, 100k subsample, holdout check")
    train_df, test_df, test_ids = make_frames()
    sub = train_df.sample(130000, random_state=1).reset_index(drop=True)
    tr_df, ho_df = train_test_split(sub, test_size=30000, stratify=sub['class'], random_state=99)
    t0 = time.time()
    pred = TabularPredictor(label='class', eval_metric='balanced_accuracy',
                            path=str(ART8 / 'ag_smoke'), verbosity=2)
    pred.fit(tr_df, presets='medium_quality', time_limit=600, ag_args_fit={'num_gpus': 1})
    p = pred.predict(ho_df.drop(columns=['class']))
    ba = balanced_accuracy_score(ho_df['class'].values, p.values)
    log(f"SMOKE AutoGluon holdout balanced_acc = {ba:.4f}  ({time.time()-t0:.0f}s)")
    log("(reference: our HGB ~0.964, v4 meta OOF 0.9660)")
    print(pred.leaderboard(ho_df, silent=True).to_string())
    return ba


def run_full():
    from autogluon.tabular import TabularPredictor
    log("=" * 60)
    log("V8 FULL — AutoGluon best_quality, full train, target beat 0.97154")
    train_df, test_df, test_ids = make_frames()
    log(f"train {train_df.shape}  test {test_df.shape}")
    t0 = time.time()
    pred = TabularPredictor(label='class', eval_metric='balanced_accuracy',
                            path=str(ART8 / 'ag_full'), verbosity=2)
    pred.fit(train_df, presets='best_quality', time_limit=14400, ag_args_fit={'num_gpus': 1})
    log(f"fit done ({time.time()-t0:.0f}s)")

    # OOF balanced accuracy (leak-free, from AutoGluon internal bagging)
    try:
        oof = pred.predict_proba_oof()                 # DataFrame (n_train, n_classes)
        oof = oof[CLASSES].values
        oof_pred = np.array(CLASSES)[oof.argmax(1)]
        ba_oof = balanced_accuracy_score(train_df['class'].values, oof_pred)
        np.save(ART8 / 'ag_oof.npy', oof)
        log(f"AutoGluon OOF balanced_acc = {ba_oof:.4f}  (v4 was 0.9660)")
    except Exception as e:
        log(f"OOF extract failed: {e}")
        ba_oof = float('nan')

    # test prediction
    test_pred = pred.predict(test_df).values
    test_proba = pred.predict_proba(test_df)[CLASSES].values
    np.save(ART8 / 'ag_test.npy', test_proba)
    write_sub(test_ids, test_pred, ROOT / 'submission_v8_autogluon.csv')

    log("LEADERBOARD (internal models, val score = balanced_accuracy):")
    print(pred.leaderboard(silent=True).to_string())
    log(f"V8 DONE — OOF {ba_oof:.4f} vs v4 0.9660 / target 0.97154")
    log("HONEST GATE: if OOF ~0.967 (plateau), STOP and keep best single model.")


def run_full_balanced():
    """Same as run_full but with sample_weight='balance_weight' — injects our proven balanced
    lever (worth +0.008-0.010 on every base) into AutoGluon's stronger stacking machinery."""
    from autogluon.tabular import TabularPredictor
    log("=" * 60)
    log("V8 FULL BALANCED — AutoGluon best_quality + sample_weight=balance_weight")
    train_df, test_df, test_ids = make_frames()
    log(f"train {train_df.shape}  test {test_df.shape}")
    t0 = time.time()
    pred = TabularPredictor(label='class', eval_metric='balanced_accuracy',
                            sample_weight='balance_weight', weight_evaluation=False,
                            path=str(ART8 / 'ag_bal'), verbosity=2)
    # dynamic_stacking=False avoids the detection pre-fit that doubled the default run's time;
    # 2h budget is plenty for 8-fold bag + L2 stack on this data.
    pred.fit(train_df, presets='best_quality', time_limit=7200, dynamic_stacking=False,
             ag_args_fit={'num_gpus': 1})
    log(f"fit done ({time.time()-t0:.0f}s)")
    try:
        oof = pred.predict_proba_oof()[CLASSES].values
        ba_oof = balanced_accuracy_score(train_df['class'].values, np.array(CLASSES)[oof.argmax(1)])
        np.save(ART8 / 'ag_bal_oof.npy', oof)
        log(f"AutoGluon BALANCED OOF balanced_acc = {ba_oof:.4f}  (v4 0.9660, default-AG was lower)")
    except Exception as e:
        log(f"OOF extract failed: {e}"); ba_oof = float('nan')
    test_pred = pred.predict(test_df).values
    np.save(ART8 / 'ag_bal_test.npy', pred.predict_proba(test_df)[CLASSES].values)
    write_sub(test_ids, test_pred, ROOT / 'submission_v8_balanced.csv')
    log("LEADERBOARD:")
    print(pred.leaderboard(silent=True).to_string())
    log(f"V8 BALANCED DONE — OOF {ba_oof:.4f} vs v4 0.9660 / target 0.97154")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if mode == 'smoke':
        run_smoke()
    elif mode == 'full':
        run_full()
    elif mode == 'balanced':
        run_full_balanced()
    else:
        print("usage: python pipeline_v8.py [smoke|full|balanced]")
