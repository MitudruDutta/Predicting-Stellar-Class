"""
v5: propagate balanced class weighting to ALL bases (the lever — HGB balanced=0.9644 vs none=0.9550)
+ add HGB variants. Re-stack with TabPFN. Reuses pipeline.py helpers + cached v4 bases where valid.

New bases (balanced): xgbB, lgbB, catB, hgb2, hgb3. Plus reuse hgb, et, mlp from v4.
Caches to artifacts/ with _v5 / B suffixes so v4 stays intact.
"""
import os, sys, time, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import (ROOT, ART, log, get_matrices, to_logit, oof_cached,
                      fit_predict_sklearn, SKF, N_CLASSES, N_FOLDS, SEED,
                      build_meta_features, tabpfn_oof, tabpfn_test,
                      tabpfn_predict_chunked, write_sub, I2C)
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def fit_predict_weighted(m, Xtr, ytr, Xq):
    """fit with balanced sample weights -> directly optimizes for balanced_accuracy."""
    sw = compute_sample_weight('balanced', ytr)
    m.fit(Xtr, ytr, sample_weight=sw)
    return m.predict_proba(Xq)


def build_bases_v5(X, Xt, y):
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    oof, test = {}, {}

    # --- balanced-weighted boosters (the fix) ---
    def xgbB(sd):
        return xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=8,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.5,
            objective='multi:softprob', num_class=N_CLASSES, tree_method='hist',
            device='cuda', eval_metric='mlogloss', random_state=sd)

    def lgbB(sd):
        return lgb.LGBMClassifier(n_estimators=1800, learning_rate=0.03, num_leaves=127,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.5,
            min_child_samples=40, objective='multiclass', num_class=N_CLASSES,
            device='cpu', random_state=sd, n_jobs=-1, verbose=-1)

    def catB(sd):
        return CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, loss_function='MultiClass', task_type='GPU', devices='0',
            auto_class_weights='Balanced', random_seed=sd, verbose=False, allow_writing_files=False)

    # --- HGB variants (the star model family) ---
    def hgb2(sd):
        return HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.03,
            max_iter=1200, max_leaf_nodes=127, l2_regularization=2.0, max_depth=None,
            min_samples_leaf=30, class_weight='balanced', early_stopping=True,
            n_iter_no_change=40, validation_fraction=0.1, random_state=sd)

    def hgb3(sd):
        return HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.08,
            max_iter=400, max_leaf_nodes=31, l2_regularization=0.5,
            class_weight='balanced', early_stopping=True, random_state=sd)

    # balanced boosters via sample_weight
    oof['xgbB'], test['xgbB'] = oof_cached('xgbB', xgbB, fit_predict_weighted, X, Xt, y, seeds=(42, 1, 7))
    oof['lgbB'], test['lgbB'] = oof_cached('lgbB', lgbB, fit_predict_weighted, X, Xt, y, seeds=(42, 1))
    # cat uses auto_class_weights (its own balanced) -> plain fit
    oof['catB'], test['catB'] = oof_cached('catB', catB, fit_predict_sklearn, X, Xt, y, seeds=(42, 1, 7))
    # HGB variants (class_weight native)
    oof['hgb2'], test['hgb2'] = oof_cached('hgb2', hgb2, fit_predict_sklearn, X, Xt, y, seeds=(42, 1))
    oof['hgb3'], test['hgb3'] = oof_cached('hgb3', hgb3, fit_predict_sklearn, X, Xt, y, seeds=(42, 1))

    # reuse v4 cached bases that are still useful + diverse
    for n in ['hgb', 'et', 'mlp', 'xgb', 'lgb', 'cat']:
        fo, ft = ART / f'{n}_oof.npy', ART / f'{n}_test.npy'
        if fo.exists() and ft.exists():
            oof[n], test[n] = np.load(fo), np.load(ft)
            log(f"[{n}] reused v4 cache bacc={balanced_accuracy_score(y, oof[n].argmax(1)):.4f}")
    return oof, test


def main():
    log("=" * 60)
    log("PIPELINE V5 START — balanced weights everywhere + HGB variants. Target: beat 0.97092")
    X, Xt, y, test_ids, feats = get_matrices()

    oof, test = build_bases_v5(X, Xt, y)
    log("ALL base OOF bacc:")
    for n in sorted(oof, key=lambda k: -balanced_accuracy_score(y, oof[k].argmax(1))):
        log(f"  {n}: {balanced_accuracy_score(y, oof[n].argmax(1)):.4f}")

    # meta = strongest + most diverse bases. Drop weak unweighted trees (xgb/lgb/cat) if their
    # balanced versions exist; keep diversity (mlp, et).
    meta_names = ['xgbB', 'lgbB', 'catB', 'hgb', 'hgb2', 'hgb3', 'et', 'mlp']
    meta_names = [n for n in meta_names if n in oof]
    mtr, mte = build_meta_features(oof, test, X, Xt, meta_names)
    log(f"meta bases: {meta_names}  -> {mtr.shape[1]} cols")

    # avg-blend reference
    avg = sum(oof[n] for n in meta_names) / len(meta_names)
    log(f"avg-blend OOF bacc={balanced_accuracy_score(y, avg.argmax(1)):.4f}")

    # TabPFN meta — fewer OOF seeds (OOF score stable, save time); full seeds for test
    fo, ft = ART / 'tabpfn_v5_oof.npy', ART / 'tabpfn_v5_test.npy'
    if fo.exists():
        tp_oof = np.load(fo)
    else:
        tp_oof = tabpfn_oof(mtr, y, ctx=30000, n_est=12, seeds=(42,))   # 1 seed for speed
        np.save(fo, tp_oof)
    log(f"TabPFN v5 meta OOF bacc={balanced_accuracy_score(y, tp_oof.argmax(1)):.4f}")

    if ft.exists():
        tp_test = np.load(ft)
    else:
        tp_test = tabpfn_test(mtr, y, mte, ctx=30000, n_est=12, seeds=(42, 1, 7))
        np.save(ft, tp_test)

    write_sub(test_ids, tp_test, ROOT / 'submission_v5.csv')
    log(f"submission_v5.csv = TabPFN meta (OOF {balanced_accuracy_score(y, tp_oof.argmax(1)):.4f})")
    log(f"BEST OOF = {balanced_accuracy_score(y, tp_oof.argmax(1)):.4f}  (v4 LB was 0.96758, target 0.97092)")
    log("PIPELINE V5 DONE")


if __name__ == '__main__':
    main()
