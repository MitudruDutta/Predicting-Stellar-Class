"""
v6: break base redundancy (v5 bases agreed 98-99%) + lift GALAXY recall (weakest, 0.957).
Diagnosis showed stronger correlated trees won't cross 0.971; need DIVERSE errors + features.

New:
  - richer features (redshift bins/quantile-rank, color*redshift, per-class centroid distances,
    band ratios) -> X6 (wider matrix)
  - diverse non-tree bases that make DIFFERENT errors: KNN(scaled), 2nd MLP (different arch),
    LogisticRegression(scaled balanced). Break the 98% agreement.
  - reuse v5 balanced trees (xgbB/lgbB/catB/hgb/hgb2/hgb3) — but RETRAIN them on X6 (new feats).
  - TabPFN meta n_est16, 3 OOF seeds (v4 setting, more ensembling than v5's 12/1seed).

Caches to artifacts_v6/.
"""
import os, sys, time, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import (ROOT, log, get_matrices, to_logit, SKF, N_CLASSES, N_FOLDS, SEED,
                      tabpfn_oof, tabpfn_test, tabpfn_predict_chunked, write_sub, I2C,
                      load_and_clean)
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ART6 = ROOT / 'artifacts_v6'
ART6.mkdir(exist_ok=True)


# ---------------------------------------------------------------- richer features
def build_features_v6(df):
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
    # redshift-color slope proxy (how color changes — separates QSO/GALAXY)
    out['ug_minus_iz'] = out['u_g'] - out['i_z']
    out['gr_minus_ri'] = out['g_r'] - out['r_i']
    out = pd.get_dummies(out, columns=['spectral_type', 'galaxy_population'], dtype=float)
    return out


def get_matrices_v6():
    tr, te = load_and_clean()
    trf, tef = build_features_v6(tr), build_features_v6(te)
    drop = ['id', 'class']
    feats = [c for c in trf.columns if c not in drop]
    X = trf[feats].astype('float32').values
    Xt = tef.reindex(columns=feats, fill_value=0.0).astype('float32').values
    y = tr['class'].map(P.C2I).astype('int64').values
    ids = te['id'].values
    # centroid-distance features (per-class mean in scaled space) — fit on train, add as cols
    log(f"v6 base features={len(feats)} X={X.shape}")
    return X, Xt, y, ids, feats


def add_centroid_dist(Xtr, ytr, Xq):
    """Distance from each row to per-class centroid (scaled). Adds 3 cols. Leak-free: centroids
    from the training fold only."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xq_s = sc.transform(Xtr), sc.transform(Xq)
    cents = np.stack([Xtr_s[ytr == c].mean(0) for c in range(N_CLASSES)])  # (3, d)
    def dists(Z):
        return np.stack([np.linalg.norm(Z - cents[c], axis=1) for c in range(N_CLASSES)], axis=1)
    return dists(Xq_s).astype('float32')


# ---------------------------------------------------------------- oof runner (v6 cache)
def oof6(name, make_model, fit_predict, X, Xt, y, seeds=(SEED,), add_cent=False):
    fo, ft = ART6 / f'{name}_oof.npy', ART6 / f'{name}_test.npy'
    if fo.exists() and ft.exists():
        oof, test = np.load(fo), np.load(ft)
        log(f"[{name}] CACHED bacc={balanced_accuracy_score(y, oof.argmax(1)):.4f}")
        return oof, test
    oof = np.zeros((len(y), N_CLASSES), dtype='float32')
    test = np.zeros((Xt.shape[0], N_CLASSES), dtype='float32')
    t0 = time.time()
    for sd in seeds:
        for f, (tr, va) in enumerate(SKF.split(X, y)):
            Xtr, Xva = X[tr], X[va]
            if add_cent:
                cd_tr = add_centroid_dist(Xtr, y[tr], Xtr)
                cd_va = add_centroid_dist(Xtr, y[tr], Xva)
                cd_te = add_centroid_dist(Xtr, y[tr], Xt)
                Xtr = np.concatenate([Xtr, cd_tr], 1)
                Xva = np.concatenate([Xva, cd_va], 1)
                Xte = np.concatenate([Xt, cd_te], 1)
            else:
                Xte = Xt
            m = make_model(sd)
            oof[va] += fit_predict(m, Xtr, y[tr], Xva) / len(seeds)
            test += fit_predict(m, Xtr, y[tr], Xte) / (len(seeds) * N_FOLDS)
            del m; gc.collect()
        log(f"[{name}] seed {sd} ({time.time()-t0:.0f}s)")
    ba = balanced_accuracy_score(y, oof.argmax(1))
    log(f"[{name}] OOF bacc={ba:.4f} ({time.time()-t0:.0f}s)")
    np.save(fo, oof); np.save(ft, test)
    return oof, test


def fp_plain(m, Xtr, ytr, Xq):
    m.fit(Xtr, ytr); return m.predict_proba(Xq)

def fp_weighted(m, Xtr, ytr, Xq):
    sw = compute_sample_weight('balanced', ytr)
    m.fit(Xtr, ytr, sample_weight=sw); return m.predict_proba(Xq)

def fp_scaled(m, Xtr, ytr, Xq):
    sc = StandardScaler().fit(Xtr)
    m.fit(sc.transform(Xtr), ytr); return m.predict_proba(sc.transform(Xq))

def fp_knn_fast(m, Xtr, ytr, Xq):
    """KNN with a subsampled reference set (40k stratified) — full 462k brute-force is too slow.
    Still captures distance structure for diversity."""
    from sklearn.model_selection import train_test_split
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    n_ref = min(40000, len(ytr))
    idx, _ = train_test_split(np.arange(len(ytr)), train_size=n_ref, stratify=ytr, random_state=SEED)
    m.fit(Xtr_s[idx], ytr[idx])
    return m.predict_proba(sc.transform(Xq))


def build_mlp6(X, Xt, y, arch=(512, 256, 128), p=0.3, name='mlp6'):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fo, ft = ART6 / f'{name}_oof.npy', ART6 / f'{name}_test.npy'
    if fo.exists() and ft.exists():
        oof, test = np.load(fo), np.load(ft)
        log(f"[{name}] CACHED bacc={balanced_accuracy_score(y, oof.argmax(1)):.4f}")
        return oof, test

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            L, dd = [], d
            for h in arch:
                L += [nn.Linear(dd, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]; dd = h
            L += [nn.Linear(dd, N_CLASSES)]
            self.net = nn.Sequential(*L)
        def forward(self, x): return self.net(x)

    oof = np.zeros((len(y), N_CLASSES), 'float32')
    test = np.zeros((Xt.shape[0], N_CLASSES), 'float32')
    t0 = time.time()
    for f, (tr, va) in enumerate(SKF.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva, Xte = sc.transform(X[tr]).astype('float32'), sc.transform(X[va]).astype('float32'), sc.transform(Xt).astype('float32')
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = MLP(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        w = torch.tensor(np.sqrt(len(y[tr])/(N_CLASSES*np.bincount(y[tr]))), dtype=torch.float32).to(dev)
        crit = nn.CrossEntropyLoss(weight=w)
        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y[tr])), batch_size=4096, shuffle=True)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, steps_per_epoch=len(dl), epochs=30)
        model.train()
        for _ in range(30):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            oof[va] = torch.softmax(model(torch.from_numpy(Xva).to(dev)), 1).cpu().numpy()
            test += torch.softmax(model(torch.from_numpy(Xte).to(dev)), 1).cpu().numpy() / N_FOLDS
        del model; torch.cuda.empty_cache()
        log(f"[{name}] fold {f+1} ({time.time()-t0:.0f}s)")
    ba = balanced_accuracy_score(y, oof.argmax(1))
    log(f"[{name}] OOF bacc={ba:.4f} ({time.time()-t0:.0f}s)")
    np.save(fo, oof); np.save(ft, test)
    return oof, test


def main():
    log("=" * 60)
    log("PIPELINE V6 — diversity + features. Target beat 0.97127")
    X, Xt, y, ids, feats = get_matrices_v6()

    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    oof, test = {}, {}

    # strong balanced trees on the WIDER v6 feature set
    oof['xgbB'], test['xgbB'] = oof6('xgbB', lambda sd: xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=8, subsample=0.85, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.5, objective='multi:softprob', num_class=N_CLASSES,
        tree_method='hist', device='cuda', eval_metric='mlogloss', random_state=sd),
        fp_weighted, X, Xt, y, seeds=(42, 1, 7))
    oof['hgb2'], test['hgb2'] = oof6('hgb2', lambda sd: HistGradientBoostingClassifier(
        loss='log_loss', learning_rate=0.03, max_iter=1200, max_leaf_nodes=127, l2_regularization=2.0,
        min_samples_leaf=30, class_weight='balanced', early_stopping=True, n_iter_no_change=40,
        validation_fraction=0.1, random_state=sd), fp_plain, X, Xt, y, seeds=(42, 1))
    oof['catB'], test['catB'] = oof6('catB', lambda sd: CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, loss_function='MultiClass',
        task_type='GPU', devices='0', auto_class_weights='Balanced', random_seed=sd,
        verbose=False, allow_writing_files=False), fp_plain, X, Xt, y, seeds=(42, 1))

    # DIVERSE bases (different error structure)
    # KNN on scaled (distance-based — totally different from trees)
    oof['knn'], test['knn'] = oof6('knn', lambda sd: KNeighborsClassifier(
        n_neighbors=48, weights='distance', algorithm='kd_tree', n_jobs=-1), fp_knn_fast, X, Xt, y, seeds=(42,))
    # LogReg balanced (linear boundary — different errors)
    oof['lr'], test['lr'] = oof6('lr', lambda sd: LogisticRegression(
        max_iter=3000, C=1.0, class_weight='balanced'), fp_scaled, X, Xt, y, seeds=(42,))
    # wide deep MLP (GELU, bigger) on v6 feats
    oof['mlp6'], test['mlp6'] = build_mlp6(X, Xt, y)

    # reuse v5 cached strong trees (trained on 33-feat X — still diverse vs v6 versions)
    from pipeline import ART as ART5
    for n in ['lgbB', 'hgb', 'hgb3']:
        fo, ft = ART5 / f'{n}_oof.npy', ART5 / f'{n}_test.npy'
        if fo.exists():
            oof[n], test[n] = np.load(fo), np.load(ft)
            log(f"[{n}] reused v5 bacc={balanced_accuracy_score(y, oof[n].argmax(1)):.4f}")

    log("ALL base OOF:")
    for n in sorted(oof, key=lambda k: -balanced_accuracy_score(y, oof[k].argmax(1))):
        log(f"  {n}: {balanced_accuracy_score(y, oof[n].argmax(1)):.4f}")

    # diversity check
    names = list(oof)
    preds = {n: oof[n].argmax(1) for n in names}
    log("min pairwise agreement (want < 0.97 for diversity):")
    mn = 1.0
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = (preds[names[i]] == preds[names[j]]).mean()
            mn = min(mn, a)
    log(f"  min agreement = {mn:.4f}")

    # meta over all
    meta_names = names
    mtr = np.concatenate([to_logit(oof[n]) for n in meta_names] + [X], axis=1).astype('float32')
    mte = np.concatenate([to_logit(test[n]) for n in meta_names] + [Xt], axis=1).astype('float32')
    log(f"meta: {len(meta_names)} bases -> {mtr.shape[1]} cols")
    avg = sum(oof[n] for n in meta_names) / len(meta_names)
    log(f"avg-blend OOF={balanced_accuracy_score(y, avg.argmax(1)):.4f}")

    fo, ft = ART6 / 'tabpfn_v6_oof.npy', ART6 / 'tabpfn_v6_test.npy'
    tp_oof = np.load(fo) if fo.exists() else tabpfn_oof(mtr, y, ctx=30000, n_est=16, seeds=(42, 1, 7))
    if not fo.exists(): np.save(fo, tp_oof)
    log(f"TabPFN v6 meta OOF={balanced_accuracy_score(y, tp_oof.argmax(1)):.4f}")
    tp_test = np.load(ft) if ft.exists() else tabpfn_test(mtr, y, mte, ctx=30000, n_est=16, seeds=(42, 1, 7))
    if not ft.exists(): np.save(ft, tp_test)

    write_sub(ids, tp_test, ROOT / 'submission_v6.csv')
    log(f"submission_v6.csv OOF={balanced_accuracy_score(y, tp_oof.argmax(1)):.4f}")
    log(f"BEST v6 OOF = {balanced_accuracy_score(y, tp_oof.argmax(1)):.4f} (v4 LB 0.96758, target 0.97127)")
    log("PIPELINE V6 DONE")


if __name__ == '__main__':
    main()
