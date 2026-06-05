"""
Autonomous stacking pipeline for Kaggle PS-S6E6 "Predicting Stellar Class".
Goal: beat 0.97092 balanced_accuracy.

Runs standalone (no notebook). Saves base OOF/test probs to artifacts/ so reruns
skip completed work. Logs everything to pipeline.log.

Stages:
  1. load + clean (winsorize mags) + engineer features (v2: 33 features)
  2. base models, 5-fold OOF, multi-seed: XGB, LGB, Cat, MLP, ExtraTrees, HistGBM
  3. TabPFN-3 meta over [base OOF logits + raw features]; also LogReg/HistGBM meta
  4. pick best meta on holdout, seed-average for final, write submissions
"""
import os, sys, time, json, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent          # project root
DATA = ROOT / 'data'
ART  = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)
LOG  = ROOT / 'pipeline.log'

CLASSES = ['GALAXY', 'QSO', 'STAR']
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}
N_CLASSES = 3
N_FOLDS = 5
SEED = 42

# load token from .env
if not os.environ.get('TABPFN_TOKEN'):
    envf = ROOT / '.env'
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------- data + features
def load_and_clean():
    tr = pd.read_csv(DATA / 'train.csv')
    te = pd.read_csv(DATA / 'test.csv')
    mags = ['u', 'g', 'r', 'i', 'z']
    # winsorize mags to [0.1, 99.9] pct, bounds from train
    for c in mags:
        lo, hi = tr[c].quantile(0.001), tr[c].quantile(0.999)
        tr[c] = tr[c].clip(lo, hi)
        te[c] = te[c].clip(lo, hi)
    return tr, te


def build_features(df):
    out = df.copy()
    bands = ['u', 'g', 'r', 'i', 'z']
    # all 10 band-pair colors
    for a in range(len(bands)):
        for b in range(a + 1, len(bands)):
            out[f'{bands[a]}_{bands[b]}'] = out[bands[a]] - out[bands[b]]
    z = out['redshift']
    out['redshift_log1p'] = np.log1p(z.clip(lower=0))
    out['redshift_sq'] = z ** 2
    for col in ['u_g', 'g_r', 'r_i', 'i_z']:
        out[f'z_x_{col}'] = z * out[col]
    out['mag_mean'] = out[bands].mean(axis=1)
    out['mag_std'] = out[bands].std(axis=1)
    color_all = [f'{bands[a]}_{bands[b]}' for a in range(len(bands)) for b in range(a + 1, len(bands))]
    out['color_spread'] = out[color_all].max(axis=1) - out[color_all].min(axis=1)
    out = pd.get_dummies(out, columns=['spectral_type', 'galaxy_population'], dtype=float)
    return out


def get_matrices():
    tr, te = load_and_clean()
    tr_fe, te_fe = build_features(tr), build_features(te)
    drop = ['id', 'class']
    feats = [c for c in tr_fe.columns if c not in drop]
    X = tr_fe[feats].astype('float32').values
    Xt = te_fe.reindex(columns=feats, fill_value=0.0).astype('float32').values
    y = tr['class'].map(C2I).astype('int64').values
    test_ids = te['id'].values
    log(f"features={len(feats)} X={X.shape} Xt={Xt.shape}")
    return X, Xt, y, test_ids, feats


EPS = 1e-6
def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).astype('float32')


# ---------------------------------------------------------------- base models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

SKF = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def oof_cached(name, make_model, fit_predict, X, Xt, y, seeds=(SEED,)):
    """5-fold OOF averaged over seeds. Caches to artifacts/{name}_oof.npy + _test.npy."""
    fo, ft = ART / f'{name}_oof.npy', ART / f'{name}_test.npy'
    if fo.exists() and ft.exists():
        oof, test = np.load(fo), np.load(ft)
        ba = balanced_accuracy_score(y, oof.argmax(1))
        log(f"[{name}] CACHED oof_bacc={ba:.4f}")
        return oof, test
    oof = np.zeros((len(y), N_CLASSES), dtype='float32')
    test = np.zeros((Xt.shape[0], N_CLASSES), dtype='float32')
    t0 = time.time()
    for sd in seeds:
        for f, (tr, va) in enumerate(SKF.split(X, y)):
            m = make_model(sd)
            oof[va] += fit_predict(m, X[tr], y[tr], X[va]) / len(seeds)
            test += fit_predict(m, X[tr], y[tr], Xt) / (len(seeds) * N_FOLDS)
            del m; gc.collect()
        log(f"[{name}] seed {sd} done ({time.time()-t0:.0f}s)")
    ba = balanced_accuracy_score(y, oof.argmax(1))
    log(f"[{name}] OOF bacc={ba:.4f} total {time.time()-t0:.0f}s")
    np.save(fo, oof); np.save(ft, test)
    return oof, test


def fit_predict_sklearn(m, Xtr, ytr, Xq):
    m.fit(Xtr, ytr)
    return m.predict_proba(Xq)


def build_bases(X, Xt, y):
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    oof, test = {}, {}

    def xgb_m(sd):
        return xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=8,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.5,
            objective='multi:softprob', num_class=N_CLASSES, tree_method='hist',
            device='cuda', eval_metric='mlogloss', random_state=sd)

    def lgb_m(sd):
        # CPU: GPU util was only ~15% here (overhead-bound on 33 features) -> CPU w/ all cores is faster
        return lgb.LGBMClassifier(n_estimators=1800, learning_rate=0.03, num_leaves=127,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.5,
            min_child_samples=40, objective='multiclass', num_class=N_CLASSES,
            device='cpu', random_state=sd, n_jobs=-1, verbose=-1)

    def cat_m(sd):
        return CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, loss_function='MultiClass', task_type='GPU', devices='0',
            random_seed=sd, verbose=False, allow_writing_files=False)

    def et_m(sd):
        return ExtraTreesClassifier(n_estimators=600, max_depth=None, min_samples_leaf=4,
            max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=sd)

    def hgb_m(sd):
        return HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.05,
            max_iter=600, max_leaf_nodes=63, l2_regularization=1.0,
            class_weight='balanced', early_stopping=True, random_state=sd)

    oof['xgb'], test['xgb'] = oof_cached('xgb', xgb_m, fit_predict_sklearn, X, Xt, y, seeds=(42, 1, 7))
    oof['lgb'], test['lgb'] = oof_cached('lgb', lgb_m, fit_predict_sklearn, X, Xt, y, seeds=(42, 1))
    oof['cat'], test['cat'] = oof_cached('cat', cat_m, fit_predict_sklearn, X, Xt, y, seeds=(42, 1, 7))
    oof['et'],  test['et']  = oof_cached('et',  et_m,  fit_predict_sklearn, X, Xt, y, seeds=(42,))
    oof['hgb'], test['hgb'] = oof_cached('hgb', hgb_m, fit_predict_sklearn, X, Xt, y, seeds=(42,))
    oof['mlp'], test['mlp'] = oof_cached('mlp', None, None, X, Xt, y) if (ART/'mlp_oof.npy').exists() \
        else build_mlp(X, Xt, y)
    return oof, test


def build_mlp(X, Xt, y):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class MLP(nn.Module):
        def __init__(self, d, h=(256, 128, 64), p=0.2):
            super().__init__()
            L, dd = [], d
            for hh in h:
                L += [nn.Linear(dd, hh), nn.BatchNorm1d(hh), nn.ReLU(), nn.Dropout(p)]; dd = hh
            L += [nn.Linear(dd, N_CLASSES)]
            self.net = nn.Sequential(*L)
        def forward(self, x): return self.net(x)

    def fp(_m, Xtr, ytr, Xq):
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xq_s = sc.transform(Xtr).astype('float32'), sc.transform(Xq).astype('float32')
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = MLP(Xtr_s.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        w = torch.tensor(np.sqrt(len(ytr)/(N_CLASSES*np.bincount(ytr))), dtype=torch.float32).to(dev)
        crit = nn.CrossEntropyLoss(weight=w)
        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr)),
                        batch_size=4096, shuffle=True)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, steps_per_epoch=len(dl), epochs=25)
        model.train()
        for _ in range(25):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            pr = torch.softmax(model(torch.from_numpy(Xq_s).to(dev)), 1).cpu().numpy()
        del model; torch.cuda.empty_cache()
        return pr

    return oof_cached('mlp', lambda sd: None, fp, X, Xt, y)


# ---------------------------------------------------------------- meta layer
def build_meta_features(oof, test, X, Xt, names):
    mtr = np.concatenate([to_logit(oof[n]) for n in names] + [X], axis=1).astype('float32')
    mte = np.concatenate([to_logit(test[n]) for n in names] + [Xt], axis=1).astype('float32')
    return mtr, mte


def tabpfn_predict_chunked(clf, Xq, chunk=20000):
    out = [clf.predict_proba(Xq[s:s+chunk]) for s in range(0, len(Xq), chunk)]
    return np.concatenate(out, axis=0)


def tabpfn_oof(meta_train, y, ctx=30000, n_est=8, seeds=(42, 1, 7)):
    """Leak-free OOF for TabPFN meta via 5-fold; context subsampled from each fold's train part.
    Averaged over seeds. Returns oof probs (N,3)."""
    from tabpfn import TabPFNClassifier
    from sklearn.model_selection import train_test_split
    import torch
    oof = np.zeros((len(y), N_CLASSES), dtype='float32')
    t0 = time.time()
    for sd in seeds:
        for f, (tr, va) in enumerate(SKF.split(meta_train, y)):
            ytr = y[tr]
            sub, _ = train_test_split(np.arange(len(tr)), train_size=min(ctx, len(tr)),
                                      stratify=ytr, random_state=sd)
            clf = TabPFNClassifier(device='cuda', n_estimators=n_est,
                                   balance_probabilities=True, ignore_pretraining_limits=True)
            clf.fit(meta_train[tr][sub], ytr[sub])
            oof[va] += tabpfn_predict_chunked(clf, meta_train[va]) / len(seeds)
            del clf; torch.cuda.empty_cache()
        log(f"  tabpfn_oof seed {sd} done ({time.time()-t0:.0f}s)")
    ba = balanced_accuracy_score(y, oof.argmax(1))
    log(f"  TabPFN meta OOF bacc={ba:.4f} ({time.time()-t0:.0f}s)")
    return oof


def tabpfn_test(meta_train, y, meta_test, ctx=30000, n_est=8, seeds=(42, 1, 7)):
    """Fit on full train (subsampled context), predict test, seed-averaged."""
    from tabpfn import TabPFNClassifier
    from sklearn.model_selection import train_test_split
    import torch
    proba = np.zeros((len(meta_test), N_CLASSES), dtype='float32')
    t0 = time.time()
    for sd in seeds:
        sub, _ = train_test_split(np.arange(len(y)), train_size=ctx, stratify=y, random_state=sd)
        clf = TabPFNClassifier(device='cuda', n_estimators=n_est,
                               balance_probabilities=True, ignore_pretraining_limits=True)
        clf.fit(meta_train[sub], y[sub])
        proba += tabpfn_predict_chunked(clf, meta_test) / len(seeds)
        del clf; torch.cuda.empty_cache()
        log(f"  tabpfn_test seed {sd} done ({time.time()-t0:.0f}s)")
    return proba


def write_sub(test_ids, proba, path):
    pred = np.array([I2C[i] for i in proba.argmax(1)])
    pd.DataFrame({'id': test_ids, 'class': pred}).to_csv(path, index=False)
    d = dict(pd.Series(pred).value_counts())
    log(f"  wrote {Path(path).name}  dist={d}")
    return pred


# ---------------------------------------------------------------- alt metas
def alt_meta_oof_test(meta_train, mte, y):
    """LogReg + HistGBM meta-learners (CPU/fast) over the same meta features.
    Returns dict name -> (oof, test). Used as cheap alternatives / blend members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    out = {}
    # logreg (scaled)
    try:
        sc = StandardScaler().fit(meta_train)
        mtr_s, mte_s = sc.transform(meta_train), sc.transform(mte)
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight=None)
        oof = cross_val_predict(lr, mtr_s, y, cv=SKF, method='predict_proba', n_jobs=-1)
        lr.fit(mtr_s, y)
        out['lr'] = (oof.astype('float32'), lr.predict_proba(mte_s).astype('float32'))
        log(f"  alt meta lr  OOF bacc={balanced_accuracy_score(y, oof.argmax(1)):.4f}")
    except Exception as e:
        log(f"  alt meta lr FAILED: {e}")
    # histgbm
    try:
        hg = HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.05,
            max_iter=500, max_leaf_nodes=63, l2_regularization=1.0,
            early_stopping=True, random_state=SEED)
        oof = cross_val_predict(hg, meta_train, y, cv=SKF, method='predict_proba', n_jobs=-1)
        hg.fit(meta_train, y)
        out['hgbmeta'] = (oof.astype('float32'), hg.predict_proba(mte).astype('float32'))
        log(f"  alt meta hgb OOF bacc={balanced_accuracy_score(y, oof.argmax(1)):.4f}")
    except Exception as e:
        log(f"  alt meta hgb FAILED: {e}")
    return out


# ---------------------------------------------------------------- main
def stage(name, fn, *a, **k):
    """Run a stage with crash-safety so one failure doesn't kill the night."""
    try:
        return fn(*a, **k)
    except Exception as e:
        import traceback
        log(f"!! STAGE '{name}' FAILED: {e}")
        log(traceback.format_exc())
        return None


def main():
    log("=" * 60)
    log("PIPELINE START — target: beat 0.97092")
    X, Xt, y, test_ids, feats = get_matrices()

    # stage 1: bases (each cached individually)
    oof, test = build_bases(X, Xt, y)
    base_names = ['xgb', 'lgb', 'cat', 'et', 'hgb', 'mlp']
    log("base OOF bacc: " + ", ".join(
        f"{n}={balanced_accuracy_score(y, oof[n].argmax(1)):.4f}" for n in base_names))
    avg = sum(oof[n] for n in base_names) / len(base_names)
    log(f"avg-blend OOF bacc={balanced_accuracy_score(y, avg.argmax(1)):.4f}")

    # meta features (all 6 bases + raw)
    meta_names = ['xgb', 'lgb', 'cat', 'mlp', 'et', 'hgb']
    mtr, mte = build_meta_features(oof, test, X, Xt, meta_names)
    log(f"meta features: {mtr.shape[1]} cols")

    # candidate meta predictions: collect (name -> (oof_proba, test_proba))
    metas = {}

    # --- TabPFN meta (n_est=16, the best from the sweep) ---
    def _tabpfn():
        fo, ft = ART / 'tabpfn_oof.npy', ART / 'tabpfn_test.npy'
        tp_oof = np.load(fo) if fo.exists() else tabpfn_oof(mtr, y, ctx=30000, n_est=16, seeds=(42, 1, 7))
        if not fo.exists(): np.save(fo, tp_oof)
        tp_test = np.load(ft) if ft.exists() else tabpfn_test(mtr, y, mte, ctx=30000, n_est=16, seeds=(42, 1, 7))
        if not ft.exists(): np.save(ft, tp_test)
        return tp_oof, tp_test
    r = stage('tabpfn_meta', _tabpfn)
    if r is not None:
        metas['tabpfn'] = r

    # --- alt metas (cheap, robust) ---
    r = stage('alt_metas', alt_meta_oof_test, mtr, mte, y)
    if r:
        metas.update(r)

    if not metas:
        log("!! NO META SUCCEEDED — falling back to base avg-blend")
        write_sub(test_ids, avg, ROOT / 'submission_v4.csv')
        return

    # rank metas by OOF bacc
    ranked = sorted(metas.items(),
                    key=lambda kv: balanced_accuracy_score(y, kv[1][0].argmax(1)), reverse=True)
    log("META RANKING (by OOF bacc):")
    for nm, (o, t) in ranked:
        log(f"  {nm}: {balanced_accuracy_score(y, o.argmax(1)):.4f}")

    # best single meta
    best_name, (best_oof, best_test) = ranked[0]
    write_sub(test_ids, best_test, ROOT / 'submission_v4.csv')
    log(f"submission_v4.csv = best meta '{best_name}' "
        f"(OOF {balanced_accuracy_score(y, best_oof.argmax(1)):.4f})")

    # blend of all metas (probability average) — often beats any single
    blend_oof = sum(o for _, (o, _) in metas.items()) / len(metas)
    blend_test = sum(t for _, (_, t) in metas.items()) / len(metas)
    blend_ba = balanced_accuracy_score(y, blend_oof.argmax(1))
    write_sub(test_ids, blend_test, ROOT / 'submission_v4_blend.csv')
    log(f"submission_v4_blend.csv = mean of {list(metas)} (OOF {blend_ba:.4f})")

    # weighted blend (TabPFN heavier if present)
    if 'tabpfn' in metas:
        w = {nm: (2.0 if nm == 'tabpfn' else 1.0) for nm in metas}
        sw = sum(w.values())
        wb_oof = sum(w[nm] * metas[nm][0] for nm in metas) / sw
        wb_test = sum(w[nm] * metas[nm][1] for nm in metas) / sw
        wb_ba = balanced_accuracy_score(y, wb_oof.argmax(1))
        write_sub(test_ids, wb_test, ROOT / 'submission_v4_wblend.csv')
        log(f"submission_v4_wblend.csv = tabpfn-weighted blend (OOF {wb_ba:.4f})")

    # final summary: which to submit
    candidates = {'best_single': balanced_accuracy_score(y, best_oof.argmax(1)),
                  'blend': blend_ba}
    if 'tabpfn' in metas:
        candidates['wblend'] = wb_ba
    log("=" * 60)
    log("SUBMISSION CANDIDATES (OOF bacc, higher=better):")
    for k, v in sorted(candidates.items(), key=lambda x: -x[1]):
        log(f"  {k}: {v:.4f}")
    log(f"BEST OOF = {max(candidates.values()):.4f}  (vs prior best LB 0.96711, target 0.97092)")
    log("PIPELINE DONE")


if __name__ == '__main__':
    main()
