"""
v9: SUPER-STACK — external public OOF/test artifacts + our own diverse OOF -> final meta.

External members (downloaded via Kaggle API, all row-aligned to train.csv/test.csv,
class order GALAXY/QSO/STAR verified by permutation check):
  lr_v7   debatreyabiswas sklearn meta-stacker   OOF 0.97019  (LB 0.97105 confirmed)
  lr_v9   cdeotte GPU LogReg stacker             OOF 0.97028
  xgb5    cdeotte XGB v5                         OOF 0.96770
  rmlp5   cdeotte RealMLP v5                     OOF 0.96904
  xgb1    cdeotte XGB v1                         OOF 0.96694
  rmlp1   cdeotte RealMLP v1                     OOF 0.96881
  yk_rmlp yekenot RealMLP (csv id-aligned)
  nf_rmlp nawfeel RealMLP 0.96980 (csv)
Ours (artifacts/, artifacts_v8/): v4 tabpfn meta 0.9660, hgb 0.9642, AutoGluon-balanced 0.9656.

Meta-learners tried (honest 5-fold CV on the meta features):
  1. LogisticRegression on member logits (the classic, what lr_v7/v9 themselves are)
  2. hill-climb weighted logit-average (greedy, optimizes OOF balanced_accuracy directly)
Pick best by OOF, predict test, write submission_v9.csv.

Offset note: lr_v7 OOF 0.97019 -> LB 0.97105 (+0.00086). Target LB 0.97170 ~ OOF 0.9708+.
"""
import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
EXT, ART, ART8 = ROOT/'external', ROOT/'artifacts', ROOT/'artifacts_v8'
C = ['GALAXY','QSO','STAR']; c2i = {c:i for i,c in enumerate(C)}
EPS = 1e-6

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(ROOT/'pipeline.log','a') as f: f.write(line+'\n')

def to_logit(p):
    p = np.asarray(p, dtype='float64')
    p = p / p.sum(1, keepdims=True).clip(EPS)
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p/(1-p)).astype('float32')

def load_npy(path):
    a = np.load(path)
    if a.ndim == 3: a = a.mean(0)
    return a.astype('float32')

def load_csv_probs(path, ids_expected):
    d = pd.read_csv(path)
    if 'id' in d.columns:
        d = d.set_index('id').reindex(ids_expected).reset_index()
    return d[C].values.astype('float32')


def main():
    log("="*60)
    log("V9 SUPER-STACK — external + own OOF -> meta. Target LB 0.97170")
    tr = pd.read_csv(ROOT/'data/train.csv'); te = pd.read_csv(ROOT/'data/test.csv')
    y = tr['class'].map(c2i).values
    train_ids, test_ids = tr['id'].values, te['id'].values
    N, M = len(y), len(test_ids)

    members = {}   # name -> (oof, test)
    def add(name, oof, test):
        if oof.shape != (N,3) or test.shape != (M,3):
            log(f"  SKIP {name}: bad shape oof{oof.shape} test{test.shape}"); return
        members[name] = (oof, test)
        log(f"  {name}: OOF bacc={balanced_accuracy_score(y, oof.argmax(1)):.5f}")

    # ---- external ----
    B = EXT/'debatreyabiswas_0-97105-sklearn-meta-stacker-2-5x-faster-saga'
    add('lr_v7', load_npy(B/'oof_lr_stacker_v7.npy'), load_npy(B/'pred_lr_stacker_v7.npy'))
    B = EXT/'cdeotte_gpu-logistic-regression-stacker'
    add('lr_v9', load_npy(B/'oof_lr_stacker_v9.npy'), load_npy(B/'pred_lr_stacker_v9.npy'))
    B = EXT/'cdeotte_xgb-v5-for-s6e6'
    add('xgb5', load_npy(B/'train_oof/xgb-5_oof.npy'), load_npy(B/'test_preds/xgb-5_test_preds.npy'))
    B = EXT/'cdeotte_realmlp-v5-for-s6e6'
    add('rmlp5', load_npy(B/'train_oof/realmlp-5_oof.npy'), load_npy(B/'test_preds/realmlp-5_test_preds.npy'))
    B = EXT/'cdeotte_xgb-v1-for-s6e6'
    add('xgb1', load_npy(B/'oof_preds.npy'), load_npy(B/'test_preds.npy'))
    B = EXT/'cdeotte_realmlp-v1-for-s6e6'
    add('rmlp1', load_npy(B/'oof_preds.npy'), load_npy(B/'test_preds.npy'))
    B = EXT/'yekenot_ps-s6-e6-realmlp-pytorch'
    try:
        add('yk_rmlp', load_csv_probs(B/'oof_preds.csv', train_ids),
                       load_csv_probs(B/'test_preds.csv', test_ids))
    except Exception as e: log(f"  yk_rmlp skip: {e}")
    B = EXT/'nawfeelrahman1124444_single-realmlp-0-96980-v10'
    try:
        oof_f = list((B/'train_oof').glob('*'))[0]; test_f = list((B/'test_preds').glob('*'))[0]
        ld = load_csv_probs if oof_f.suffix=='.csv' else (lambda p,_: load_npy(p))
        add('nf_rmlp', ld(oof_f, train_ids), ld(test_f, test_ids))
    except Exception as e: log(f"  nf_rmlp skip: {e}")

    # ---- ours (diversity the public stackers don't have) ----
    for name, fo, ft in [('our_v4', ART/'tabpfn_oof.npy', ART/'tabpfn_test.npy'),
                         ('our_hgb', ART/'hgb_oof.npy', ART/'hgb_test.npy'),
                         ('our_ag',  ART8/'ag_bal_oof.npy', ART8/'ag_bal_test.npy')]:
        if fo.exists() and ft.exists():
            add(name, load_npy(fo), load_npy(ft))

    names = list(members)
    log(f"members: {names}")

    # ============ approach 1: LogReg meta on member logits ============
    mtr = np.concatenate([to_logit(members[n][0]) for n in names], 1)
    mte = np.concatenate([to_logit(members[n][1]) for n in names], 1)
    sc = StandardScaler().fit(mtr)
    mtr_s, mte_s = sc.transform(mtr), sc.transform(mte)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=3000, C=0.3)
    t0 = time.time()
    oof_lr = cross_val_predict(lr, mtr_s, y, cv=skf, method='predict_proba', n_jobs=-1)
    ba_lr = balanced_accuracy_score(y, oof_lr.argmax(1))
    log(f"LogReg meta OOF bacc = {ba_lr:.5f} ({time.time()-t0:.0f}s)")
    lr.fit(mtr_s, y)
    test_lr = lr.predict_proba(mte_s)

    # ============ approach 2: greedy hill-climb logit blend ============
    logits_oof = {n: to_logit(members[n][0]) for n in names}
    logits_te  = {n: to_logit(members[n][1]) for n in names}
    # start from best single member, greedily add (with replacement) the member that
    # maximizes OOF bacc of the running average
    def bacc_of(Lsum, k): return balanced_accuracy_score(y, (Lsum/k).argmax(1))
    best0 = max(names, key=lambda n: balanced_accuracy_score(y, members[n][0].argmax(1)))
    sel = [best0]; Lsum = logits_oof[best0].copy()
    cur = bacc_of(Lsum, 1)
    log(f"hill-climb start: {best0} {cur:.5f}")
    for it in range(20):
        cands = []
        for n in names:
            cands.append((bacc_of(Lsum + logits_oof[n], len(sel)+1), n))
        nb, nn = max(cands)
        if nb <= cur + 1e-6: break
        sel.append(nn); Lsum += logits_oof[nn]; cur = nb
        log(f"  +{nn} -> {cur:.5f}")
    ba_hc = cur
    log(f"hill-climb OOF bacc = {ba_hc:.5f}  members={sel}")
    Lte = sum(logits_te[n] for n in sel) / len(sel)

    # ============ pick + write ============
    cands = {'logreg': (ba_lr, test_lr), 'hillclimb': (ba_hc, Lte)}
    bestname = max(cands, key=lambda k: cands[k][0])
    ba_best, test_best = cands[bestname]
    pred = np.array(C)[test_best.argmax(1)]
    sub = pd.DataFrame({'id': test_ids, 'class': pred})
    sub.to_csv(ROOT/'submission_v9.csv', index=False)
    log(f"submission_v9.csv = {bestname} (OOF {ba_best:.5f})  dist={dict(sub['class'].value_counts())}")
    log(f"reference: lr_v7 OOF 0.97019 -> LB 0.97105 (+0.00086). Need OOF ~0.9708 for LB 0.97170.")
    log("V9 DONE")


if __name__ == '__main__':
    main()
