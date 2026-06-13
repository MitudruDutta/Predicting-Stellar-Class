"""
v11: TabPFN-3 stacker over the public base-model pool + raw features.

Goal: beat the public LogReg stackers (lr_v9 OOF 0.97028) by using TabPFN-3 as the meta
(stronger than LogReg) over the SAME strong base pool they used, PLUS our raw 33 features that
the public stackers omitted. If OOF > 0.97028 we have a genuinely stronger member to mine/select.

Pool (13 leak-free base OOFs): cdeotte lgbm5/rmlp0v12/rmlp2v10/xgb6/tabm1/xgb5/rmlp5/xgb1/rmlp1,
yekenot rmlp, our tabpfn_v10 / tabpfn_v4 / hgb. Meta features = [13*3 base logits + 33 raw] = 72 cols.
TabPFN-3 ctx30k n_est16, OOF 1 seed (check) + test 3 seeds.
"""
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import (ROOT, log, get_matrices, to_logit, tabpfn_oof, tabpfn_test, write_sub, I2C)
from sklearn.metrics import balanced_accuracy_score

C = ['GALAXY', 'QSO', 'STAR']; c2i = {c: i for i, c in enumerate(C)}
EXT, ART = ROOT/'external', ROOT/'artifacts'
D = EXT/'cdeotte_s6e6-oof-and-test-preds'


def load_pair(oof_p, test_p, ids_tr, ids_te, N, M):
    def ld(p, ids_exp, n):
        p = Path(p)
        if p.suffix == '.npy':
            a = np.load(p);  a = a.mean(0) if a.ndim == 3 else a
        else:
            d = pd.read_csv(p)
            if 'id' in d.columns:
                d = d.set_index('id').reindex(ids_exp).reset_index()
            cols = [c for c in d.columns if c in C]
            a = d[cols].values if len(cols) == 3 else d.iloc[:, -3:].values
        return a.astype('float32')
    o, t = ld(oof_p, ids_tr, N), ld(test_p, ids_te, M)
    return (o, t) if (o.shape[0] == N and t.shape[0] == M) else (None, None)


def main():
    log("="*60)
    log("V11 — TabPFN-3 stacker over public base pool + raw feats. Beat lr_v9 0.97028")
    X, Xt, y, test_ids, feats = get_matrices()   # 33 raw feats
    tr = pd.read_csv(ROOT/'data/train.csv'); te = pd.read_csv(ROOT/'data/test.csv')
    ids_tr, ids_te = tr['id'].values, te['id'].values
    N, M = len(y), len(ids_te)

    specs = {
        'lgbm5':   (D/'oof_preds_lgbm5_v1.csv',    D/'test_preds_lgbm5_v1.csv'),
        'rmlp0v12':(D/'oof_preds_realmlp0_v12.csv',D/'test_preds_realmlp0_v12.csv'),
        'rmlp2v10':(D/'oof_preds_realmlp2_v10.csv',D/'test_preds_realmlp2_v10.csv'),
        'xgb6':    (D/'oof_final_xgb6_v1.csv',     D/'test_final_xgb6_v1.csv'),
        'tabm1':   (D/'oof_final_tabm1_v1.csv',    D/'test_final_tabm1_v1.csv'),
        'xgb5':    (EXT/'cdeotte_xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy', EXT/'cdeotte_xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy'),
        'rmlp5':   (EXT/'cdeotte_realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy', EXT/'cdeotte_realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy'),
        'xgb1':    (EXT/'cdeotte_xgb-v1-for-s6e6/oof_preds.npy', EXT/'cdeotte_xgb-v1-for-s6e6/test_preds.npy'),
        'rmlp1':   (EXT/'cdeotte_realmlp-v1-for-s6e6/oof_preds.npy', EXT/'cdeotte_realmlp-v1-for-s6e6/test_preds.npy'),
        'yk':      (EXT/'yekenot_ps-s6-e6-realmlp-pytorch/oof_preds.csv', EXT/'yekenot_ps-s6-e6-realmlp-pytorch/test_preds.csv'),
        'tp10':    (ART/'tabpfn_v10_oof.npy', ART/'tabpfn_v10_test.npy'),
        'v4':      (ART/'tabpfn_oof.npy', ART/'tabpfn_test.npy'),
        'hgb':     (ART/'hgb_oof.npy', ART/'hgb_test.npy'),
    }
    oofs, tests, names = [], [], []
    for n, (op, tp) in specs.items():
        o, t = load_pair(op, tp, ids_tr, ids_te, N, M)
        if o is None:
            log(f"  skip {n}"); continue
        oofs.append(o); tests.append(t); names.append(n)
        log(f"  {n}: OOF {balanced_accuracy_score(y, o.argmax(1)):.5f}")
    log(f"pool: {names}")

    mtr = np.concatenate([to_logit(o) for o in oofs] + [X.astype('float32')], axis=1).astype('float32')
    mte = np.concatenate([to_logit(t) for t in tests] + [Xt.astype('float32')], axis=1).astype('float32')
    log(f"meta features: {mtr.shape[1]} cols")
    avg = sum(oofs) / len(oofs)
    log(f"pool avg-blend OOF: {balanced_accuracy_score(y, avg.argmax(1)):.5f}")

    fo, ft = ART/'tabpfn_v11_oof.npy', ART/'tabpfn_v11_test.npy'
    tp_oof = np.load(fo) if fo.exists() else tabpfn_oof(mtr, y, ctx=30000, n_est=16, seeds=(42,))
    if not fo.exists(): np.save(fo, tp_oof)
    ba = balanced_accuracy_score(y, tp_oof.argmax(1))
    log(f"V11 TabPFN stacker OOF = {ba:.5f}  (public lr_v9 = 0.97028; lr_v7 = 0.97019)")

    tp_test = np.load(ft) if ft.exists() else tabpfn_test(mtr, y, mte, ctx=30000, n_est=16, seeds=(42, 1, 7))
    if not ft.exists(): np.save(ft, tp_test)
    write_sub(test_ids, np.array(C)[tp_test.argmax(1)], ROOT/'submission_v11_tpstack.csv')
    log(f"submission_v11_tpstack.csv written. OOF {ba:.5f}")
    log("V11 DONE")


if __name__ == '__main__':
    main()
