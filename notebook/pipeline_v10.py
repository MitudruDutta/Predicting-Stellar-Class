"""
v10: TabPFN-3 meta over STRONG EXTERNAL bases + raw features.
The missing diverse family for the final blend (Psi's TabPFN stacker = 0.97061 LB standalone).
Members: cdeotte xgb5/xgb1/lgbm5/rmlp5/rmlp1/rmlp0v12/rmlp2v10/tabm0 + yekenot rmlp + our hgb.
Meta input = [member logits + 33 raw engineered features].
Output artifacts: artifacts/tabpfn_v10_oof.npy / _test.npy -> feed into the v9 blend.
"""
import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import ROOT, ART, log, get_matrices, tabpfn_oof, tabpfn_test
from pipeline_v9 import to_logit, load_npy, load_csv_probs, EXT, C, c2i
from sklearn.metrics import balanced_accuracy_score

D = EXT/'cdeotte_s6e6-oof-and-test-preds'

def loadD(f, ids_exp):
    d = pd.read_csv(D/f)
    cols = [c for c in d.columns if c in C]
    if 'id' in d.columns: d = d.set_index('id').reindex(ids_exp).reset_index()
    if len(cols)==3: return d[cols].values.astype('float32')
    if d.shape[1]==1: return d.iloc[:,0].values.reshape(-1,3).astype('float32')
    return d.iloc[:,-3:].values.astype('float32')

def main():
    log("="*60)
    log("V10 — TabPFN-3 meta over external bases. The diverse member for the final blend.")
    X, Xt, y, test_ids, feats = get_matrices()
    tr_ids = pd.read_csv(ROOT/'data/train.csv')['id'].values

    pairs = []
    B=EXT/'cdeotte_xgb-v5-for-s6e6';      pairs.append(('xgb5', load_npy(B/'train_oof/xgb-5_oof.npy'), load_npy(B/'test_preds/xgb-5_test_preds.npy')))
    B=EXT/'cdeotte_realmlp-v5-for-s6e6';  pairs.append(('rmlp5', load_npy(B/'train_oof/realmlp-5_oof.npy'), load_npy(B/'test_preds/realmlp-5_test_preds.npy')))
    B=EXT/'cdeotte_xgb-v1-for-s6e6';      pairs.append(('xgb1', load_npy(B/'oof_preds.npy'), load_npy(B/'test_preds.npy')))
    B=EXT/'cdeotte_realmlp-v1-for-s6e6';  pairs.append(('rmlp1', load_npy(B/'oof_preds.npy'), load_npy(B/'test_preds.npy')))
    B=EXT/'yekenot_ps-s6-e6-realmlp-pytorch'
    pairs.append(('yk', load_csv_probs(B/'oof_preds.csv', tr_ids), load_csv_probs(B/'test_preds.csv', test_ids)))
    for nm,fo,ft in [('lgbm5','oof_preds_lgbm5_v1.csv','test_preds_lgbm5_v1.csv'),
                     ('rmlp0','oof_preds_realmlp0_v12.csv','test_preds_realmlp0_v12.csv'),
                     ('rmlp2','oof_preds_realmlp2_v10.csv','test_preds_realmlp2_v10.csv'),
                     ('tabm0','oof_preds_tabm0_v2.csv','test_preds_tabm0_v2.csv')]:
        pairs.append((nm, loadD(fo, tr_ids), loadD(ft, test_ids)))
    pairs.append(('hgb', load_npy(ART/'hgb_oof.npy'), load_npy(ART/'hgb_test.npy')))

    for n,o,t in pairs:
        log(f"  {n}: oof bacc={balanced_accuracy_score(y, o.argmax(1)):.5f}")

    mtr = np.concatenate([to_logit(o) for _,o,_ in pairs] + [X.astype('float32')], 1)
    mte = np.concatenate([to_logit(t) for _,_,t in pairs] + [Xt.astype('float32')], 1)
    log(f"meta: {len(pairs)} bases -> {mtr.shape[1]} cols")

    fo, ft = ART/'tabpfn_v10_oof.npy', ART/'tabpfn_v10_test.npy'
    tp_oof = np.load(fo) if fo.exists() else tabpfn_oof(mtr, y, ctx=30000, n_est=8, seeds=(42,))
    if not fo.exists(): np.save(fo, tp_oof)
    log(f"TabPFN v10 meta OOF = {balanced_accuracy_score(y, tp_oof.argmax(1)):.5f}")
    tp_test = np.load(ft) if ft.exists() else tabpfn_test(mtr, y, mte, ctx=30000, n_est=8, seeds=(42, 1))
    if not ft.exists(): np.save(ft, tp_test)
    pred = np.array(C)[tp_test.argmax(1)]
    pd.DataFrame({'id': test_ids, 'class': pred}).to_csv(ROOT/'submission_v10_tabpfn.csv', index=False)
    log("V10 DONE — artifacts saved; blend tabpfn_v10 into v9 weights next.")

if __name__ == '__main__':
    main()
