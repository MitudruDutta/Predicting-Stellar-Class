"""
v7: meta-of-everything. Stack ALL accumulated leak-free OOF signal into one final TabPFN.
Members (all proper 5-fold OOF, leak-free):
  - v4 tabpfn meta (0.9660)   artifacts/tabpfn_oof/test
  - v5 tabpfn meta (0.9658)   artifacts/tabpfn_v5_oof/test
  - v6 diverse bases: xgbB hgb2 catB mlp6 lr knn   artifacts_v6/*
  - v4 strong bases: hgb (0.9642)  artifacts/hgb_*
Final = TabPFN over [all these OOF logits + raw 33 feats].
Combines every model built so far. Cheap (bases already trained).
"""
import os, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

import pipeline as P
from pipeline import (ROOT, ART, log, get_matrices, to_logit, N_CLASSES, SEED,
                      tabpfn_oof, tabpfn_test, write_sub)
from sklearn.metrics import balanced_accuracy_score

ART6 = ROOT / 'artifacts_v6'


def load_pair(folder, name):
    o = np.load(folder / f'{name}_oof.npy')
    t = np.load(folder / f'{name}_test.npy')
    return o.astype('float32'), t.astype('float32')


def main():
    log("=" * 60)
    log("PIPELINE V7 — meta-of-everything. Target beat 0.97127")
    X, Xt, y, ids, feats = get_matrices()   # 33 raw feats for context

    members = []   # (name, oof, test)
    # v4 + v5 metas
    for nm, fol, key in [('m_v4', ART, 'tabpfn'), ('m_v5', ART, 'tabpfn_v5')]:
        o, t = load_pair(fol, key)
        members.append((nm, o, t))
    # v6 bases (diverse + strong)
    for nm in ['xgbB', 'hgb2', 'catB', 'mlp6', 'lr', 'knn']:
        o, t = load_pair(ART6, nm)
        members.append((nm, o, t))
    # v4 hgb (strongest v4 base)
    o, t = load_pair(ART, 'hgb')
    members.append(('hgb', o, t))

    log("members (OOF bacc):")
    for nm, o, _ in members:
        log(f"  {nm}: {balanced_accuracy_score(y, o.argmax(1)):.4f}")

    # avg-blend of members (sanity)
    avg = sum(o for _, o, _ in members) / len(members)
    log(f"member avg-blend OOF={balanced_accuracy_score(y, avg.argmax(1)):.4f}")

    # build meta features = all member OOF logits + raw feats
    mtr = np.concatenate([to_logit(o) for _, o, _ in members] + [X.astype('float32')], axis=1).astype('float32')
    mte = np.concatenate([to_logit(t) for _, _, t in members] + [Xt.astype('float32')], axis=1).astype('float32')
    log(f"v7 meta: {len(members)} members -> {mtr.shape[1]} cols")

    # TabPFN meta — 1 OOF seed to check, then test if good
    fo, ft = ART / 'tabpfn_v7_oof.npy', ART / 'tabpfn_v7_test.npy'
    tp_oof = np.load(fo) if fo.exists() else tabpfn_oof(mtr, y, ctx=30000, n_est=16, seeds=(42,))
    if not fo.exists(): np.save(fo, tp_oof)
    v7_ba = balanced_accuracy_score(y, tp_oof.argmax(1))
    log(f"TabPFN v7 meta OOF={v7_ba:.4f}  (v4 0.9660, v5 0.9658, target LB 0.97127)")

    tp_test = np.load(ft) if ft.exists() else tabpfn_test(mtr, y, mte, ctx=30000, n_est=16, seeds=(42, 1, 7))
    if not ft.exists(): np.save(ft, tp_test)

    write_sub(ids, tp_test, ROOT / 'submission_v7.csv')

    # also a simple logit-average of the two best metas + member-avg, as alt sub
    alt = (members[0][2] + members[1][2]) / 2   # v4+v5 test avg
    write_sub(ids, alt, ROOT / 'submission_v7_metaavg.csv')

    log(f"submission_v7.csv OOF={v7_ba:.4f}")
    log("PIPELINE V7 DONE")


if __name__ == '__main__':
    main()
