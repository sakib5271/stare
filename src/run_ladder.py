"""Consolidated competitor LADDER on the full 92-camera set with 5-fold camera-holdout, so every
number is consistent with the deployed headline (STARE K=3 = 0.991 +- 0.004). Groups baselines by
category to show precisely which ingredient matters:

  no invariance (raw / augmentation / DANN)  ->  non-adaptive background (batch-median)
  ->  classical ADAPTIVE background (MOG2 / KNN)  ->  STARE (streaming multi-timescale)

Each representation is computed once, then a fresh classifier is trained per fold (folds hold out
different cameras). Results are written incrementally to paper/ladder_results.txt.

  python -m src.run_ladder --folds 5 --max-seq 202
"""
import argparse
import os
import numpy as np
from sklearn.metrics import average_precision_score

from .run_baselines import load, split, make_rep
from .run_adaptive import stare_rep
from .run_invariance_baselines import fit_stare, fit_plain, fit_dann, _predict
from .run_classical_baseline import classical_rep
from .run_strong_baseline import batch_bg_rep, stare_k1_rep

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "ladder_results.txt")


def _plain(X, y, cams, tr, s):  return fit_stare(X[tr], y[tr], s)
def _aug(X, y, cams, tr, s):    return fit_plain(X[tr], y[tr], s, aug=True)
def _dann(X, y, cams, tr, s):   return fit_dann(X[tr], y[tr], cams[tr], s)


def folds_eval(X, y, cams, fit_fn, nf):
    aps = []
    for f in range(nf):
        tr, te = split(cams, "camera", seed=f)
        m = fit_fn(X, y, cams, tr, f)
        aps.append(average_precision_score(y[te], _predict(m, X[te])))
    return float(np.mean(aps)), float(np.std(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-seq", type=int, default=202)
    args = ap.parse_args()
    seqs = load(args.max_seq)
    ncam = len({s["cam"] for s in seqs})
    print(f"loaded {len(seqs)} sequences, {ncam} cameras; {args.folds}-fold camera-holdout", flush=True)
    log = open(OUT, "w", encoding="utf-8")
    log.write(f"# competitor ladder, {len(seqs)} seq / {ncam} cam, {args.folds}-fold camera-holdout\n")
    log.flush()

    def emit(name, X, y, cams, fit_fn):
        mu, sd = folds_eval(X, y, cams, fit_fn, args.folds)
        line = f"  {name:<34} {mu:.3f} +- {sd:.3f}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    # --- raw group (raw / aug / DANN share the raw representation) ---
    Xr, yr, cr = make_rep(seqs, "raw")
    emit("raw (no invariance)",        Xr, yr, cr, _plain)
    emit("raw + augmentation",         Xr, yr, cr, _aug)
    emit("raw + domain-adversarial",   Xr, yr, cr, _dann)
    del Xr

    # --- non-adaptive background ---
    Xb, yb, cb = batch_bg_rep(seqs, "median")
    emit("batch-median bg (non-causal)", Xb, yb, cb, _plain); del Xb

    # --- classical adaptive background ---
    Xm, ym, cm = classical_rep(seqs, "mog2")
    emit("MOG2 (classical adaptive)",  Xm, ym, cm, _plain); del Xm
    Xk, yk, ck = classical_rep(seqs, "knn")
    emit("KNN (classical adaptive)",   Xk, yk, ck, _plain); del Xk

    # --- STARE (streaming) ---
    X1, y1, c1 = stare_k1_rep(seqs)
    emit("STARE K=1 (streaming slow)", X1, y1, c1, _plain); del X1
    X3, y3, c3 = stare_rep(seqs)
    emit("STARE K=3 (deployed)",       X3, y3, c3, _plain); del X3

    log.write("DONE\n"); log.close()
    print("\nDONE -> " + OUT, flush=True)


if __name__ == "__main__":
    main()
