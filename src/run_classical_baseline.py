"""Independent, named strong competitors: classical ADAPTIVE background subtractors (MOG2, KNN),
the Stauffer-Grimson GMM lineage the paper cites. Given their BEST config for this task (slow
adaptation, so they do not absorb the slow-growing plume), and fed through the SAME |frame - bg|
residual + tiny-CNN pipeline as STARE. The only difference from STARE is the background ESTIMATOR
(online GMM / KNN vs STARE's streaming EWMA bank). Shows whether STARE's specific streaming
multi-timescale design beats an established adaptive background method.

  python -m src.run_classical_baseline --seeds 3 --max-seq 150
"""
import argparse
import numpy as np
import cv2
from sklearn.metrics import average_precision_score

from .run_baselines import load, split, RES, MARGIN
from .run_adaptive import stare_rep
from .run_invariance_baselines import fit_stare, _predict


def classical_rep(seqs, kind, lr=0.001):
    """|frame - adaptive_background| residual using an OpenCV subtractor (slow adaptation)."""
    Xs, ys, cams = [], [], []
    for sq in seqs:
        f, offs, cam = sq["frames"], sq["offs"], sq["cam"]
        f8 = (f * 255).astype("uint8")
        bs = (cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
              if kind == "mog2"
              else cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False))
        R = np.empty_like(f)
        for t in range(len(f)):
            bs.apply(f8[t], learningRate=lr)
            bg = bs.getBackgroundImage()
            bgf = f[t] if bg is None else bg.astype("float32") / 255.0
            R[t] = np.abs(f[t] - bgf)
        mask = np.abs(offs) >= MARGIN
        Xs.append(R[mask]); ys.append((offs[mask] >= 0).astype(int))
        cams.append(np.array([cam] * int(mask.sum())))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cams)


def score(rep_fn, seqs, seeds):
    X, y, cams = rep_fn(seqs)
    tr, te = split(cams, "camera", seed=1)
    aps = [average_precision_score(y[te], _predict(fit_stare(X[tr], y[tr], s), X[te]))
           for s in range(seeds)]
    del X
    return float(np.mean(aps)), float(np.std(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=150)
    args = ap.parse_args()
    seqs = load(args.max_seq)
    print(f"loaded {len(seqs)} sequences, {len({s['cam'] for s in seqs})} cameras", flush=True)

    methods = [
        ("MOG2 (classical GMM, slow)", lambda s: classical_rep(s, "mog2")),
        ("KNN (classical, slow)",      lambda s: classical_rep(s, "knn")),
        ("STARE (K=3, deployed)",      stare_rep),
    ]
    print("\n===== classical adaptive background competitors vs STARE (cross-camera PR-AUC) =====", flush=True)
    for name, fn in methods:
        mu, sd = score(fn, seqs, args.seeds)
        print(f"  {name:<30} {mu:.3f} ± {sd:.3f}", flush=True)


if __name__ == "__main__":
    main()
