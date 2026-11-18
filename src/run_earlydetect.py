"""Early detection as a first-class result (deployed K=3, camera-holdout, multi-seed).

For each held-out fire we set a 1% FPR threshold on the test negatives and report:
  (a) detection rate vs minutes since ignition (the curve), and
  (b) time-to-detection: the first post-ignition minute each fire crosses the threshold, the median
      over fires, and the fraction of fires detected within 5 and 10 minutes.
This is the operationally meaningful number and lets us compare to SmokeyNet's 3.12 min (which,
unlike ours, is measured with the same cameras in train and test).

  python -m src.run_earlydetect --seeds 3 --max-seq 160
"""
import argparse
import numpy as np
from sklearn.metrics import average_precision_score

from .run_baselines import load, RES, MARGIN
from .run_invariance_baselines import fit_stare, _predict

RATES = (0.001, 0.0141, 0.2)          # deployed K=3
BINS = [0, 1, 3, 6, 12, 25]           # minutes since ignition (bin left edges)


def kres(f, rates=RATES):
    """K-background residuals for every frame of one sequence (n, RES, RES, 3K)."""
    n = len(f); K = len(rates); B = [None] * K
    res = np.empty((n, RES, RES, 3 * K), "float32")
    for t in range(n):
        for k, a in enumerate(rates):
            bk = f[t] if B[k] is None else B[k]
            res[t, :, :, 3 * k:3 * k + 3] = np.abs(f[t] - bk)
            B[k] = f[t] if B[k] is None else (1 - a) * B[k] + a * f[t]
    return res


def test_cameras(cam_list, seed):
    """Same camera-holdout as split(): hold out a different fifth of cameras per seed."""
    rng = np.random.default_rng(seed)
    u = sorted(set(cam_list))
    k = max(1, len(u) // 5)
    return {u[i] for i in rng.permutation(len(u))[:k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=160)
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()
    seqs = load(args.max_seq)
    cams = [s["cam"] for s in seqs]
    print(f"loaded {len(seqs)} sequences, {len(set(cams))} cameras", flush=True)

    med_ttd, frac5, frac10, ndet = [], [], [], []
    bin_rates = []
    for seed in range(args.seeds):
        te_c = test_cameras(cams, seed)
        # train on clear frames of train cameras
        Xtr, ytr = [], []
        for s in seqs:
            if s["cam"] in te_c:
                continue
            r = kres(s["frames"]); m = np.abs(s["offs"]) >= MARGIN
            Xtr.append(r[m]); ytr.append((s["offs"][m] >= 0).astype(int))
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        model = fit_stare(Xtr, ytr, seed); del Xtr, ytr

        # score held-out fires; collect negatives for the threshold
        per_fire = []           # (offsets_sec, scores) for each test fire
        neg = []
        for s in seqs:
            if s["cam"] not in te_c:
                continue
            r = kres(s["frames"]); p = _predict(model, r); o = s["offs"]
            per_fire.append((o, p)); neg.append(p[o < 0]); del r
        thr = float(np.quantile(np.concatenate(neg), 1 - args.fpr))

        # per-fire time-to-detection + binned detection rate
        ttds = []
        for o, p in per_fire:
            post = o >= 0
            hit = post & (p >= thr)
            ttds.append(o[hit].min() / 60.0 if hit.any() else np.inf)
        ttds = np.array(ttds)
        det = ttds[np.isfinite(ttds)]
        med_ttd.append(float(np.median(det)) if det.size else np.inf)
        frac5.append(float((ttds <= 5).mean())); frac10.append(float((ttds <= 10).mean()))
        ndet.append(float(np.isfinite(ttds).mean()))
        # binned rate over all post-ignition frames
        allo = np.concatenate([o for o, _ in per_fire]); allp = np.concatenate([p for _, p in per_fire])
        pos = allo >= 0; mins = allo[pos] / 60.0; dhit = allp[pos] >= thr
        edges = BINS + [1e9]
        br = [float(dhit[(mins >= edges[i]) & (mins < edges[i + 1])].mean()) for i in range(len(BINS))]
        bin_rates.append(br)
        print(f"  seed {seed}: {len(per_fire)} held-out fires | median TTD {med_ttd[-1]:.1f} min | "
              f"det@5min {frac5[-1]:.2f} | det@10min {frac10[-1]:.2f} | ever-detected {ndet[-1]:.2f}",
              flush=True)

    br = np.array(bin_rates)
    print("\n===== EARLY DETECTION (deployed K=3, camera-holdout, "
          f"{args.seeds} seeds) =====", flush=True)
    print(f"  median time-to-detection : {np.mean(med_ttd):.1f} +- {np.std(med_ttd):.1f} min", flush=True)
    print(f"  fires detected within 5min: {np.mean(frac5):.2f} +- {np.std(frac5):.2f}", flush=True)
    print(f"  fires detected within 10min:{np.mean(frac10):.2f} +- {np.std(frac10):.2f}", flush=True)
    print(f"  fires ever detected        : {np.mean(ndet):.2f} +- {np.std(ndet):.2f}", flush=True)
    print("  detection rate by minute-bin (mean over seeds):", flush=True)
    for i, mn in enumerate(BINS):
        print(f"    >= {mn:>2} min : {br[:, i].mean():.3f}", flush=True)
    print("PYLINE_K3 =", [round(br[:, i].mean(), 3) for i in range(len(BINS))], flush=True)


if __name__ == "__main__":
    main()
