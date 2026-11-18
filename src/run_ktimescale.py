"""K-timescale ablation (v2, framework generalization #1).

Generalizes STARE from two backgrounds to a bank of K exponential backgrounds at
LOG-SPACED rates alpha_1<...<alpha_K (a temporal filter bank). Each background is a
temporal low-pass; the residuals |x_t - B_k| are temporal high-passes at different cutoffs,
so the stacked residuals act as a temporal band-pass that isolates the slow smoke signature.
This ablation shows how cross-camera PR-AUC and cost scale with K, and that K=2 is the
sweet spot at a microcontroller budget (each extra background costs ~27 KB of state).

  python -m src.run_ktimescale --seeds 1 --max-seq 120
"""
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from .run_baselines import load, split, RES, MARGIN
from .models.backbone import build_for_budget
from .simulate import simulate
from .config import DEFAULT_MCU

ALPHA_MIN, ALPHA_MAX = 0.001, 0.2       # slowest (camera invariance) .. fastest (onset)


def make_krep(seqs, rates):
    """Per-frame residuals against K causal EWMA backgrounds -> (N, H, W, 3K), clear frames."""
    K = len(rates)
    Xs, ys, cams = [], [], []
    for sq in seqs:
        f, offs, cam = sq["frames"], sq["offs"], sq["cam"]
        n = len(f)
        B = [None] * K
        res = np.empty((n, RES, RES, 3 * K), "float32")
        for t in range(n):
            for k, a in enumerate(rates):
                bk = f[t] if B[k] is None else B[k]
                res[t, :, :, 3*k:3*k+3] = np.abs(f[t] - bk)
                B[k] = f[t] if B[k] is None else (1 - a) * B[k] + a * f[t]
        mask = np.abs(offs) >= MARGIN
        Xs.append(res[mask]); ys.append((offs[mask] >= 0).astype(int))
        cams.append(np.array([cam] * int(mask.sum())))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=120)
    ap.add_argument("--budget", type=int, default=64)
    args = ap.parse_args()
    seqs = load(args.max_seq)
    print(f"loaded {len(seqs)} sequences, {len({s['cam'] for s in seqs})} cameras")
    mcu = dict(DEFAULT_MCU)

    print("\n===== K-TIMESCALE ABLATION (camera-holdout) =====")
    print(f"{'K':>2}  {'rates (alpha)':<28}{'PR-AUC':>14}{'params':>8}{'peakRAM':>9}{'lat.':>8}")
    for K in [1, 2, 3, 4]:
        rates = list(np.geomspace(ALPHA_MIN, ALPHA_MAX, K)) if K > 1 else [ALPHA_MIN]
        X, y, cams = make_krep(seqs, rates)
        tr, te = split(cams, "camera")
        aps = []
        last = None
        for s in range(args.seeds):
            tf.keras.utils.set_random_seed(s)
            m, meta = build_for_budget((RES, RES, 3 * K), args.budget, name=f"k{K}s{s}")
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
            pos = y[tr].sum(); neg = len(tr) - pos
            m.fit(X[tr], y[tr], epochs=18, batch_size=64, verbose=0,
                  class_weight={0: 1.0, 1: neg / max(pos, 1)})
            p = 1 / (1 + np.exp(-m.predict(X[te], verbose=0).ravel()))
            aps.append(average_precision_score(y[te], p)); last = (m, meta)
        m, meta = last
        sim = simulate(m, meta["est_flash_kb"], mcu)
        ram = round(sim["peak_ram_kb"] + K * RES * RES * 3 / 1024.0, 1)   # + K background buffers
        rate_str = ", ".join(f"{r:.3g}" for r in rates)
        print(f"{K:>2}  {rate_str:<28}{np.mean(aps):>8.3f}±{np.std(aps):<4.3f}"
              f"{meta['params']:>8}{ram:>8.0f}K{sim['latency_ms']:>7.1f}ms", flush=True)
        del X, y, cams


if __name__ == "__main__":
    main()
