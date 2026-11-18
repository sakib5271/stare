"""Generate the paper's figures from this session's confirmed results (research-spec.md §12).
Numbers are hardcoded from the logged experiments (all reproducible via src/run_s3_headline.py
and the diagnostics). Run: python paper/make_figures.py  ->  paper/figures/*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
RES = "#c0392b"      # STARE / residual
RAW = "#999999"      # raw / baseline
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# --- Fig 1: 2x2 difficulty map (MobileNetV2 @96px) ------------------------------------
def fig_map():
    # rows = label margin, cols = split ; values = PR-AUC
    M = np.array([[0.970, 0.545],    # all-frames: within-camera, cross-camera
                  [0.988, 0.536]])   # clear-frames
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1], ["within-camera\n(random)", "cross-camera\n(holdout)"])
    ax.set_yticks([0, 1], ["all frames", "clear frames"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="black" if M[i, j] > 0.7 else "white")
    ax.set_title("FIgLib difficulty map (MobileNetV2, 96$\\times$96)\ncross-camera = the real problem")
    fig.colorbar(im, ax=ax, label="PR-AUC", shrink=0.8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_difficulty_map.png", dpi=150); plt.close(fig)


# --- Fig 2: headline raw vs residual, cross-camera (92 cams, 3 seeds) ------------------
def fig_headline():
    names = ["raw\nsingle-frame", "slow bg\n($K$=1)", "STARE\n($K$=3)"]
    vals = [0.481, 0.863, 0.991]
    errs = [0.008, 0.016, 0.004]
    cols = [RAW, "#e07b6a", RES]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(names, vals, yerr=errs, color=cols, capsize=4, width=0.6, edgecolor="white")
    ax.axhline(0.50, ls="--", color="k", lw=1)
    ax.text(2.4, 0.51, "chance", fontsize=8, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("cross-camera PR-AUC")
    ax.set_title("Background subtraction restores\ncross-camera detection", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_headline.png", dpi=150); plt.close(fig)


# --- Fig 3: Ablation A — background update rate (alpha) --------------------------------
def fig_alpha():
    a = [0.1, 0.05, 0.02, 0.005, 0.001]
    pr = [0.614, 0.797, 0.859, 0.903, 0.930]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(a, pr, "-o", color=RES, ms=6)
    ax.axhline(0.53, ls="--", color=RAW); ax.text(0.1, 0.545, "raw / chance", fontsize=8, color=RAW)
    ax.set_xscale("log"); ax.invert_xaxis()      # slower background -> right
    ax.set_xlabel("background update rate α  (slower →)")
    ax.set_ylabel("cross-camera PR-AUC"); ax.set_ylim(0.5, 1.0)
    ax.annotate("fast bg absorbs the smoke", (0.1, 0.614), (0.03, 0.66), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_title("(A)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_alpha_ablation.png", dpi=150); plt.close(fig)


# --- Fig 4: Ablation B — camera jitter (static-camera boundary) ------------------------
def fig_jitter():
    j = [0, 2, 4, 8]
    pr = [0.930, 0.851, 0.826, 0.676]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(j, pr, "-o", color=RES, ms=6)
    ax.axhline(0.48, ls="--", color=RAW); ax.text(0.1, 0.50, "raw / chance", fontsize=8, color=RAW)
    ax.set_xlabel("per-frame camera jitter (pixels)")
    ax.set_ylabel("cross-camera PR-AUC"); ax.set_ylim(0.45, 1.0)
    ax.set_title("(B)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_jitter_boundary.png", dpi=150); plt.close(fig)


# --- Fig 5: early detection vs time-since-ignition (slow-only vs two-timescale) --------
def fig_early():
    x = [0, 1, 3, 6, 12, 25]                       # minutes since ignition (bin starts)
    k1 = [0.000, 0.019, 0.062, 0.104, 0.130, 0.207]   # measured, camera-holdout, seed 0
    k2 = [0.033, 0.056, 0.075, 0.098, 0.291, 0.481]
    k3 = [0.033, 0.074, 0.138, 0.337, 0.723, 0.865]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(x, k1, "-o", color="#f0b8ae", ms=5, label="$K$=1 (slow only)")
    ax.plot(x, k2, "-^", color="#e07b6a", ms=5, label="$K$=2")
    ax.plot(x, k3, "-s", color=RES, ms=5, label="$K$=3 (deployed)")
    ax.set_xlabel("minutes since ignition"); ax.set_ylabel("detection rate @1% FPR")
    ax.set_ylim(0, 1); ax.set_title("Early detection vs. time since ignition")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_temporal_growth.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_map(); fig_headline(); fig_alpha(); fig_jitter(); fig_early()
    print(f"wrote 5 figures to {OUT}")
