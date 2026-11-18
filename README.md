# STARE: Streaming Temporal Adaptive Residual Estimation

Code for the paper:

> **STARE: Streaming Temporal Adaptive Residual Estimation for Cross-Camera Wildfire Smoke Detection on Microcontrollers**
> Sakib Mahtab Khandaker and Kayes Bin Yousuf
> *29th IEEE International Conference on Computer and Information Technology (ICCIT), 2026*

## Overview

STARE removes camera-specific background cues via a bank of K exponential moving average backgrounds at log-spaced timescales. The residuals feed a 473-parameter int8 depthwise-separable CNN, reaching 0.991 cross-camera PR-AUC on FIgLib within 251 KB of peak SRAM (STM32F746 Cortex-M7 target).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and TensorFlow 2.21+.

## Data

Download the HPWREN FIgLib dataset:

```bash
python data/download_figlib.py --sequences 202 --out data/raw/figlib
```

Build the processed dataset (96x96, camera-holdout splits):

```bash
python data/make_dataset.py --input data/raw/figlib --out data/processed
```

## Reproducing paper results

**Headline result** (Table II, cross-camera comparison):

```bash
python -m src.run_ladder
```

**K-timescale ablation** (Table IV):

```bash
python -m src.run_ktimescale
```

**Early detection** (Fig. 5):

```bash
python -m src.run_earlydetect
```

**Classical baselines** (MOG2/KNN, Table II):

```bash
python -m src.run_classical_baseline
```

**Invariance baselines** (augmentation/DANN, Table II):

```bash
python -m src.run_invariance_baselines
```

**STARE headline with int8 quantization** (Table I memory, Fig. 2):

```bash
python -m src.run_s3_headline --seeds 3 --alpha 0.001
```

## Generating figures

```bash
python make_figures.py
```

Outputs to `figures/`.

## Project structure

```
stare/
├── data/
│   ├── download_figlib.py      # FIgLib downloader (HPWREN CDN)
│   └── make_dataset.py         # 96x96 dataset builder with camera splits
├── src/
│   ├── features/
│   │   ├── bg_subtraction.py   # EWMA background subtraction (core of STARE)
│   │   ├── color_texture.py    # Color/texture feature extraction
│   │   └── frame_diff.py       # Frame differencing baseline
│   ├── models/
│   │   ├── backbone.py         # Depthwise-separable CNN (budget-fitted)
│   │   ├── cascade.py          # Cascade model variant
│   │   └── streamgate.py       # Stream-gated model variant
│   ├── config.py               # YAML config loader
│   ├── dataio.py               # Data I/O utilities
│   ├── evaluate.py             # Evaluation pipeline
│   ├── metrics.py              # PR-AUC, recall@FPR metrics
│   ├── train.py                # Training loop
│   ├── quantize.py             # Int8 post-training quantization
│   ├── export_tflite.py        # TFLite model export
│   ├── simulate.py             # MCU latency/RAM simulation
│   ├── energy.py               # Energy estimation
│   ├── run_s3_headline.py      # Headline STARE experiment
│   ├── run_ladder.py           # Cross-camera comparison ladder
│   ├── run_ktimescale.py       # K-timescale ablation
│   ├── run_earlydetect.py      # Early detection evaluation
│   ├── run_classical_baseline.py  # MOG2/KNN baselines
│   └── run_invariance_baselines.py # Augmentation/DANN baselines
├── configs/                    # Example experiment configs
├── make_figures.py             # Paper figure generation
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{khandaker2026stare,
  title     = {{STARE}: Streaming Temporal Adaptive Residual Estimation for Cross-Camera Wildfire Smoke Detection on Microcontrollers},
  author    = {Khandaker, Sakib Mahtab and Yousuf, Kayes Bin},
  booktitle = {29th IEEE International Conference on Computer and Information Technology (ICCIT)},
  year      = {2026}
}
```

## Acknowledgments

We thank HPWREN and the FIgLib authors for making the data publicly available.

## License

MIT
