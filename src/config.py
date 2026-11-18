"""Config loading + MCU defaults for the energy/latency model."""
import yaml

# Named target MCU (§8). Numbers are datasheet-order-of-magnitude for a Cortex-M4F.
# Override per-config under `mcu:`. All downstream energy is a labeled ESTIMATE.
DEFAULT_MCU = {
    "name": "cortex-m4f@80MHz",
    "freq_mhz": 80,
    "cycles_per_mac": 1.5,        # CMSIS-NN int8, ballpark
    "active_mw": 12.0,            # active core power
    "sleep_uw": 20.0,             # deep-sleep floor
    "gate_uj": 0.5,               # stage-0 gate cost (cascade)
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    mcu = dict(DEFAULT_MCU)
    mcu.update(cfg.get("mcu", {}))
    cfg["mcu"] = mcu
    cfg.setdefault("input_size", 96)
    cfg.setdefault("temporal_frames", 3)
    cfg.setdefault("seeds", [0, 1, 2])
    cfg.setdefault("epochs", 12)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("target_fpr", 0.01)
    cfg.setdefault("cross_regions", ["boreal", "socal_chaparral"])
    cfg.setdefault("duty_cycle", 0.01)
    cfg.setdefault("decisions_per_hour", 3600)
    return cfg
