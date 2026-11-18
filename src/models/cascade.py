"""Cascade mechanism (H2 — the second, energy-at-fixed-accuracy track).

Stage-0 gate: a sub-KB always-on trigger (here a tiny logistic model over the same
cheap handcrafted stats, standing in for a scalar-sensor MLP or frame-difference
'something changed' trigger). Stage-1: the tiny single-frame CNN, run only when the
gate fires. Energy lives in src/energy.py, which uses the measured gate wake-rate.

Kept minimal for now: the H1/H3 track is the current focus. This wires the interface
so the cascade rows of the matrix can be filled later without refactoring.
"""
from .backbone import build_for_budget
from ..features import HANDCRAFTED_DIM


def build_cascade(budget_kb, input_size=96):
    h = w = input_size
    # Stage-1 confirm CNN gets ~all the budget (gate is sub-KB, negligible flash).
    model, meta = build_for_budget((h, w, 3), budget_kb, name="cascade_stage1")
    meta["cascade"] = True
    meta["gate"] = {"type": "handcrafted_logistic", "dim": HANDCRAFTED_DIM,
                    "flash_kb": round((HANDCRAFTED_DIM + 1) / 1024.0, 3)}
    # 'inputs' == 'single' for stage-1; the gate is evaluated separately in energy.py.
    return {"model": model, "meta": meta, "inputs": "single"}
