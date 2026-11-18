"""Model registry: mechanism -> (input builder, keras model).

Every deployed mechanism uses the SAME budget-fitted tiny CNN backbone so the
ablation isolates the *mechanism*, not the architecture. Front-ends differ:

  single-frame  : raw 96x96x3 image
  +temporal     : frame_diff.temporal_stack -> 96x96x(3+3*(T-1))   (H1)
  +handcrafted  : raw image + 11-d handcrafted feature vector       (H3)
  cascade       : stage-0 gate + stage-1 single-frame CNN           (H2, second track)

`build_mechanism` returns a dict with everything run_cell needs.
"""
from .backbone import build_for_budget, count_params
from ..features.frame_diff import temporal_channels
from ..features import HANDCRAFTED_DIM

MECHANISMS = ("single", "temporal", "handcrafted", "cascade", "streamgate")


def build_mechanism(mechanism, budget_kb, input_size=96, temporal_frames=3):
    """Return {'model', 'meta', 'inputs'} for a mechanism at a memory budget.
    'inputs' names which prepared arrays run_cell must feed:
       'single'      -> X_single
       'temporal'    -> X_temporal (stacked)
       'handcrafted' -> [X_single, F_handcrafted]
    """
    h = w = input_size
    if mechanism == "single":
        model, meta = build_for_budget((h, w, 3), budget_kb, name="single")
        return {"model": model, "meta": meta, "inputs": "single"}
    if mechanism == "temporal":
        c = temporal_channels(temporal_frames)
        model, meta = build_for_budget((h, w, c), budget_kb, name="temporal")
        meta["temporal_frames"] = temporal_frames
        return {"model": model, "meta": meta, "inputs": "temporal"}
    if mechanism == "handcrafted":
        model, meta = build_for_budget((h, w, 3), budget_kb,
                                       extra_feat_dim=HANDCRAFTED_DIM, name="handcrafted")
        meta["handcrafted_dim"] = HANDCRAFTED_DIM
        return {"model": model, "meta": meta, "inputs": "handcrafted"}
    if mechanism == "streamgate":
        from .streamgate import build_streamgate
        model, meta = build_streamgate(input_size, budget_kb, temporal_frames=temporal_frames)
        return {"model": model, "meta": meta, "inputs": "stack"}
    if mechanism == "cascade":
        # H2 (second track). Stage-1 is a single-frame CNN; the sub-KB gate lives in
        # src/models/cascade.py and is analysed for energy in src/energy.py.
        from .cascade import build_cascade
        return build_cascade(budget_kb, input_size)
    raise ValueError(f"unknown mechanism {mechanism!r}; choose from {MECHANISMS}")
