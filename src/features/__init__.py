"""Fixed-DSP front-ends (no learned params): cheap enough for an MCU."""
from .frame_diff import temporal_stack
from .color_texture import handcrafted_features, HANDCRAFTED_DIM

__all__ = ["temporal_stack", "handcrafted_features", "HANDCRAFTED_DIM"]
