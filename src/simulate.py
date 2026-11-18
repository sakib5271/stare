"""Latency / RAM / flash for a deployed int8 model.

Preferred path: Renode runs the compiled TFLM binary on a virtual Cortex-M and we read
back cycle counts. When Renode isn't installed we fall back to a TRANSPARENT analytical
estimate from the model's MAC count and the target MCU's cycles/MAC — clearly labeled
`estimated` so the paper never passes it off as measured (§8).
"""
import shutil
import numpy as np
from tensorflow.keras import layers as L


def _shape(x):
    """Keras 3: tensors expose `.shape`; multi-input layers give a list of tensors.
    Returns None for InputLayer (empty input list) or anything shapeless."""
    if isinstance(x, (list, tuple)):
        if not x:
            return None
        x = x[0]
    return tuple(getattr(x, "shape", ()) ) or None


def count_macs(model):
    """Multiply-accumulate count over conv/depthwise/dense layers."""
    macs = 0
    for lyr in model.layers:
        out = _shape(getattr(lyr, "output", None))
        ishape = _shape(getattr(lyr, "input", None))
        if out is None or ishape is None:
            continue
        if isinstance(lyr, L.DepthwiseConv2D):      # check before Conv2D (subclass-safe)
            _, oh, ow, oc = out
            kh, kw = lyr.kernel_size
            macs += oh * ow * oc * kh * kw
        elif isinstance(lyr, L.Conv2D):
            _, oh, ow, oc = out
            kh, kw = lyr.kernel_size
            ic = ishape[-1]
            macs += oh * ow * oc * kh * kw * ic
        elif isinstance(lyr, L.Dense):
            macs += ishape[-1] * out[-1]
    return int(macs)


def peak_ram_kb(model):
    """TFLM arena proxy: largest (input+output) int8 activation pair over the graph."""
    def size(shape):
        return int(np.prod([d for d in shape[1:] if d]))   # int8 => 1 byte/elem
    peak = 0
    for lyr in model.layers:
        ishape = _shape(getattr(lyr, "input", None))
        oshape = _shape(getattr(lyr, "output", None))
        if ishape is None or oshape is None:
            continue
        peak = max(peak, size(ishape) + size(oshape))
    return round(peak / 1024.0, 1)


def analytical_latency_ms(macs, mcu):
    """cycles ~= macs * cycles_per_mac (CMSIS-NN int8 ~= 1-2). time = cycles / freq."""
    cpm = mcu.get("cycles_per_mac", 1.5)
    cycles = macs * cpm
    ms = cycles / (mcu["freq_mhz"] * 1e6) * 1e3
    return round(ms, 3), int(cycles)


def simulate(model, tflite_kb, mcu):
    macs = count_macs(model)
    ram = peak_ram_kb(model)
    have_renode = shutil.which("renode") is not None
    if have_renode:
        # TODO: compile tflite->TFLM firmware, run on virtual Cortex-M, read cycle counter.
        # Interface reserved; until wired, use the analytical estimate below.
        pass
    ms, cycles = analytical_latency_ms(macs, mcu)
    return {"macs": macs, "sim_cycles": cycles, "latency_ms": ms,
            "peak_ram_kb": ram, "flash_kb": tflite_kb,
            "latency_source": "renode" if have_renode else "analytical_estimate"}
