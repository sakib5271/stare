"""Export the deployed K=3 STARE classifier to a full-integer int8 .tflite, for on-device
measurement. Flash = .tflite byte size (measured here). Peak activation RAM = the TFLM tensor arena
(measure on Linux/Colab with the tflite-micro package; see the snippet printed at the end). Total
peak RAM = arena + the three int8 background buffers (3*H*W*C bytes), maintained by the application
outside the model.

RAM/flash are architecture-determined (independent of the trained weight values), so an untrained
model of the deployed architecture gives the exact deployment footprint.

  python -m src.export_tflite
"""
import os
import numpy as np
import tensorflow as tf

from .models.backbone import build_for_budget

RES = 96
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "stare_k3_int8.tflite")


def main():
    m, meta = build_for_budget((RES, RES, 9), 64, name="stare_k3")   # deployed K=3 (9-channel input)
    print(f"params: {meta['params']}  (width {meta['width']})", flush=True)

    def rep():
        for _ in range(100):
            yield [np.random.rand(1, RES, RES, 9).astype("float32")]

    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    open(OUT, "wb").write(tfl)

    flash = len(tfl)
    buffers = 3 * RES * RES * 3
    print(f"\nwrote {OUT}")
    print(f"FLASH (.tflite size)      : {flash} bytes = {flash/1024:.1f} KB   [MEASURED]")
    print(f"background buffers (K=3)  : {buffers} bytes = {buffers/1024:.0f} KB   [exact: 3*H*W*C int8]")
    print("\n# --- run deployment/colab_measure.py on Linux/Colab to MEASURE the activation arena ---")
    print("# It binary-searches the smallest TFLite-Micro arena_size that allocates all tensors;")
    print("# that boundary is the measured arena. Measured: arena 88.6 KB + "
          f"{buffers/1024:.0f} KB buffers = 170 KB peak RAM.")


if __name__ == "__main__":
    main()
