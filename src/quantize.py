"""Post-training int8 quantization (TFLite-Micro target) + float->int8 accuracy delta.

Full-integer PTQ (int8 weights AND activations) is what actually runs on a Cortex-M
via TFLM. We record the .tflite byte size (flash proxy) and the PR-AUC drop from
float32 to int8 — itself a reportable result per §8.
"""
import numpy as np
import tensorflow as tf


def _rep_dataset(X, n=200):
    """Yield representative samples for the converter. X is an array or list of arrays."""
    multi = isinstance(X, (list, tuple))
    m = len(X[0]) if multi else len(X)
    idx = np.random.default_rng(0).choice(m, size=min(n, m), replace=False)
    def gen():
        for i in idx:
            if multi:
                yield [np.asarray(x[i:i + 1], "float32") for x in X]
            else:
                yield [np.asarray(X[i:i + 1], "float32")]
    return gen


def to_int8(model, rep_inputs, n_rep=200):
    """Return (tflite_bytes, size_kb)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = _rep_dataset(rep_inputs, n_rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    return blob, round(len(blob) / 1024.0, 1)


def tflite_predict(blob, X):
    """Run int8 tflite over X (array or list of arrays) in a single batched invoke.
    Returns sigmoid scores (N,)."""
    interp = tf.lite.Interpreter(model_content=blob)
    in_details = interp.get_input_details()
    out_det = interp.get_output_details()[0]
    arrays = list(X) if isinstance(X, (list, tuple)) else [X]
    n = len(arrays[0])

    # resize every input tensor to the full batch, then allocate once
    for det, arr in zip(in_details, arrays):
        interp.resize_tensor_input(det["index"], list(np.asarray(arr).shape))
    interp.allocate_tensors()
    for det, arr in zip(in_details, arrays):
        s_in, z_in = det["quantization"]
        q = np.round(np.asarray(arr, "float32") / s_in + z_in).astype(det["dtype"])
        interp.set_tensor(det["index"], q)
    interp.invoke()

    s_out, z_out = out_det["quantization"]
    raw = interp.get_tensor(out_det["index"]).astype("float32").reshape(n)
    logits = (raw - z_out) * s_out
    return 1.0 / (1.0 + np.exp(-logits))
