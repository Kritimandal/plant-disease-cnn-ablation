import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

WARMUP = 20
RUNS = 100
THROUGHPUT_BATCH = 32


def _flops(model, input_shape):
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

        forward = tf.function(lambda x: model(x, training=False)).get_concrete_function(
            tf.TensorSpec([1, *input_shape], tf.float32)
        )
        opts = ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        return int(profile(forward.graph, options=opts).total_float_ops)
    except Exception:
        return None


def _serialized_mb(model):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.h5"
        try:
            model.save(path)
        except Exception:
            path = Path(tmp) / "model.keras"
            model.save(path)
        return round(path.stat().st_size / (1024 * 1024), 3)


def measure(model, input_shape, runs=RUNS, warmup=WARMUP):
    total = model.count_params()
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))

    with tf.device("/CPU:0"):
        single = tf.constant(np.random.rand(1, *input_shape).astype("float32") * 255)
        for _ in range(warmup):
            model(single, training=False)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(single, training=False)
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()

        batch = tf.constant(np.random.rand(THROUGHPUT_BATCH, *input_shape).astype("float32") * 255)
        for _ in range(3):
            model(batch, training=False)
        start = time.perf_counter()
        for _ in range(10):
            model(batch, training=False)
        elapsed = time.perf_counter() - start

    return {
        "total_params": int(total),
        "trainable_params": trainable,
        "frozen_params": int(total) - trainable,
        "model_size_mb": _serialized_mb(model),
        "flops": _flops(model, input_shape),
        "latency_ms_median": round(statistics.median(timings), 3),
        "latency_ms_p90": round(timings[int(0.9 * len(timings)) - 1], 3),
        "throughput_img_per_s": round(THROUGHPUT_BATCH * 10 / elapsed, 1),
        "latency_runs": runs,
        "latency_device": "CPU",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--img-size", type=int, default=128)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    stats = measure(model, (args.img_size, args.img_size, 3))
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
