import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as data_lib
from benchmark import measure
from models import ARMS, build

RESULTS_DIR = ROOT / "results"


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(round(time.perf_counter() - self._start, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    classes = data_lib.load_classes()
    img_size = (args.img_size, args.img_size)
    input_shape = (args.img_size, args.img_size, 3)

    train_ds, n_train = data_lib.make_dataset(
        "train", img_size, args.batch_size, args.seed, classes, augment=True, shuffle=True)
    val_ds, n_val = data_lib.make_dataset(
        "val", img_size, args.batch_size, args.seed, classes)
    test_ds, n_test = data_lib.make_dataset(
        "test", img_size, args.batch_size, args.seed, classes)

    run_dir = RESULTS_DIR / f"{args.arm}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "best.weights.h5"

    model = build(args.arm, input_shape=input_shape, num_classes=len(classes))
    print(f"\n{'=' * 64}\n{args.arm}: {ARMS[args.arm][0]}  (seed {args.seed})\n{'=' * 64}")
    print(f"train {n_train:,} | val {n_val:,} | test {n_test:,} | {len(classes)} classes")

    timer = EpochTimer()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_path), monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        timer,
    ]

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)
    train_seconds = time.perf_counter() - start

    val_acc = history.history["val_accuracy"]
    best_epoch = int(np.argmax(val_acc)) + 1
    model.load_weights(weights_path)

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    assert len(y_true) == len(y_pred), f"{len(y_true)} labels vs {len(y_pred)} predictions"

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    efficiency = measure(model, input_shape)

    results = {
        "arm": args.arm,
        "description": ARMS[args.arm][0],
        "seed": args.seed,
        "protocol": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "selection": "best val_accuracy epoch, no early stopping",
        },
        "training": {
            "total_seconds": round(train_seconds, 1),
            "seconds_per_epoch_median": round(float(np.median(timer.times)), 1),
            "epoch_seconds": timer.times,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(float(max(val_acc)), 5),
            "final_train_accuracy": round(float(history.history["accuracy"][-1]), 5),
            "final_val_accuracy": round(float(val_acc[-1]), 5),
            "generalization_gap": round(
                float(history.history["accuracy"][best_epoch - 1] - val_acc[best_epoch - 1]), 5),
        },
        "test": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
            "macro_precision": round(report["macro avg"]["precision"], 5),
            "macro_recall": round(report["macro avg"]["recall"], 5),
            "macro_f1": round(report["macro avg"]["f1-score"], 5),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 5),
        },
        "efficiency": efficiency,
        "per_class_f1": {c: round(report[c]["f1-score"], 5) for c in classes},
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(run_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)

    t = results["test"]
    e = results["efficiency"]
    print(f"\ntest accuracy {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f}")
    print(f"params {e['total_params']:,} | {e['model_size_mb']} MB | "
          f"{e['latency_ms_median']} ms/image")
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
