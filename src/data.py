import csv
import json
import random
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"

AUTOTUNE = tf.data.AUTOTUNE


def load_classes():
    with open(SPLIT_DIR / "split_meta.json", encoding="utf-8") as f:
        return json.load(f)["classes"]


def _read_manifest(split, classes):
    index = {name: i for i, name in enumerate(classes)}
    paths, labels = [], []
    with open(SPLIT_DIR / f"{split}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(str(ROOT / row["filepath"]))
            labels.append(index[row["class"]])
    return paths, labels


def _decode(path, label, img_size, num_classes):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, img_size, method="bilinear")
    return tf.cast(image, tf.uint8), tf.one_hot(label, num_classes)


def _cache_bytes(count, img_size):
    return count * img_size[0] * img_size[1] * 3


CACHE_LIMIT_BYTES = 3 * 1024 ** 3


def build_augmenter(seed):
    return tf.keras.Sequential([
        RandomFlip("horizontal", seed=seed),
        RandomRotation(0.1, seed=seed),
        RandomZoom(0.1, seed=seed),
    ], name="augmentation")


def make_dataset(split, img_size, batch_size, seed, classes=None, augment=False, shuffle=False):
    classes = classes or load_classes()
    paths, labels = _read_manifest(split, classes)
    num_classes = len(classes)

    if shuffle:
        order = list(range(len(paths)))
        random.Random(seed).shuffle(order)
        paths = [paths[i] for i in order]
        labels = [labels[i] for i in order]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: _decode(p, l, img_size, num_classes), num_parallel_calls=AUTOTUNE)

    if _cache_bytes(len(paths), img_size) <= CACHE_LIMIT_BYTES:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE), len(paths)


def true_labels(split, classes=None):
    classes = classes or load_classes()
    _, labels = _read_manifest(split, classes)
    return labels
