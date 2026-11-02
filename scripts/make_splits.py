import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "archive" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
OUT_DIR = ROOT / "data" / "splits"

AUG_SUFFIX = re.compile(r"_(\d+deg|flipTB|flipLR|newPixel\d*|new\w+|\w*[Nn]oise\w*)$")


def source_id(filename):
    return filename.split("___")[0]


def is_augmented(stem):
    return AUG_SUFFIX.search(stem) is not None


def scan():
    groups = defaultdict(list)
    group_class = {}

    for split_dir in ("train", "valid"):
        for class_dir in sorted((DATA_DIR / split_dir).iterdir()):
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                uid = source_id(path.name)
                prev = group_class.setdefault(uid, class_dir.name)
                if prev != class_dir.name:
                    raise ValueError(
                        f"source image {uid} appears under two classes: {prev} / {class_dir.name}"
                    )
                groups[uid].append(path)

    return groups, group_class


def assign(groups, group_class, val_frac, test_frac, seed):
    by_class = defaultdict(list)
    for uid in groups:
        by_class[group_class[uid]].append(uid)

    rng = random.Random(seed)
    assignment = {}
    for class_name in sorted(by_class):
        uids = sorted(by_class[class_name])
        rng.shuffle(uids)
        n = len(uids)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for uid in uids[:n_test]:
            assignment[uid] = "test"
        for uid in uids[n_test:n_test + n_val]:
            assignment[uid] = "val"
        for uid in uids[n_test + n_val:]:
            assignment[uid] = "train"
    return assignment


def build_rows(groups, group_class, assignment, cap_per_class, seed):
    rows = {"train": [], "val": [], "test": []}
    for uid, paths in groups.items():
        split = assignment[uid]
        for path in paths:
            augmented = is_augmented(path.stem)
            if split != "train" and augmented:
                continue
            rows[split].append({
                "filepath": path.relative_to(ROOT).as_posix(),
                "class": group_class[uid],
                "source_id": uid,
                "augmented": int(augmented),
            })

    if cap_per_class:
        rng = random.Random(seed + 1)
        capped = []
        by_class = defaultdict(list)
        for row in rows["train"]:
            by_class[row["class"]].append(row)
        for class_name in sorted(by_class):
            items = sorted(by_class[class_name], key=lambda r: r["filepath"])
            rng.shuffle(items)
            capped.extend(items[:cap_per_class])
        rows["train"] = capped

    for split in rows:
        rows[split].sort(key=lambda r: r["filepath"])
    return rows


def verify(rows):
    ids = {s: {r["source_id"] for r in rows[s]} for s in rows}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} source images shared between {a} and {b}")
    for split in ("val", "test"):
        if any(r["augmented"] for r in rows[split]):
            raise AssertionError(f"{split} contains augmented images")
    print("leakage check passed: no source image is shared across splits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--cap-per-class", type=int, default=300,
                    help="max training files per class (reduced CPU protocol); 0 disables")
    args = ap.parse_args()

    groups, group_class = scan()
    print(f"scanned {sum(len(v) for v in groups.values()):,} files "
          f"from {len(groups):,} source photographs")

    assignment = assign(groups, group_class, args.val_frac, args.test_frac, args.seed)
    rows = build_rows(groups, group_class, assignment, args.cap_per_class, args.seed)
    verify(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with open(OUT_DIR / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filepath", "class", "source_id", "augmented"])
            w.writeheader()
            w.writerows(items)

    classes = sorted({r["class"] for r in rows["test"]})
    meta = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "cap_per_class": args.cap_per_class,
        "num_classes": len(classes),
        "classes": classes,
        "counts": {s: len(v) for s, v in rows.items()},
        "source_photographs": {
            s: len({r["source_id"] for r in rows[s]}) for s in rows
        },
        "note": "grouped by source photograph; val/test are non-augmented originals only",
    }
    with open(OUT_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    for split in ("train", "val", "test"):
        n_files = len(rows[split])
        n_src = meta["source_photographs"][split]
        print(f"{split:<6} {n_files:>7,} files   {n_src:>7,} source photographs")
    print(f"\nmanifests written to {OUT_DIR}")


if __name__ == "__main__":
    main()
