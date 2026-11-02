import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_ARMS = ["A0", "A1", "A2", "A3", "A4", "B0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(a, s) for s in args.seeds for a in args.arms]
    print(f"{len(jobs)} run(s) queued: {args.arms} x seeds {args.seeds}\n")

    started = time.perf_counter()
    for i, (arm, seed) in enumerate(jobs, 1):
        out = RESULTS_DIR / f"{arm}_seed{seed}" / "results.json"
        if out.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] {arm} seed {seed}: already complete, skipping")
            continue

        print(f"[{i}/{len(jobs)}] {arm} seed {seed}: starting", flush=True)
        t0 = time.perf_counter()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts" / "train_ablation.py"),
            "--arm", arm, "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--img-size", str(args.img_size),
            "--batch-size", str(args.batch_size),
        ], env=env)
        mins = (time.perf_counter() - t0) / 60
        status = "done" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"[{i}/{len(jobs)}] {arm} seed {seed}: {status} in {mins:.1f} min\n", flush=True)

    print(f"sweep finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
