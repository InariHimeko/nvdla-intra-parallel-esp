#!/usr/bin/env python3
"""
validate.py - Compare split conv1 outputs against the full baseline.

Reads the rawdump .dimg files produced by nvdla_runtime --rawdump.
The rawdump format is space-separated values in CHW order (one value per
output element: channel × height × width).

For output-channel splitting, correctness means:
    concat(split_0, split_1, ..., split_{N-1}) along channel dim == full

Usage:
    python3 validate.py [--num-splits N] [--output-dir DIR]
"""

import os
import sys
import argparse

import numpy as np


def read_rawdump(path):
    """Read a rawdump .dimg file into a flat numpy array of ints."""
    with open(path) as f:
        text = f.read()
    values = [int(x) for x in text.split()]
    return np.array(values, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(
        description="Validate split conv1 outputs against full baseline")
    ap.add_argument("--num-splits", type=int, default=2,
                    help="Number of splits (default: 2)")
    ap.add_argument("--output-dir", type=str, default=".",
                    help="Directory containing output_*.dimg files")
    ap.add_argument("--mode", choices=["sequential", "parallel", "both"],
                    default="both",
                    help="Which split outputs to validate (default: both)")
    args = ap.parse_args()

    d = args.output_dir

    # ---- Read baseline ----
    full_path = os.path.join(d, "output_full.dimg")
    if not os.path.exists(full_path):
        print(f"ERROR: {full_path} not found"); sys.exit(1)
    full = read_rawdump(full_path)
    print(f"Full baseline: {len(full)} values")

    # ---- Validate sequential and/or parallel splits ----
    modes = []
    if args.mode in ("sequential", "both"):
        modes.append(("sequential", "output_split_{}.dimg"))
    if args.mode in ("parallel", "both"):
        modes.append(("parallel", "output_par_split_{}.dimg"))

    for label, pattern in modes:
        print(f"\n--- Validating {label} splits ---")
        parts = []
        for i in range(args.num_splits):
            path = os.path.join(d, pattern.format(i))
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping {label}")
                break
            part = read_rawdump(path)
            parts.append(part)
            print(f"  Split {i}: {len(part)} values")
        else:
            # All split files found
            merged = np.concatenate(parts)
            print(f"  Merged:  {len(merged)} values")

            if len(merged) != len(full):
                print(f"  FAIL: length mismatch "
                      f"(merged={len(merged)}, full={len(full)})")
                continue

            diff = merged - full
            num_mismatch = np.count_nonzero(diff)
            max_abs_diff = np.max(np.abs(diff)) if num_mismatch > 0 else 0

            if num_mismatch == 0:
                print(f"  PASS: exact match")
            else:
                pct = 100.0 * num_mismatch / len(full)
                print(f"  MISMATCH: {num_mismatch}/{len(full)} values "
                      f"differ ({pct:.2f}%), max |diff| = {max_abs_diff}")
                # Show first few mismatches for debugging
                idxs = np.where(diff != 0)[0][:10]
                for idx in idxs:
                    print(f"    [{idx}] full={full[idx]}  split={merged[idx]}  "
                          f"diff={diff[idx]}")

    print()


if __name__ == "__main__":
    main()
