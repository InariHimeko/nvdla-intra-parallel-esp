#!/usr/bin/env python3
"""
plot_roofline.py - Host-side analysis of the roofline sweep.

Reads:
    manifest.json       analytical metrics per config (from gen_roofline.py)
    roofline.csv        per-acc-tile counters per run (from monitor_run.exe
                        + sweep_roofline.sh on target)

For each config, computes:
    achieved_ops_per_cycle    = manifest.ops_int8 / acc_tot_cycles
    achieved_dram_BW_B_per_cy = ddr_words * BYTES_PER_WORD / acc_tot_cycles
    achieved_noc_BW_B_per_cy  = (sum p0+p3 NoC packets) * BYTES_PER_PKT
                                  / acc_tot_cycles
    mem_bound_pct             = 100 * acc_mem / acc_tot
    llc_hit_rate              = llc_hits / (llc_hits + llc_misses)

And labels each kernel as one of:
    compute-bound  - ops/cycle near the hardware peak
    dram-bound     - bytes/cycle saturates the DRAM ceiling
    noc-bound      - NoC inject rate saturates while DRAM has slack
    coherence-bound- coh_reqs >> ddr_words and llc_hit_rate high
                     (i.e., serialized at the LLC port, not DRAM)

Plots (matplotlib, optional):
    1. Roofline (DRAM): ops/cycle vs ops/DRAM-byte, with peak compute and
       peak DRAM BW ceilings overlaid
    2. Roofline (NoC):  ops/cycle vs ops/NoC-byte
    3. Achieved ops/cycle vs arithmetic intensity, colored by group

Also prints a sortable text table to stdout for the report.

Usage:
    python3 plot_roofline.py --csv roofline.csv --manifest manifest.json \\
                             [--clock-mhz 78] [--mac-per-cycle 64] \\
                             [--bytes-per-word 8] [--bytes-per-noc-pkt 8] \\
                             [--out-dir plots]

The peak-compute and peak-DRAM-BW ceilings are estimated from the data points
(highest measured ops/cycle and bytes/cycle respectively); if you have known
hardware values (e.g. nv_small INT8 = 64 MAC/cycle = 128 ops/cycle), pass
them via --mac-per-cycle.
"""

import argparse
import csv
import json
import os
import sys

# matplotlib is optional — script still prints the table and CSV without it.
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    by_tag = {entry["tag"]: entry for entry in m}
    return by_tag


def load_csv(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # cast every numeric column
            for k, v in r.items():
                if k == "label":
                    continue
                try:
                    r[k] = int(v)
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def aggregate_by_label(rows):
    """One run produces SOC_NACC rows (one per acc tile). Pick the row with
    max acc_tot_cycles (the "active" tile in single-tile mode; the slowest
    tile in N-way mode). For single-tile this is the only non-zero row."""
    by_label = {}
    for r in rows:
        lab = r["label"]
        cur = by_label.get(lab)
        if cur is None or r["acc_tot_cycles"] > cur["acc_tot_cycles"]:
            by_label[lab] = r
    return by_label


def derive(row, manifest_entry, args):
    cyc = row["acc_tot_cycles"] or 1
    ops = manifest_entry["metrics"]["ops_int8"]
    ai_dense = manifest_entry["metrics"]["ai_ops_per_byte"]

    ddr_bytes = row["ddr_words"] * args.bytes_per_word
    noc_bytes_dma = (row["noc_inject_p3"] + row["noc_inject_p4"]) \
                      * args.bytes_per_noc_pkt
    noc_bytes_coh = (row["noc_inject_p0"] + row["noc_inject_p1"]
                     + row["noc_inject_p2"]) * args.bytes_per_noc_pkt
    noc_bytes_total = noc_bytes_dma + noc_bytes_coh

    achieved_ops_per_cy   = ops / cyc
    achieved_dram_bw_bpc  = ddr_bytes / cyc
    achieved_noc_bw_bpc   = noc_bytes_total / cyc

    ai_dram = ops / ddr_bytes if ddr_bytes else 0.0
    ai_noc  = ops / noc_bytes_total if noc_bytes_total else 0.0

    mem_bound_pct = (100.0 * row["acc_mem_cycles"] / cyc) if cyc else 0.0

    llc_total = row["llc_hits"] + row["llc_misses"]
    llc_hit_rate = (100.0 * row["llc_hits"] / llc_total) if llc_total else 0.0

    return {
        "tag":                  manifest_entry["tag"],
        "group":                manifest_entry["group"],
        "ai_dense":             ai_dense,
        "ai_dram":              ai_dram,
        "ai_noc":               ai_noc,
        "acc_tot_cycles":       cyc,
        "acc_mem_cycles":       row["acc_mem_cycles"],
        "ops":                  ops,
        "ops_per_cycle":        achieved_ops_per_cy,
        "dram_bw_B_per_cycle":  achieved_dram_bw_bpc,
        "noc_bw_B_per_cycle":   achieved_noc_bw_bpc,
        "ddr_words":            row["ddr_words"],
        "noc_p0":               row["noc_inject_p0"],
        "noc_p3":               row["noc_inject_p3"],
        "noc_p4":               row["noc_inject_p4"],
        "mem_bound_pct":        mem_bound_pct,
        "llc_hit_rate_pct":     llc_hit_rate,
        "mem_coh_reqs":         row["mem_coh_reqs"],
    }


# ---------------------------------------------------------------------------
# Classification heuristic
# ---------------------------------------------------------------------------
def classify(d, peak_ops_per_cy, peak_dram_bpc, peak_noc_bpc):
    """Crude classification: which ceiling is each kernel close to?
    Skip a ceiling whose peak is essentially zero (counter not firing)."""
    near = lambda x, ceil: ceil > 1.0 and x >= 0.7 * ceil
    if near(d["ops_per_cycle"], peak_ops_per_cy):
        return "compute"
    if near(d["dram_bw_B_per_cycle"], peak_dram_bpc):
        return "dram"
    if near(d["noc_bw_B_per_cycle"], peak_noc_bpc):
        return "noc"
    if d["mem_bound_pct"] > 95 and d["llc_hit_rate_pct"] > 50:
        return "coherence"
    # 100% memory-stalled but DRAM utilization low — latency-bound, not BW-bound
    if d["mem_bound_pct"] > 95 and d["dram_bw_B_per_cycle"] < 0.3 * peak_dram_bpc:
        return "latency-stall"
    return "memory-stall"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roofline(derived, peak_ops_per_cy, peak_bw_bpc,
                  ai_key, label, out_path):
    if not HAS_PLT:
        print(f"  (matplotlib not available; skipping {out_path})")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # data points colored by group
    groups = sorted({d["group"] for d in derived})
    for g in groups:
        xs = [d[ai_key]          for d in derived if d["group"] == g and d[ai_key] > 0]
        ys = [d["ops_per_cycle"] for d in derived if d["group"] == g and d[ai_key] > 0]
        ax.scatter(xs, ys, label=g, s=40)

    # ceilings
    if derived:
        ai_min = max(min(d[ai_key] for d in derived if d[ai_key] > 0) * 0.5, 1e-3)
        ai_max = max(d[ai_key] for d in derived if d[ai_key] > 0) * 2
        xs_line = [ai_min, peak_ops_per_cy / peak_bw_bpc, ai_max]
        xs_line.sort()
        # memory ceiling: ops/cy = peak_bw_bpc * (ops/byte)
        mem_ys = [peak_bw_bpc * x for x in xs_line]
        ax.plot(xs_line, mem_ys, "k--",
                label=f"BW peak = {peak_bw_bpc:.2f} B/cy")
        ax.axhline(peak_ops_per_cy, color="k", linestyle=":",
                   label=f"compute peak = {peak_ops_per_cy:.0f} ops/cy")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"arithmetic intensity ({label}, ops/byte)")
    ax.set_ylabel("achieved ops / cycle")
    ax.set_title(f"Roofline: {label}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def print_table(derived, classification):
    fmt = ("{:<22} {:<11} {:>9} {:>9} {:>9} {:>11} {:>11} {:>11} "
           "{:>9} {:>9} {:<14}")
    print(fmt.format("tag", "group",
                     "AI_dense", "AI_dram", "AI_noc",
                     "ops/cy", "dramB/cy", "nocB/cy",
                     "mem%", "llc_hit%", "class"))
    print("-" * 150)
    for d in derived:
        cls = classification.get(d["tag"], "?")
        print(fmt.format(d["tag"][:22], d["group"][:11],
                         f'{d["ai_dense"]:.2f}',
                         f'{d["ai_dram"]:.2f}',
                         f'{d["ai_noc"]:.2f}',
                         f'{d["ops_per_cycle"]:.2f}',
                         f'{d["dram_bw_B_per_cycle"]:.2f}',
                         f'{d["noc_bw_B_per_cycle"]:.2f}',
                         f'{d["mem_bound_pct"]:.1f}',
                         f'{d["llc_hit_rate_pct"]:.1f}',
                         cls))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir",  default="roofline_plots")
    ap.add_argument("--bytes-per-word",     type=float, default=8,
                    help="bytes per ddr_words counter increment")
    ap.add_argument("--bytes-per-noc-pkt",  type=float, default=8,
                    help="bytes per NoC packet (depends on NoC width)")
    ap.add_argument("--mac-per-cycle",      type=float, default=None,
                    help="HW peak MACs/cycle (nv_small INT8 typically 64); "
                         "if omitted, peaks are inferred from data")
    ap.add_argument("--clock-mhz",          type=float, default=78,
                    help="reported only for context, doesn't affect roofline")
    args = ap.parse_args()

    by_tag = load_manifest(args.manifest)
    rows = load_csv(args.csv)
    if not rows:
        print(f"no rows in {args.csv}", file=sys.stderr); return 1

    by_label = aggregate_by_label(rows)

    # Build derived metrics for each label that matches a manifest tag
    derived = []
    skipped = []
    for label, row in by_label.items():
        # the on-target script tags rows as "<tag>_t1" or "<tag>_t4"; strip
        for suffix in ("_t1", "_t4"):
            if label.endswith(suffix):
                tag = label[:-len(suffix)]
                break
        else:
            tag = label
        if tag not in by_tag:
            skipped.append(label)
            continue
        derived.append(derive(row, by_tag[tag], args))

    if skipped:
        print(f"warning: {len(skipped)} CSV labels did not match manifest tags: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}",
              file=sys.stderr)
    if not derived:
        print("no derived rows", file=sys.stderr); return 1

    # Peak ceilings
    peak_ops_per_cy = (2 * args.mac_per_cycle) if args.mac_per_cycle \
                      else max(d["ops_per_cycle"] for d in derived)
    peak_dram_bpc  = max(d["dram_bw_B_per_cycle"] for d in derived) or 1.0
    peak_noc_bpc   = max(d["noc_bw_B_per_cycle"]  for d in derived) or 1.0

    classification = {d["tag"]: classify(d, peak_ops_per_cy,
                                         peak_dram_bpc, peak_noc_bpc)
                      for d in derived}

    print(f"\nclock = {args.clock_mhz} MHz  (informational)")
    print(f"peak compute  = {peak_ops_per_cy:.1f} ops/cycle "
          f"({'fixed' if args.mac_per_cycle else 'inferred'})")
    print(f"peak DRAM BW  = {peak_dram_bpc:.2f} bytes/cycle (inferred from data)")
    print(f"peak NoC BW   = {peak_noc_bpc:.2f}  bytes/cycle (inferred from data)\n")

    derived.sort(key=lambda d: d["ai_dense"])
    print_table(derived, classification)

    # Save derived as CSV for spreadsheet/report use
    os.makedirs(args.out_dir, exist_ok=True)
    derived_csv = os.path.join(args.out_dir, "derived.csv")
    with open(derived_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(derived[0].keys()) + ["class"])
        w.writeheader()
        for d in derived:
            row = dict(d); row["class"] = classification[d["tag"]]
            w.writerow(row)
    print(f"\n-> {derived_csv}")

    # Plots
    plot_roofline(derived, peak_ops_per_cy, peak_dram_bpc,
                  "ai_dram", "DRAM",
                  os.path.join(args.out_dir, "roofline_dram.png"))
    plot_roofline(derived, peak_ops_per_cy, peak_noc_bpc,
                  "ai_noc",  "NoC",
                  os.path.join(args.out_dir, "roofline_noc.png"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
