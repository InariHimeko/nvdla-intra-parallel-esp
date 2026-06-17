# Intra-Layer Parallelism on Tiled NVDLA Accelerators in ESP

**Author:** Wenxuan Xu
**Course:** CSEE E6868 — Embedded Scalable Platforms, Columbia University, Spring 2026
**Platform:** Xilinx VCU118 FPGA · ESP SoC · 4× NVDLA `nv_small_fpga` tiles

This repository is a self-contained snapshot of a semester project that
implements and characterizes **coarse-grained intra-layer parallelism** —
splitting a single convolution layer across multiple NVDLA accelerator tiles
inside the [ESP](https://www.esp.cs.columbia.edu/) platform.

> **All performance data in this repository is measured on FPGA hardware, not
> simulation.**

---

## TL;DR — what we found

1. **The split is correct.** A compile-time output-channel splitting workflow
   produces bit-exact results for all **11 distinct convolution shapes** in
   ResNet-18 (`verify.sh`: 11/11 PASS).
2. **But 4-way parallelism is _slower_, not faster** — about **0.30× per-tile
   speedup** (i.e. each tile takes ~3.2× longer), uniformly across layer types.
3. **Root cause is the memory subsystem, not compute or the partitioning
   strategy.** A 27-configuration roofline sweep and a 4-tile contention test
   localize the bottleneck to two stacked mechanisms:
   - NVDLA `nv_small_fpga` issues **32-byte AXI bursts** vs. the NoC's
     **2048-byte** capacity;
   - ESP serializes outstanding transactions (**one in flight per master**),
     so per-AXI-transaction round-trip latency dominates and does not pipeline.
   With a single shared memory tile, four tiles contend for one serialization
   slot, so per-tile time grows ~linearly with tile count.

The consolidated results are in
[`docs/results_summary_2026-05-06.txt`](docs/results_summary_2026-05-06.txt).
The honest limitations and what we could *not* measure are in
[`docs/audit_2026-05-07.txt`](docs/audit_2026-05-07.txt).

---

## Repository layout

```
nvdla-intra-parallel/
├── README.md                     ← you are here
├── src/
│   ├── host/                     Host-side Python (run on the build machine)
│   │   ├── gen_resnet18.py         Generate baseline + N-way split loadables
│   │   │                           for every distinct conv shape in ResNet-18
│   │   ├── gen_roofline.py         Generate the 27-config roofline sweep
│   │   ├── split_conv1.py          Original midterm LeNet-conv1 splitter
│   │   ├── plot_roofline.py        Roofline analysis + plots from a CSV
│   │   ├── validate.py             Host-side output comparison
│   │   └── ditcaffe_pb2.py         Generated Caffe protobuf (dependency)
│   ├── target/                   On-FPGA shell scripts (run under ESP Linux)
│   │   ├── sweep_roofline.sh       Walk the manifest, run each config under
│   │   │                           monitor_run, emit CSV
│   │   ├── verify.sh               Bit-exact correctness check (md5sum-based)
│   │   └── run_split_test.sh       Original conv1 split test
│   └── monitor_run/              HW-counter harness (built into the ESP tree)
│       ├── monitor_run.c           Wraps nvdla_runtime, snapshots ESP monitors
│       ├── Makefile
│       └── INTEGRATION.md          Where this goes in the ESP source tree
├── data/
│   ├── raw/                       Raw FPGA serial captures (source of truth)
│   │   ├── minicom.cap              ResNet-18 4-data-point timing study
│   │   ├── minicom-May6.cap         verify.sh correctness run (PASS=11)
│   │   ├── minicom-May6-2.cap       roofline sweep, first (failed) attempt
│   │   ├── minicom-May6-3.cap       roofline sweep, 27 configs (the data)
│   │   └── minicom-May6-4.cap       4-tile contention test + re-run
│   ├── roofline_plots/
│   │   ├── derived.csv              Per-config derived metrics (27 rows)
│   │   └── roofline_dram.png        Roofline plot
│   └── manifests/                 Loadable manifests (the bulk .nvdla files
│       │                          are NOT included — they are regenerable)
│       ├── resnet18_manifest.json
│       └── roofline_manifest.json
└── docs/
    ├── measurement_summary.txt     ResNet-18 4-data-point study, raw numbers
    ├── results_summary_2026-05-06.txt  Consolidated results
    ├── audit_2026-05-07.txt         Corrected conclusions + counter semantics
    │                                (READ THIS for what each counter means and
    │                                 the honest limitations)
    └── meeting_transcription.txt    TA meeting on DRAM-bandwidth interpretation
```


### What is intentionally excluded

The generated `.nvdla` loadables (~95 MB total: `resnet18_out/` and
`roofline_out/`, plus their `wisdom.dir` compiler caches) are **not** in this
repo because they are fully regenerable from the scripts in `src/host/`. The
`data/manifests/*.json` files record exactly which shapes were generated and
their analytical metrics. See the reproduction steps below to rebuild them.

---

## Provenance

- **ESP baseline commit:** `a45f2bb8406350883c169505661bedf5b469e1a3`
  (2026-03-24). All scripts assume an ESP tree at this revision with the
  VCU118 / 4×NVDLA SoC configuration described below.
- **SoC configuration:** `xilinx-vcu118-xcvu9p`, 3×3 tile grid,
  `SOC_NACC=4`, `SOC_NMEM=1`, `BASE_FREQ_MHZ=78`.
- **NVDLA configuration:** `nv_small_fpga` — 8×8 INT8 MAC array,
  `PRIMARY_MEMIF_WIDTH=64`, `PRIMARY_MEMIF_MAX_BURST_LENGTH=4`
  (⇒ 32 B per AXI transaction).
- **Only one file is added inside the ESP source tree:** `monitor_run.c`
  (see `src/monitor_run/INTEGRATION.md`). Everything else in `src/` is
  standalone and lives outside the ESP tree.

---

## Reproduction manual

### Prerequisites
- An ESP checkout at the commit above, configured for the
  `xilinx-vcu118-xcvu9p` SoC with 4 NVDLA tiles.
- NVDLA prebuilt compiler/runtime (ships with ESP's NVDLA fork under
  `accelerators/third-party/NV_NVDLA/sw/prebuilt/`).
- Host Python 3 with `numpy`, `protobuf`, and (for plots) `matplotlib`.
- A VCU118 board, Vivado for `make fpga-program`, and `minicom` for the
  serial console.

### Step 1 — Build the HW-counter harness (one-time ESP-tree edit)
```bash
# Copy the harness into the ESP examples directory
cp src/monitor_run/monitor_run.c src/monitor_run/Makefile \
   <ESP>/soft/common/apps/examples/monitor_run/
cd <ESP>/socs/xilinx-vcu118-xcvu9p
make examples          # builds monitor_run.exe into the sysroot
```
See `src/monitor_run/INTEGRATION.md` for the static-link note (the harness
must be statically linked to avoid a glibc SYMVER mismatch on the target).

### Step 2 — Generate split loadables (host side)
```bash
cd <ESP>/socs/xilinx-vcu118-xcvu9p/intra_parallel   # gen scripts expect to run here
# ResNet-18: baseline + 4-way output-channel splits for all 11 conv shapes
python3 gen_resnet18.py --num-splits 4
# Roofline sweep: 27 single-conv configs across arithmetic intensities 49–664
python3 gen_roofline.py
```
This regenerates the `resnet18_out/` and `roofline_out/` trees referenced by
`data/manifests/*.json`.

### Step 3 — Stage to the FPGA root filesystem and rebuild Linux
```bash
# Stage the loadables + target scripts into the buildroot sysroot
mkdir -p soft-build/ariane/sysroot/root/roofline
cp -r roofline_out/*               soft-build/ariane/sysroot/root/roofline/
cp <repo>/src/target/sweep_roofline.sh soft-build/ariane/sysroot/root/roofline/
# (similarly for resnet18_out + verify.sh under .../root/resnet18/)
make linux
make fpga-program        # bitstream load; needs Vivado + the board
make fpga-run-linux
# in a second terminal:
make uart
```
> **Note on rootfs size:** the initramfs is RAM-limited. Stage *either*
> `resnet18/` *or* `roofline/`, not both at once, or the kernel will panic in
> `populate_rootfs`. (We hit this — see the captures.)

### Step 4 — Run on the board
```bash
# Correctness (per-layer bit-exact check)
cd /root/resnet18 && ./verify.sh                 # expect: PASS=11 FAIL=0

# Roofline sweep (single tile, 27 configs)
cd /root/roofline && ./sweep_roofline.sh single  # writes /tmp/roofline.csv
cat /tmp/roofline.csv                            # capture via minicom
```
The `minicom*.cap` files in `data/raw/` are exactly these serial sessions.

### Step 5 — Re-derive the analysis (host side)
```bash
# Extract the CSV rows from a minicom capture (23-column rows only)
awk -F, 'NF==23' data/raw/minicom-May6-3.cap > roofline.csv
python3 src/host/plot_roofline.py \
    --csv roofline.csv \
    --manifest data/manifests/roofline_manifest.json \
    --mac-per-cycle 64 --bytes-per-word 8
# -> roofline_plots/derived.csv + roofline plot
```

---

## Key results at a glance

| experiment | result | source |
|---|---|---|
| Correctness, 11 ResNet-18 conv shapes | 11/11 bit-exact PASS | `minicom-May6.cap` |
| conv1 4-way per-tile cycles | 22.95 M → 75.41 M (3.29× slower) | `minicom.cap` |
| layer4_3x3 4-way per-tile cycles | 9.70 M → 31.09 M (3.20× slower) | `minicom.cap` |
| Roofline, best achieved throughput | 26.3 ops/cy = 20.6% of compute peak | `derived.csv` |
| 4-indep contention, per-tile slowdown | 3.5–3.9× uniformly | `minicom-May6-4.cap` |
| 4-indep aggregate DRAM-write scaling | 1.04–1.14× (vs ideal 4×) | `minicom-May6-4.cap` |

---

## Caveats (see `docs/audit_2026-05-07.txt` for the full discussion)

- **`ddr_words` counts writes only** (AXI W-channel beats); read traffic is
  invisible to this counter.
- **Per-NVDLA-tile NoC inject counters do not fire** for NVDLA bulk data in
  this bitstream; the invalid "NoC roofline" plot has been removed.
- **The 65-cycle round-trip used in the per-tile model is an estimate**, not a
  direct measurement on this bitstream (TA reference value; the instructor
  suggested 20–30 cycles may be closer). The qualitative conclusions are
  invariant to the exact value.
- FPGA access was shared and intermittent; the 2-tile/3-tile contention sweep
  and NVDLA-internal stall-register reads were not completed. These are the
  top items of future work.
```
