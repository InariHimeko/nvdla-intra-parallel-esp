# Integrating `monitor_run` into the ESP source tree

`monitor_run` is a small host-launched harness that wraps an arbitrary
subprocess (here, `nvdla_runtime`), snapshots the ESP per-tile hardware
monitor counters before and after the run via `libmonitors`, and emits a
23-column CSV per accelerator tile. It is the instrument behind every
hardware-counter number in this project.

## Where it goes

It is a generic ESP "example" application, so it lives alongside the other
examples in the ESP source tree:

```
<ESP>/soft/common/apps/examples/monitor_run/
    monitor_run.c
    Makefile
```

Copy both files there, then build from the SoC directory:

```bash
cp monitor_run.c Makefile <ESP>/soft/common/apps/examples/monitor_run/
cd <ESP>/socs/xilinx-vcu118-xcvu9p
make examples      # discovers the new example, builds monitor_run.exe
make linux         # bakes it into the FPGA root filesystem
```

After `make linux`, the binary is at `/examples/monitor_run/monitor_run.exe`
on the target.

## The static-link requirement (important)

The `Makefile` adds `LDFLAGS += -static`. This is **required**, not optional.
The `/opt/riscv` cross-toolchain used by `make examples` is built against a
glibc with a `GLIBC_2.26` symbol baseline, but the target root filesystem
ships glibc 2.27. A dynamically-linked binary fails to load on the target
with:

```
version `GLIBC_2.26' not found (required by monitor_run.exe)
```

Static linking removes the runtime libc dependency entirely and sidesteps the
mismatch. The resulting binary is ~1 MB, which is fine for a benchmark helper.

## Usage

```
monitor_run.exe <label> -- <command> [args...]
monitor_run.exe <label> -- "<sh -c command string>"
```

The single-token form runs the argument through `sh -c`, which lets you launch
several `nvdla_runtime` processes in parallel with `&` + `wait` and capture
them all in one monitor window (used for the multi-tile contention test).

The CSV columns are: `label, acc_index, acc_invocations, acc_tot_cycles,
acc_mem_cycles, acc_tlb_cycles, noc_inject_p0..p5, ddr_words, mem_coh_reqs,
mem_coh_fwds, mem_coh_rsps_rcv, mem_coh_rsps_snd, mem_dma_reqs, mem_dma_rsps,
mem_coh_dma_reqs, mem_coh_dma_rsps, llc_hits, llc_misses`.

A full per-run dump (every monitor field) is also written to
`/tmp/<label>.mon.txt`.

See `docs/audit_2026-05-07.txt` for the precise semantics of each counter and
which ones are trustworthy for NVDLA tiles.
