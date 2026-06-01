#!/bin/sh
#
# sweep_roofline.sh - Walk the roofline manifest and run each config under
# monitor_run.exe, emitting one CSV row per acc tile per config.
#
# Designed to be staged at /root/roofline/sweep_roofline.sh on target.
# manifest.json must be at /root/roofline/manifest.json (see gen_roofline.py
# host-side staging instructions in the project README).
#
# Usage on target:
#   ./sweep_roofline.sh [single|fourtile]
#
#     single    - run each config on instance 0 only (default; baseline)
#     fourtile  - also run a 4-way output-channel split for each config
#                 (not implemented in this script; intended hook for later
#                 once gen_roofline.py is extended to emit splits)
#
# Output: /tmp/roofline.csv  (with header, one CSV row per acc tile per run)

set -eu

MODE=${1:-single}
ROOF=/root/roofline
MAN=$ROOF/manifest.json
RT=/root/intra_parallel/nvdla_runtime
MON=/examples/monitor_run/monitor_run.exe
OUT=/tmp/roofline.csv

export LD_LIBRARY_PATH=/root/intra_parallel

if [ ! -x "$MON" ]; then
    echo "ERROR: $MON not found. Did you 'make examples; make linux'?"
    exit 1
fi
if [ ! -r "$MAN" ]; then
    echo "ERROR: $MAN not found. Stage manifest.json first."
    exit 1
fi

# CSV header (capture from monitor_run's first emission, then de-dup)
HDR_DONE=0

# Extract every "tag" from manifest.json. We construct the loadable path
# from the tag below, ignoring the manifest's "loadable" field — that field
# holds the *host-side* absolute path used at compile time, not the on-target
# path under $ROOF/. Walking by tag keeps the sweep order (and groups) from
# the manifest while making the script independent of where files were built.
extract_tags() {
    awk '
    /"tag"[[:space:]]*:/ { gsub(/[",]/,""); print $NF }
    ' "$MAN"
}

run_one() {
    label=$1
    cmd=$2

    out=$("$MON" "$label" -- "$cmd" 2>/tmp/${label}.stderr.log)
    if [ $HDR_DONE -eq 0 ]; then
        # busybox in this rootfs has no sed; use cut to strip the "# " prefix.
        echo "$out" | grep '^# ' | head -1 | cut -c3- > "$OUT"
        HDR_DONE=1
    fi
    echo "$out" | grep -v '^# ' >> "$OUT"
}

echo "============================================================"
echo " Roofline sweep (mode=$MODE)"
echo "============================================================"

# Materialize tag/loadable pairs to a tempfile so the while-loop runs in
# the *current* shell (not a pipe subshell — busybox ash can't share state
# back out of a subshell, and we need HDR_DONE to persist).
TMP=/tmp/roofline_tags.$$
extract_tags > "$TMP"

i=0
while read tag; do
    [ -z "$tag" ] && continue

    layer_dir="$ROOF/$tag"
    base="${tag}.nvdla"
    loadable="$layer_dir/$base"

    i=$((i + 1))
    echo ""
    echo "--- [$i] $tag  loadable=$loadable ---"

    if [ ! -r "$loadable" ]; then
        echo "  (skip: $loadable not found)"
        continue
    fi

    cmd="cd $layer_dir && $RT --loadable $base --instance 0"
    run_one "${tag}_t1" "$cmd"

    # Optional: 4-way split block (placeholder). Uncomment + adapt once
    # gen_roofline.py emits split loadables under split_dirs alongside the
    # full one. For now, print a note.
    if [ "$MODE" = "fourtile" ]; then
        if [ -d "${layer_dir}_split0_of4" ]; then
            split_cmd=""
            for k in 0 1 2 3; do
                sd="${layer_dir}_split${k}_of4"
                sb="${tag}_split${k}_of4.nvdla"
                split_cmd="${split_cmd}( cd $sd && $RT --loadable $sb --instance $k ) &"
            done
            run_one "${tag}_t4" "${split_cmd} wait"
        else
            echo "  (no 4-way split loadables for $tag, skipping fourtile mode)"
        fi
    fi
done < "$TMP"

rm -f "$TMP"

echo ""
echo "============================================================"
echo " Sweep complete. Results: $OUT"
echo "============================================================"
wc -l "$OUT"
echo ""
echo "Copy /tmp/roofline.csv off the board and run host-side:"
echo "    python3 plot_roofline.py --csv roofline.csv \\"
echo "                             --manifest manifest.json"
