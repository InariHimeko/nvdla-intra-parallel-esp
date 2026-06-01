#!/bin/sh
#
# verify.sh - Bit-exact correctness check for output-channel split.
#
# For every layer in /root/resnet18/*_full/, runs --rawdump on the baseline
# and the 4 splits (parallel, on instances 0..3), then concatenates the four
# split outputs and compares to the baseline.
#
# Output: PASS / FAIL line per layer, plus a summary count.

set -u

RES=/root/resnet18
RT=/root/intra_parallel/nvdla_runtime
DIMG=/tmp/dimg
N=4

export LD_LIBRARY_PATH=/root/intra_parallel
mkdir -p "$DIMG"

cd "$RES"

PASS=0
FAIL=0
SKIP=0

for full_dir in *_full; do
    layer=${full_dir%_full}
    [ -d "$full_dir" ] || continue

    # check splits exist for N=4
    have_splits=1
    for i in 0 1 2 3; do
        [ -d "${layer}_split${i}_of${N}" ] || { have_splits=0; break; }
    done
    if [ $have_splits -eq 0 ]; then
        echo "$layer: SKIP (no ${N}-way splits)"
        SKIP=$((SKIP+1))
        continue
    fi

    echo ""
    echo "=== $layer ==="

    # baseline rawdump on instance 0
    ( cd "$full_dir" && "$RT" --loadable "${full_dir}.nvdla" --rawdump --instance 0 \
                                                                          >/dev/null 2>&1
      cp output.dimg "$DIMG/${layer}_full.dimg" )

    # 4-way parallel rawdump
    for i in 0 1 2 3; do
        sd="${layer}_split${i}_of${N}"
        ( cd "$sd" && "$RT" --loadable "${sd}.nvdla" --rawdump --instance $i \
                                                                  >/dev/null 2>&1
          cp output.dimg "$DIMG/${layer}_split${i}_of${N}.dimg" ) &
    done
    wait

    # Concat splits and compare to baseline. rawdump is text (whitespace-
    # separated ints in CHW order); whitespace-normalize both sides before
    # diff so trailing-newline differences don't trip us.
    cat "$DIMG/${layer}_split0_of${N}.dimg" \
        "$DIMG/${layer}_split1_of${N}.dimg" \
        "$DIMG/${layer}_split2_of${N}.dimg" \
        "$DIMG/${layer}_split3_of${N}.dimg" \
        | tr -s '[:space:]' '\n' | grep -v '^$' > "$DIMG/${layer}_concat.txt"

    tr -s '[:space:]' '\n' < "$DIMG/${layer}_full.dimg" \
        | grep -v '^$' > "$DIMG/${layer}_full.txt"

    nf=$(wc -l < "$DIMG/${layer}_full.txt")
    nc=$(wc -l < "$DIMG/${layer}_concat.txt")

    if [ "$nf" != "$nc" ]; then
        echo "$layer: FAIL (length mismatch full=$nf concat=$nc)"
        FAIL=$((FAIL+1))
        continue
    fi

    # busybox in this rootfs has no cmp/diff — use md5sum (which it has).
    # The .txt files are both whitespace-normalized so byte-exact md5
    # equality is correct here.
    sum_full=$(md5sum "$DIMG/${layer}_full.txt"   | awk '{print $1}')
    sum_cat=$(md5sum "$DIMG/${layer}_concat.txt" | awk '{print $1}')

    if [ "$sum_full" = "$sum_cat" ]; then
        echo "$layer: PASS ($nf values match, md5=$sum_full)"
        PASS=$((PASS+1))
    else
        # awk-based mismatch count + first few mismatches (busybox awk OK)
        nmis=$(awk 'NR==FNR{a[NR]=$0;next} {if($0!=a[FNR]) n++} END{print n+0}' \
                  "$DIMG/${layer}_full.txt" "$DIMG/${layer}_concat.txt")
        echo "$layer: FAIL ($nmis line(s) differ out of $nf)"
        echo "  md5 full=$sum_full  concat=$sum_cat"
        awk 'NR==FNR{a[NR]=$0;next}
             {if($0!=a[FNR]){print "  ["FNR"] full="a[FNR]" concat="$0; c++; if(c>=10)exit}}' \
            "$DIMG/${layer}_full.txt" "$DIMG/${layer}_concat.txt"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "============================================"
echo " PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "============================================"
[ $FAIL -eq 0 ]
