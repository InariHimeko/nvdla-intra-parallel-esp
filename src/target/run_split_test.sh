#!/bin/sh
#
# run_split_test.sh - Run full and split conv1 loadables on FPGA NVDLA instances.
#
# Based on ESP NVDLA guide:
#   https://www.esp.cs.columbia.edu/docs/thirdparty_acc/thirdparty_acc-guide/
#
# Assumes all .nvdla files, seven.pgm, nvdla_runtime, and libnvdla_runtime.so
# are in the current directory.
#
# Usage:  ./run_split_test.sh [NUM_SPLITS]
#         NUM_SPLITS defaults to 2

NUM_SPLITS=${1:-2}
IMAGE="seven.pgm"

echo "============================================"
echo " Intra-Layer Parallelism - Conv1 Split Test"
echo " Splits: $NUM_SPLITS"
echo "============================================"

# Suppress kernel printk noise on the console
dmesg -n 1 2>/dev/null

# -----------------------------------------------------------
# 1. Baseline: run full conv1 on instance 0 with rawdump
# -----------------------------------------------------------
echo ""
echo "[1/3] Running full conv1 on instance 0 (rawdump)..."
./nvdla_runtime --loadable conv1_full.nvdla --image $IMAGE --rawdump --instance 0
mv output.dimg output_full.dimg
echo "  -> output_full.dimg"

# -----------------------------------------------------------
# 2. Sequential splits: run each on its own instance with rawdump
# -----------------------------------------------------------
echo ""
echo "[2/3] Running splits sequentially (rawdump)..."
i=0
while [ $i -lt $NUM_SPLITS ]; do
    echo "  Split $i on instance $i..."
    ./nvdla_runtime --loadable "conv1_split_${i}.nvdla" --image $IMAGE --rawdump --instance $i
    mv output.dimg "output_split_${i}.dimg"
    echo "  -> output_split_${i}.dimg"
    i=$((i + 1))
done

# -----------------------------------------------------------
# 3. Parallel splits: run all simultaneously with rawdump
#    Each split runs in its own subdirectory to avoid
#    output.dimg file collision.
# -----------------------------------------------------------
echo ""
echo "[3/3] Running splits in parallel (rawdump)..."

# Set up per-split working directories
i=0
while [ $i -lt $NUM_SPLITS ]; do
    mkdir -p "par_${i}"
    cp "conv1_split_${i}.nvdla" "par_${i}/"
    cp $IMAGE "par_${i}/"
    i=$((i + 1))
done

# Launch all splits in parallel, per ESP guide pattern:
#   ./nvdla_runtime --loadable ... --instance 0 &
#   ./nvdla_runtime --loadable ... --instance 1 &
i=0
while [ $i -lt $NUM_SPLITS ]; do
    (
        cd "par_${i}"
        LD_LIBRARY_PATH=..:$LD_LIBRARY_PATH ../nvdla_runtime --loadable "conv1_split_${i}.nvdla" --image $IMAGE --rawdump --instance $i
    ) &
    i=$((i + 1))
done
wait

# Collect parallel outputs and clean up
i=0
while [ $i -lt $NUM_SPLITS ]; do
    mv "par_${i}/output.dimg" "output_par_split_${i}.dimg"
    rm -rf "par_${i}"
    echo "  -> output_par_split_${i}.dimg"
    i=$((i + 1))
done

echo ""
echo "============================================"
echo " Results"
echo "============================================"
echo "Baseline output:     output_full.dimg"
i=0
while [ $i -lt $NUM_SPLITS ]; do
    echo "Sequential split $i:  output_split_${i}.dimg"
    echo "Parallel   split $i:  output_par_split_${i}.dimg"
    i=$((i + 1))
done
echo ""
echo "Validate:"
echo "  python3 validate.py --num-splits $NUM_SPLITS"
echo "============================================"
