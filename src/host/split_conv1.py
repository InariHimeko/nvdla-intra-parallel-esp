#!/usr/bin/env python3
"""
split_conv1.py - Generate full and split NVDLA loadables for LeNet conv1.

Implements output-channel splitting for intra-layer parallelism research.
Extracts conv1 weights from the trained LeNet caffemodel, creates single-layer
Caffe models (full and N-way split), and compiles each to an NVDLA loadable
targeting opendla-small (INT8).

Usage:
    python3 split_conv1.py [--num-splits N] [--output-dir DIR]

Requires: numpy, protobuf (pip3 install --user numpy protobuf)
"""

import sys
import os
import json
import struct
import subprocess
import argparse

import numpy as np

# Add local dir for generated protobuf module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ditcaffe_pb2

# ---------------------------------------------------------------------------
# Paths (adjust if your ESP tree is rooted elsewhere)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))       # intra_parallel/
_soc  = os.path.dirname(_here)                           # xilinx-vcu118-xcvu9p/
ESP_ROOT = os.environ.get("ESP_ROOT",
    os.path.dirname(os.path.dirname(_soc)))               # esp/
NVDLA_ROOT = os.path.join(ESP_ROOT, "accelerators/third-party/NV_NVDLA")
PREBUILT   = os.path.join(NVDLA_ROOT, "sw/prebuilt/x86-ubuntu")
COMPILER   = os.path.join(PREBUILT, "nvdla_compiler")
CAFFEMODEL = os.path.join(PREBUILT, "lenet_mnist.caffemodel")
CALIB_JSON = os.path.join(PREBUILT, "lenet_mnist.json")

# Hardware target for ESP's NVDLA instances
HW_TARGET  = "nv_small"
PRECISION  = "int8"


def parse_caffemodel(path):
    """Parse a Caffe .caffemodel file, return NetParameter protobuf."""
    net = ditcaffe_pb2.NetParameter()
    with open(path, "rb") as f:
        net.ParseFromString(f.read())
    return net


def extract_conv1(net):
    """Return (weights, biases) numpy arrays for the conv1 layer."""
    for layer in net.layer:
        if layer.name == "conv1" and layer.type == "Convolution":
            assert len(layer.blobs) == 2, "conv1 must have weights + bias blobs"
            w = layer.blobs[0]
            b = layer.blobs[1]
            weights = np.array(w.data, dtype=np.float32).reshape(list(w.shape.dim))
            biases  = np.array(b.data, dtype=np.float32).reshape(list(b.shape.dim))
            return weights, biases
    raise ValueError("conv1 Convolution layer not found in caffemodel")


def make_prototxt(num_output, in_channels=1, in_h=28, in_w=28, kernel=5):
    """Single-layer conv1 deploy prototxt (text format)."""
    return (
        f'input: "data"\n'
        f'input_shape {{\n'
        f'  dim: 1\n'
        f'  dim: {in_channels}\n'
        f'  dim: {in_h}\n'
        f'  dim: {in_w}\n'
        f'}}\n'
        f'layer {{\n'
        f'  name: "conv1"\n'
        f'  type: "Convolution"\n'
        f'  bottom: "data"\n'
        f'  top: "conv1"\n'
        f'  convolution_param {{\n'
        f'    num_output: {num_output}\n'
        f'    kernel_size: {kernel}\n'
        f'    stride: 1\n'
        f'  }}\n'
        f'}}\n'
    )


def make_caffemodel_bytes(weights, biases):
    """Serialize a single-layer NetParameter with conv1 weights to bytes."""
    net = ditcaffe_pb2.NetParameter()
    layer = net.layer.add()
    layer.name = "conv1"
    layer.type = "Convolution"

    w_blob = layer.blobs.add()
    w_blob.shape.dim.extend(list(weights.shape))
    w_blob.data.extend(weights.ravel().tolist())

    b_blob = layer.blobs.add()
    b_blob.shape.dim.extend(list(biases.shape))
    b_blob.data.extend(biases.ravel().tolist())

    return net.SerializeToString()


def make_calibtable(data_scale, conv1_scale):
    """INT8 calibration table (JSON) for single-layer conv1 network."""
    return {
        "data":  {"scale": data_scale,  "min": 0, "max": 0, "offset": 0},
        "conv1": {"scale": conv1_scale, "min": 0, "max": 0, "offset": 0},
    }


def compile_loadable(prototxt_path, caffemodel_path, calib_path, out_dir):
    """Invoke nvdla_compiler and return the path to the generated loadable."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PREBUILT

    cmd = [
        COMPILER,
        "--prototxt",          prototxt_path,
        "--caffemodel",        caffemodel_path,
        "--configtarget",      HW_TARGET,
        "--cprecision",        PRECISION,
        "--profile",           "fast-math",
        "--calibtable",        calib_path,
        "--quantizationMode",  "per-filter",
        "--informat",          "nchw",
        "-o",                  out_dir,
    ]
    print(f"    $ {' '.join(os.path.basename(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            cwd=out_dir)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"    (stderr) {result.stderr.strip()}")

    # Compiler writes fast-math.nvdla (named after the profile) in the -o dir
    loadable = os.path.join(out_dir, "fast-math.nvdla")
    if not os.path.exists(loadable):
        raise FileNotFoundError(
            f"{loadable} not found after compilation "
            f"(exit code {result.returncode})")
    return loadable


def write_model(name, weights, biases, data_scale, conv1_scale, out_dir):
    """Write prototxt + caffemodel + calibtable + compile to .nvdla."""
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    proto_path = os.path.join(model_dir, f"{name}.prototxt")
    model_path = os.path.join(model_dir, f"{name}.caffemodel")
    calib_path = os.path.join(model_dir, f"{name}.json")

    num_output = weights.shape[0]

    with open(proto_path, "w") as f:
        f.write(make_prototxt(num_output))
    with open(model_path, "wb") as f:
        f.write(make_caffemodel_bytes(weights, biases))
    with open(calib_path, "w") as f:
        json.dump(make_calibtable(data_scale, conv1_scale), f, indent=4)

    print(f"  [{name}] prototxt={proto_path}")
    print(f"  [{name}] caffemodel={model_path}  weights={weights.shape}  bias={biases.shape}")
    print(f"  [{name}] compiling...")
    loadable = compile_loadable(proto_path, model_path, calib_path, model_dir)

    # Rename loadable to a meaningful name
    final_path = os.path.join(model_dir, f"{name}.nvdla")
    if loadable != final_path:
        os.rename(loadable, final_path)
        loadable = final_path
    print(f"  [{name}] loadable={loadable}  ({os.path.getsize(loadable)} bytes)")
    return loadable


def main():
    ap = argparse.ArgumentParser(
        description="Generate full and split NVDLA loadables for LeNet conv1")
    ap.add_argument("--num-splits", type=int, default=2,
                    help="Number of output-channel splits (default: 2)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: ./output)")
    args = ap.parse_args()

    num_splits = args.num_splits
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output")

    # ------------------------------------------------------------------
    # 1. Extract conv1 weights from trained LeNet
    # ------------------------------------------------------------------
    print(f"Parsing {CAFFEMODEL} ...")
    net = parse_caffemodel(CAFFEMODEL)
    weights, biases = extract_conv1(net)
    num_output = weights.shape[0]
    print(f"  conv1 weights: {weights.shape}  biases: {biases.shape}")

    assert num_output % num_splits == 0, (
        f"Cannot evenly split {num_output} output channels into {num_splits} parts")
    ch_per_split = num_output // num_splits

    # Load calibration scales from the original INT8 table
    with open(CALIB_JSON) as f:
        calib = json.load(f)
    data_scale  = calib["data"]["scale"]
    conv1_scale = calib["conv1"]["scale"]

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Full single-layer conv1 (baseline)
    # ------------------------------------------------------------------
    print(f"\n--- Full conv1 ({num_output} output channels) ---")
    write_model("conv1_full", weights, biases, data_scale, conv1_scale, out_dir)

    # ------------------------------------------------------------------
    # 3. Split models
    # ------------------------------------------------------------------
    for i in range(num_splits):
        lo = i * ch_per_split
        hi = (i + 1) * ch_per_split
        print(f"\n--- Split {i} (channels {lo}..{hi-1}) ---")
        w_split = weights[lo:hi]
        b_split = biases[lo:hi]
        write_model(f"conv1_split_{i}", w_split, b_split,
                    data_scale, conv1_scale, out_dir)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generated loadables:")
    print(f"  Full:  {out_dir}/conv1_full/conv1_full.nvdla")
    for i in range(num_splits):
        lo = i * ch_per_split
        hi = (i + 1) * ch_per_split
        print(f"  Split {i} (ch {lo}-{hi-1}): "
              f"{out_dir}/conv1_split_{i}/conv1_split_{i}.nvdla")
    print(f"\nCopy these .nvdla files + seven.pgm to the FPGA, then run:")
    print(f"  ./run_split_test.sh {num_splits}")
    print("=" * 60)


if __name__ == "__main__":
    main()
