#!/usr/bin/env python3
"""
gen_resnet18.py - Generate baseline + output-channel-split NVDLA loadables for
every distinct convolution shape in ResNet-18.

Each conv is compiled in isolation as a one-layer Caffe net. For every shape
we emit:
  - one full baseline loadable
  - N split loadables, each holding C_out/N output channels

The intent is per-layer timing + memory profiling on an ESP SoC with multiple
NVDLA tiles. NVDLA cycle counts and DMA byte counts depend on tensor shape,
not on weight values, so we use deterministic random weights and placeholder
INT8 calibration scales rather than the actual ImageNet-trained ResNet-18.

Output layout:
    <out>/<layer>_full/<layer>_full.nvdla
    <out>/<layer>_splitK_ofN/<layer>_splitK_ofN.nvdla
    <out>/manifest.json     # index of every loadable, grouped by layer

Usage:
    python3 gen_resnet18.py [--num-splits N] [--output-dir DIR]
                            [--only conv1,layer4_3x3] [--seed S]
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ditcaffe_pb2

_here = os.path.dirname(os.path.abspath(__file__))
_soc = os.path.dirname(_here)
ESP_ROOT = os.environ.get("ESP_ROOT", os.path.dirname(os.path.dirname(_soc)))
NVDLA_ROOT = os.path.join(ESP_ROOT, "accelerators/third-party/NV_NVDLA")
PREBUILT = os.path.join(NVDLA_ROOT, "sw/prebuilt/x86-ubuntu")
COMPILER = os.path.join(PREBUILT, "nvdla_compiler")

HW_TARGET = "nv_small"
PRECISION = "int8"
DATA_SCALE = 1.0 / 127.0
ACT_SCALE = 1.0 / 127.0

# (tag, c_in, c_out, h_in, kernel, stride, pad)
# Spatial is always square in ResNet-18.
RESNET18_CONVS = [
    ("conv1",         3,  64, 224, 7, 2, 3),
    ("layer1_3x3",   64,  64,  56, 3, 1, 1),
    ("layer2_0_3x3", 64, 128,  56, 3, 2, 1),
    ("layer2_3x3", 128, 128,  28, 3, 1, 1),
    ("layer2_0_ds",  64, 128,  56, 1, 2, 0),
    ("layer3_0_3x3", 128, 256,  28, 3, 2, 1),
    ("layer3_3x3", 256, 256,  14, 3, 1, 1),
    ("layer3_0_ds", 128, 256,  28, 1, 2, 0),
    ("layer4_0_3x3", 256, 512,  14, 3, 2, 1),
    ("layer4_3x3", 512, 512,   7, 3, 1, 1),
    ("layer4_0_ds", 256, 512,  14, 1, 2, 0),
]


def make_prototxt(c_in, c_out, h, kernel, stride, pad, layer_name="conv"):
    return (
        f'input: "data"\n'
        f'input_shape {{\n'
        f'  dim: 1\n  dim: {c_in}\n  dim: {h}\n  dim: {h}\n'
        f'}}\n'
        f'layer {{\n'
        f'  name: "{layer_name}"\n'
        f'  type: "Convolution"\n'
        f'  bottom: "data"\n'
        f'  top: "{layer_name}"\n'
        f'  convolution_param {{\n'
        f'    num_output: {c_out}\n'
        f'    kernel_size: {kernel}\n'
        f'    stride: {stride}\n'
        f'    pad: {pad}\n'
        f'  }}\n'
        f'}}\n'
    )


def make_caffemodel_bytes(weights, biases, layer_name="conv"):
    net = ditcaffe_pb2.NetParameter()
    layer = net.layer.add()
    layer.name = layer_name
    layer.type = "Convolution"

    w_blob = layer.blobs.add()
    w_blob.shape.dim.extend(list(weights.shape))
    w_blob.data.extend(weights.ravel().tolist())

    b_blob = layer.blobs.add()
    b_blob.shape.dim.extend(list(biases.shape))
    b_blob.data.extend(biases.ravel().tolist())

    return net.SerializeToString()


def make_calibtable(layer_name="conv"):
    return {
        "data":     {"scale": DATA_SCALE, "min": 0, "max": 0, "offset": 0},
        layer_name: {"scale": ACT_SCALE,  "min": 0, "max": 0, "offset": 0},
    }


def compile_loadable(prototxt, caffemodel, calib, out_dir):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PREBUILT
    cmd = [
        COMPILER,
        "--prototxt",         prototxt,
        "--caffemodel",       caffemodel,
        "--configtarget",     HW_TARGET,
        "--cprecision",       PRECISION,
        "--profile",          "fast-math",
        "--calibtable",       calib,
        "--quantizationMode", "per-filter",
        "--informat",         "nchw",
        "-o",                 out_dir,
    ]
    print(f"    $ {' '.join(os.path.basename(c) for c in cmd)}")
    r = subprocess.run(cmd, env=env, cwd=out_dir,
                       capture_output=True, text=True)
    if r.stdout.strip():
        print(f"    {r.stdout.strip()}")
    if r.stderr.strip():
        print(f"    (stderr) {r.stderr.strip()}")
    raw = os.path.join(out_dir, "fast-math.nvdla")
    if not os.path.exists(raw):
        raise FileNotFoundError(f"{raw} missing (rc={r.returncode})")
    return raw


def emit_one(tag, c_in, c_out, h, kernel, stride, pad,
             weights, biases, out_root):
    layer_dir = os.path.join(out_root, tag)
    os.makedirs(layer_dir, exist_ok=True)

    proto = os.path.join(layer_dir, f"{tag}.prototxt")
    cmodel = os.path.join(layer_dir, f"{tag}.caffemodel")
    calib = os.path.join(layer_dir, f"{tag}.json")

    with open(proto, "w") as f:
        f.write(make_prototxt(c_in, c_out, h, kernel, stride, pad))
    with open(cmodel, "wb") as f:
        f.write(make_caffemodel_bytes(weights, biases))
    with open(calib, "w") as f:
        json.dump(make_calibtable(), f, indent=2)

    print(f"  [{tag}] {c_in}->{c_out} {kernel}x{kernel} "
          f"s={stride} p={pad} {h}x{h}")
    raw = compile_loadable(proto, cmodel, calib, layer_dir)
    final = os.path.join(layer_dir, f"{tag}.nvdla")
    if raw != final:
        os.rename(raw, final)
    print(f"  [{tag}] -> {final} ({os.path.getsize(final)} B)")
    return final


def main():
    ap = argparse.ArgumentParser(
        description="ResNet-18 per-layer baseline + O-channel split loadables")
    ap.add_argument("--num-splits", type=int, default=2,
                    help="N-way output-channel split (default: 2)")
    ap.add_argument("--output-dir", default=None,
                    help="output dir (default: ./resnet18_out)")
    ap.add_argument("--only", default=None,
                    help="comma-separated tags to build (default: all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for placeholder weights")
    args = ap.parse_args()

    n = args.num_splits
    out_root = args.output_dir or os.path.join(_here, "resnet18_out")
    os.makedirs(out_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    todo = RESNET18_CONVS
    if args.only:
        keep = set(args.only.split(","))
        todo = [c for c in RESNET18_CONVS if c[0] in keep]
        missing = keep - {c[0] for c in todo}
        if missing:
            print(f"warning: unknown tags {sorted(missing)}", file=sys.stderr)

    manifest = []
    for tag, c_in, c_out, h, k, s, p in todo:
        if c_out % n != 0:
            print(f"skip {tag}: C_out={c_out} not divisible by N={n}",
                  file=sys.stderr)
            continue
        ch_per = c_out // n

        # Small-magnitude random weights so INT8 calibration is reasonable.
        W = rng.standard_normal((c_out, c_in, k, k)).astype(np.float32) * 0.05
        B = np.zeros((c_out,), dtype=np.float32)

        print(f"\n=== {tag} : {c_in}->{c_out} {k}x{k} s={s} p={p} {h}x{h} ===")

        full = emit_one(f"{tag}_full",
                        c_in, c_out, h, k, s, p, W, B, out_root)
        splits = []
        for i in range(n):
            lo, hi = i * ch_per, (i + 1) * ch_per
            split_tag = f"{tag}_split{i}_of{n}"
            splits.append(emit_one(
                split_tag, c_in, ch_per, h, k, s, p,
                W[lo:hi], B[lo:hi], out_root))

        manifest.append({
            "layer": tag,
            "shape": {"c_in": c_in, "c_out": c_out, "h": h,
                      "kernel": k, "stride": s, "pad": p},
            "full": full,
            "splits": splits,
        })

    mpath = os.path.join(out_root, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Manifest: {mpath}")
    print(f"  {len(manifest)} layer(s) x (1 full + {n} splits)")
    print("=" * 60)


if __name__ == "__main__":
    main()
