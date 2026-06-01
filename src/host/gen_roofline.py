#!/usr/bin/env python3
"""
gen_roofline.py - Generate a curated sweep of single-conv loadables that span
the arithmetic-intensity range from output-heavy to weight-heavy, for use in
constructing a roofline model on ESP-NVDLA.

The sweep is organized in groups so the analysis can hold one variable fixed
and vary another:

  reference  - Five ResNet-18-shaped reference points (already characterized)
  vary_h     - Fixed C_in=C_out=64, K=3.  H_in in {7,14,28,56,112}
  vary_c     - Fixed H_in=14, K=3.  C_in=C_out in {16,32,64,128,256,512}
  vary_k     - Fixed C_in=C_out=64, H_in=14.  K in {1,3,5,7}
  asymmetric - C_in vs C_out skewed to expose input/output-heavy regimes
  output_h   - C_in small, output spatial large -> output-write-heavy
  weight_h   - C_in*C_out large, spatial small -> weight-read-heavy

For each config, analytical metrics are computed and stored in manifest.json
alongside the loadable path:

    macs            = H_out * W_out * C_out * C_in * K^2
    ops_int8        = 2 * macs                           (1 MAC = 1 mul + 1 add)
    weight_bytes    = C_out * C_in * K^2                  (INT8)
    input_bytes     = C_in * H_in * W_in
    output_bytes    = C_out * H_out * W_out
    dense_bytes     = weight_bytes + input_bytes + output_bytes
    AI_ops_per_byte = ops_int8 / dense_bytes              (arithmetic intensity)

Cycle counts and DRAM bytes are recorded per-config by monitor_run on target;
plot_roofline.py joins them with this manifest to draw the roofline.

Usage:
    python3 gen_roofline.py [--output-dir DIR] [--seed SEED] [--only TAGS]
                            [--list]
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


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------
# (tag, group, c_in, c_out, h_in, kernel, stride, pad)
SWEEP = [
    # ----- ResNet-18 reference shapes -----
    ("ref_conv1",       "reference",   3,   64, 224, 7, 2, 3),
    ("ref_layer1",      "reference",  64,   64,  56, 3, 1, 1),
    ("ref_layer2",      "reference", 128,  128,  28, 3, 1, 1),
    ("ref_layer3",      "reference", 256,  256,  14, 3, 1, 1),
    ("ref_layer4",      "reference", 512,  512,   7, 3, 1, 1),

    # ----- vary spatial dim (fixed C=64, K=3) -----
    ("vh_h007",         "vary_h",     64,   64,   7, 3, 1, 1),
    ("vh_h014",         "vary_h",     64,   64,  14, 3, 1, 1),
    ("vh_h028",         "vary_h",     64,   64,  28, 3, 1, 1),
    ("vh_h056",         "vary_h",     64,   64,  56, 3, 1, 1),
    ("vh_h112",         "vary_h",     64,   64, 112, 3, 1, 1),

    # ----- vary channel count (fixed H=14, K=3) -----
    ("vc_c016",         "vary_c",     16,   16,  14, 3, 1, 1),
    ("vc_c032",         "vary_c",     32,   32,  14, 3, 1, 1),
    ("vc_c064",         "vary_c",     64,   64,  14, 3, 1, 1),
    ("vc_c128",         "vary_c",    128,  128,  14, 3, 1, 1),
    ("vc_c256",         "vary_c",    256,  256,  14, 3, 1, 1),
    ("vc_c512",         "vary_c",    512,  512,  14, 3, 1, 1),

    # ----- vary kernel size (fixed C=64, H=14) -----
    ("vk_k1",           "vary_k",     64,   64,  14, 1, 1, 0),
    ("vk_k3",           "vary_k",     64,   64,  14, 3, 1, 1),
    ("vk_k5",           "vary_k",     64,   64,  14, 5, 1, 2),
    ("vk_k7",           "vary_k",     64,   64,  14, 7, 1, 3),

    # ----- asymmetric C_in/C_out -----
    ("as_in3_out512",   "asymmetric",   3, 512,  56, 3, 1, 1),
    ("as_in8_out512",   "asymmetric",   8, 512,  14, 3, 1, 1),
    ("as_in512_out8",   "asymmetric", 512,   8,  14, 3, 1, 1),
    ("as_in512_out64",  "asymmetric", 512,  64,  14, 3, 1, 1),

    # ----- output-heavy: small C_in, large output spatial -----
    ("ob_in3_h224",     "output_h",     3,  32, 224, 3, 1, 1),
    ("ob_in8_h112",     "output_h",     8,  64, 112, 3, 1, 1),
    ("ob_in16_h56",     "output_h",    16,  64,  56, 3, 1, 1),

    # ----- weight-heavy: small spatial, big channels -----
    ("wb_h7_c512_k3",   "weight_h",   512, 512,   7, 3, 1, 1),
    ("wb_h7_c512_k1",   "weight_h",   512, 512,   7, 1, 1, 0),
    ("wb_h14_c512_k1",  "weight_h",   512, 512,  14, 1, 1, 0),
]


# ---------------------------------------------------------------------------
# Helpers (unchanged from gen_resnet18.py)
# ---------------------------------------------------------------------------
def make_prototxt(c_in, c_out, h, kernel, stride, pad, name="conv"):
    return (
        f'input: "data"\n'
        f'input_shape {{\n'
        f'  dim: 1\n  dim: {c_in}\n  dim: {h}\n  dim: {h}\n'
        f'}}\n'
        f'layer {{\n'
        f'  name: "{name}"\n'
        f'  type: "Convolution"\n'
        f'  bottom: "data"\n'
        f'  top: "{name}"\n'
        f'  convolution_param {{\n'
        f'    num_output: {c_out}\n'
        f'    kernel_size: {kernel}\n'
        f'    stride: {stride}\n'
        f'    pad: {pad}\n'
        f'  }}\n'
        f'}}\n'
    )


def make_caffemodel_bytes(weights, biases, name="conv"):
    net = ditcaffe_pb2.NetParameter()
    layer = net.layer.add()
    layer.name = name
    layer.type = "Convolution"

    w_blob = layer.blobs.add()
    w_blob.shape.dim.extend(list(weights.shape))
    w_blob.data.extend(weights.ravel().tolist())

    b_blob = layer.blobs.add()
    b_blob.shape.dim.extend(list(biases.shape))
    b_blob.data.extend(biases.ravel().tolist())

    return net.SerializeToString()


def make_calibtable(name="conv"):
    return {
        "data": {"scale": DATA_SCALE, "min": 0, "max": 0, "offset": 0},
        name:   {"scale": ACT_SCALE,  "min": 0, "max": 0, "offset": 0},
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


# ---------------------------------------------------------------------------
# Analytical metrics
# ---------------------------------------------------------------------------
def conv_metrics(c_in, c_out, h_in, kernel, stride, pad):
    h_out = (h_in + 2 * pad - kernel) // stride + 1
    w_in, w_out = h_in, h_out

    macs = h_out * w_out * c_out * c_in * kernel * kernel
    ops_int8 = 2 * macs

    weight_bytes = c_out * c_in * kernel * kernel  # INT8
    input_bytes  = c_in  * h_in  * w_in
    output_bytes = c_out * h_out * w_out
    dense_bytes  = weight_bytes + input_bytes + output_bytes

    return {
        "h_out":           h_out,
        "macs":            macs,
        "ops_int8":        ops_int8,
        "weight_bytes":    weight_bytes,
        "input_bytes":     input_bytes,
        "output_bytes":    output_bytes,
        "dense_bytes":     dense_bytes,
        "ai_ops_per_byte": ops_int8 / dense_bytes,
    }


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------
def emit(tag, c_in, c_out, h, kernel, stride, pad, weights, biases, out_root):
    layer_dir = os.path.join(out_root, tag)
    os.makedirs(layer_dir, exist_ok=True)

    proto  = os.path.join(layer_dir, f"{tag}.prototxt")
    cmodel = os.path.join(layer_dir, f"{tag}.caffemodel")
    calib  = os.path.join(layer_dir, f"{tag}.json")

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
    sz = os.path.getsize(final)
    print(f"  [{tag}] -> {final} ({sz} B)")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None,
                    help="default ./roofline_out")
    ap.add_argument("--only", default=None,
                    help="comma-separated tags or groups (e.g. vary_h)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true",
                    help="list configs with analytical metrics, no compile")
    args = ap.parse_args()

    out_root = args.output_dir or os.path.join(_here, "roofline_out")

    todo = SWEEP
    if args.only:
        keep = set(args.only.split(","))
        todo = [c for c in SWEEP if c[0] in keep or c[1] in keep]
        if not todo:
            print(f"warning: --only matched nothing in {keep}", file=sys.stderr)
            return 1

    if args.list:
        print(f"{'tag':<22}{'group':<12}{'C_in':>5}{'C_out':>6}"
              f"{'H_in':>5}{'K':>3}{'s':>3}{'p':>3}"
              f"{'H_out':>6}{'macs':>14}{'wB':>10}{'inB':>10}{'outB':>10}"
              f"{'AI':>9}")
        for tag, group, c_in, c_out, h, k, s, p in todo:
            m = conv_metrics(c_in, c_out, h, k, s, p)
            print(f"{tag:<22}{group:<12}{c_in:>5}{c_out:>6}"
                  f"{h:>5}{k:>3}{s:>3}{p:>3}"
                  f"{m['h_out']:>6}{m['macs']:>14}"
                  f"{m['weight_bytes']:>10}{m['input_bytes']:>10}"
                  f"{m['output_bytes']:>10}"
                  f"{m['ai_ops_per_byte']:>9.2f}")
        return 0

    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = []
    for tag, group, c_in, c_out, h, k, s, p in todo:
        m = conv_metrics(c_in, c_out, h, k, s, p)
        print(f"\n=== {tag} [{group}] : {c_in}->{c_out} {k}x{k} "
              f"s={s} p={p} {h}x{h}  AI={m['ai_ops_per_byte']:.2f} ops/B ===")

        W = rng.standard_normal((c_out, c_in, k, k)).astype(np.float32) * 0.05
        B = np.zeros((c_out,), dtype=np.float32)

        try:
            loadable = emit(tag, c_in, c_out, h, k, s, p, W, B, out_root)
        except FileNotFoundError as e:
            print(f"  [{tag}] COMPILE FAILED: {e}", file=sys.stderr)
            continue

        manifest.append({
            "tag":   tag,
            "group": group,
            "shape": {"c_in": c_in, "c_out": c_out,
                      "h_in": h, "h_out": m["h_out"],
                      "kernel": k, "stride": s, "pad": p},
            "metrics": {
                "macs":            m["macs"],
                "ops_int8":        m["ops_int8"],
                "weight_bytes":    m["weight_bytes"],
                "input_bytes":     m["input_bytes"],
                "output_bytes":    m["output_bytes"],
                "dense_bytes":     m["dense_bytes"],
                "ai_ops_per_byte": m["ai_ops_per_byte"],
            },
            "loadable": loadable,
        })

    mpath = os.path.join(out_root, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"manifest: {mpath}  ({len(manifest)} configs)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
