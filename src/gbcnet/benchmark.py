"""Inference latency / FPS / FLOPs for a binary registry model.

    python -m gbcnet.benchmark --model ConvNeXtTiny_CBAM_MSAM --device cuda --out benchmark_gpu.json
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .models.registry import MODEL_REGISTRY, build_model
from .utils import environment_info, get_device, save_json


def main():
    parser = argparse.ArgumentParser(description="Benchmark a binary GBC model")
    parser.add_argument("--model", default="ConvNeXtTiny_CBAM_MSAM", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()

    device = get_device(args.device)
    model = build_model(args.model, pretrained=False).to(device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    dummy = torch.randn(1, 1, args.image_size, args.image_size, device=device)

    gmacs = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gmacs = macs / 1e9
    except Exception as exc:  # thop missing or unsupported op
        print(f"FLOP count skipped: {exc}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)
        sync()
        for _ in range(args.runs):
            sync()
            t0 = time.perf_counter()
            model(dummy)
            sync()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    result = {
        "model": args.model,
        "device": str(device),
        "input_shape": [1, 1, args.image_size, args.image_size],
        "mean_latency_ms": float(arr.mean()),
        "std_latency_ms": float(arr.std()),
        "median_latency_ms": float(np.median(arr)),
        "fps": float(1000.0 / arr.mean()),
        "gmacs": gmacs,
        "gflops_2x_macs": None if gmacs is None else 2 * gmacs,
        "params": sum(p.numel() for p in model.parameters()),
        "environment": environment_info(),
    }
    save_json(result, args.out)
    print({k: v for k, v in result.items() if k != "environment"})


if __name__ == "__main__":
    main()
