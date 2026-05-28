"""
Throughput benchmark: waferlens (C++/CUDA) vs NumPy baseline.

Measures wall-clock time to process N synthetic wafer maps through the
complete feature extraction pipeline and reports wafers/second.

Usage:
    python benchmarks/benchmark_throughput.py [--num-wafers 1000] [--size 64]
"""

import argparse
import time
import sys

import numpy as np


def generate_synthetic_wafers(n: int, size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wafers = np.ones((n, size, size), dtype=np.uint8)

    for i in range(n):
        pattern = rng.choice(["center", "edge", "scratch", "random", "none"],
                             p=[0.2, 0.2, 0.2, 0.3, 0.1])

        cr, cc = (size - 1) / 2.0, (size - 1) / 2.0
        rr, ccarr = np.mgrid[:size, :size]
        dist = np.sqrt((rr - cr) ** 2 + (ccarr - cc) ** 2)
        max_r = np.sqrt(cr ** 2 + cc ** 2)

        if pattern == "center":
            radius = rng.uniform(3, size // 4)
            wafers[i][dist <= radius] = 2

        elif pattern == "edge":
            threshold = rng.uniform(0.75, 0.9) * max_r
            wafers[i][dist >= threshold] = 2

        elif pattern == "scratch":
            angle = rng.uniform(0, np.pi)
            proj = (rr - cr) * np.cos(angle) + (ccarr - cc) * np.sin(angle)
            wafers[i][np.abs(proj) < 1.5] = 2

        elif pattern == "random":
            mask = rng.random((size, size)) < rng.uniform(0.01, 0.1)
            wafers[i][mask] = 2

    return wafers


def benchmark_numpy(wafers: np.ndarray) -> float:
    sys.path.insert(0, "benchmarks")
    import numpy_baseline as ref

    start = time.perf_counter()
    for i in range(len(wafers)):
        ref.compute_all(wafers[i])
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_waferlens(wafers: np.ndarray) -> float:
    try:
        import waferlens
    except ImportError:
        return -1.0

    analyzer = waferlens.WaferAnalyzer()

    start = time.perf_counter()
    _ = analyzer.compute_batch(wafers)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="waferlens throughput benchmark")
    parser.add_argument("--num-wafers", type=int, default=1000)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    print(f"Generating {args.num_wafers} synthetic wafers ({args.size}x{args.size})...")
    wafers = generate_synthetic_wafers(args.num_wafers, args.size)

    print("Benchmarking NumPy baseline...")
    np_time = benchmark_numpy(wafers)
    np_throughput = args.num_wafers / np_time
    print(f"  NumPy:     {np_time:.3f}s  ({np_throughput:.0f} wafers/sec)")

    print("Benchmarking waferlens (C++/CUDA)...")
    wl_time = benchmark_waferlens(wafers)
    if wl_time < 0:
        print("  waferlens: not installed (skipped)")
    else:
        wl_throughput = args.num_wafers / wl_time
        speedup = np_time / wl_time
        print(f"  waferlens: {wl_time:.3f}s  ({wl_throughput:.0f} wafers/sec)")
        print(f"  Speedup:   {speedup:.1f}x")


if __name__ == "__main__":
    main()
