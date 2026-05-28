"""
Roofline analysis profiling script for waferlens CUDA kernels.

Designed to run on a CUDA-capable machine (Colab, cluster, etc.).
Measures arithmetic intensity and memory bandwidth utilization for
the two flagship kernels: radial_density and connected_components.

Usage (on a CUDA machine):
    python benchmarks/roofline_profile.py [--num-wafers 1000] [--size 64]

For full Nsight Compute profiling, build the C++ benchmark binary and run:
    ncu --set full ./build/waferlens_bench
"""

import argparse
import time
import sys

import numpy as np


def generate_wafers(n, size, seed=42):
    rng = np.random.default_rng(seed)
    wafers = np.ones((n, size, size), dtype=np.uint8)
    for i in range(n):
        mask = rng.random((size, size)) < rng.uniform(0.02, 0.15)
        wafers[i][mask] = 2
    return wafers


def measure_kernel_time(analyzer, wafers, num_runs=5):
    """Measure average time per batch for warm cache."""
    for _ in range(2):
        analyzer.compute_batch(wafers)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        analyzer.compute_batch(wafers)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def estimate_memory_traffic(n, size):
    """Estimate bytes read/written per batch for the radial density kernel."""
    bytes_read = n * size * size * 1
    bytes_histogram = n * 32 * 4
    bytes_counter = n * 4
    return bytes_read + bytes_histogram + bytes_counter


def estimate_flops(n, size):
    """Estimate FLOPs for the radial density kernel (distance computation)."""
    per_cell = 6
    return n * size * size * per_cell


def main():
    parser = argparse.ArgumentParser(description="waferlens roofline profiling")
    parser.add_argument("--num-wafers", type=int, default=1000)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    try:
        import waferlens
    except ImportError:
        print("Error: waferlens not installed")
        sys.exit(1)

    analyzer = waferlens.WaferAnalyzer()
    print(f"GPU enabled: {analyzer.using_gpu()}")

    if not analyzer.using_gpu():
        print("Warning: running on CPU. Roofline analysis is most meaningful on GPU.")
        print("Proceeding with CPU measurements for reference.")
    print()

    wafers = generate_wafers(args.num_wafers, args.size)
    print(f"Profiling with {args.num_wafers} wafers ({args.size}x{args.size})")
    print()

    mean_time, std_time = measure_kernel_time(analyzer, wafers)
    throughput = args.num_wafers / mean_time

    mem_bytes = estimate_memory_traffic(args.num_wafers, args.size)
    flops = estimate_flops(args.num_wafers, args.size)

    bandwidth_gbs = (mem_bytes / mean_time) / 1e9
    gflops = (flops / mean_time) / 1e9
    arithmetic_intensity = flops / mem_bytes

    print(f"Batch time:            {mean_time*1000:.1f} +/- {std_time*1000:.1f} ms")
    print(f"Throughput:            {throughput:.0f} wafers/sec")
    print()
    print(f"Estimated memory:      {mem_bytes/1e6:.1f} MB")
    print(f"Estimated FLOPs:       {flops/1e6:.1f} MFLOP")
    print(f"Arithmetic intensity:  {arithmetic_intensity:.2f} FLOP/byte")
    print()
    print(f"Effective bandwidth:   {bandwidth_gbs:.1f} GB/s")
    print(f"Effective compute:     {gflops:.1f} GFLOP/s")
    print()

    if arithmetic_intensity < 10:
        print("Assessment: MEMORY-BOUND kernel")
        print("  Optimization targets: memory coalescing, shared memory reuse,")
        print("  reducing global memory transactions")
    else:
        print("Assessment: COMPUTE-BOUND kernel")
        print("  Optimization targets: instruction-level parallelism,")
        print("  reducing register pressure, occupancy tuning")

    print()
    print("For detailed per-kernel profiling, use NVIDIA Nsight Compute:")
    print("  ncu --set full --kernel-name radial_density_kernel ./build/waferlens_bench")
    print("  ncu --set full --kernel-name merge_kernel ./build/waferlens_bench")


if __name__ == "__main__":
    main()
