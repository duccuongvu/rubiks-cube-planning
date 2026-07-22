"""Benchmark: CPU per-item loop vs. torch-vectorized CPU vs. torch-vectorized
GPU, for the grasp-feasibility check that CubePlanner runs 24x6=144 times at
construction (planner.py:feasible_grasps).

144 real calls is tiny -- not enough work to amortize GPU dispatch overhead.
To show *where* GPU parallelism actually pays off, this sweeps batch size
from the real workload size up to a much larger sampling-based workload
(e.g. many candidate orientations/positions, as a real IK-backed RobotModel
might need), comparing wall-clock time and verifying numerical agreement
with the reference loop at every size.

Run:
    conda run -n ai python3 benchmark.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from grasp_math import random_rotation_matrices, reference_loop_numpy
from vectorized_grasp import batched_solve_grasp

BATCH_SIZES = [144, 1_440, 14_400, 144_000, 1_440_000]
N_REPEATS = 5


def time_call(fn, repeats=N_REPEATS):
    # warm-up (JIT/kernel compile, cuda context, page faults)
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_cpu_loop(R_np):
    return reference_loop_numpy(R_np)


def run_torch(R_t, device):
    def _call():
        out = batched_solve_grasp(R_t, device)
        if device == "cuda":
            torch.cuda.synchronize()
        return out
    return _call


def main():
    cuda_ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda_available={cuda_ok}")
    if cuda_ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    rng = np.random.default_rng(0)
    header = f"{'batch':>10} | {'cpu-loop (s)':>13} | {'cpu-vec (s)':>12} | {'gpu-vec (s)':>12} | {'cpu-loop/gpu':>13} | {'cpu-vec/gpu':>12}"
    print(header)
    print("-" * len(header))

    for n in BATCH_SIZES:
        R_np = random_rotation_matrices(n, rng)
        R_t = torch.from_numpy(R_np)

        ref = reference_loop_numpy(R_np[: min(n, 500)])  # correctness check subset
        vec_cpu_check = batched_solve_grasp(R_t[: min(n, 500)], "cpu").numpy()
        assert np.array_equal(ref, vec_cpu_check), "CPU vectorized mismatch vs reference loop!"
        if cuda_ok:
            vec_gpu_check = batched_solve_grasp(R_t[: min(n, 500)], "cuda").cpu().numpy()
            assert np.array_equal(ref, vec_gpu_check), "GPU vectorized mismatch vs reference loop!"

        # only run the slow python-loop reference for smaller sizes
        t_loop = time_call(lambda: run_cpu_loop(R_np), repeats=1) if n <= 14_400 else float("nan")
        t_cpu = time_call(run_torch(R_t, "cpu"))
        t_gpu = time_call(run_torch(R_t, "cuda")) if cuda_ok else float("nan")

        speedup_loop = t_loop / t_gpu if cuda_ok and t_loop == t_loop else float("nan")
        speedup_cpu = t_cpu / t_gpu if cuda_ok else float("nan")

        print(f"{n:>10} | {t_loop:>13.5f} | {t_cpu:>12.5f} | {t_gpu:>12.5f} | {speedup_loop:>13.1f} | {speedup_cpu:>12.1f}")

    print("\ncorrectness: all vectorized (CPU + GPU) outputs match the reference per-item loop.")


if __name__ == "__main__":
    main()
