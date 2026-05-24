#!/usr/bin/env python3
import argparse
import time

import torch

from syntropy_field_expanded import FieldEnhancedSNN


def run_steps(module: FieldEnhancedSNN, steps: int, dt: float) -> float:
    start = time.perf_counter()
    for _ in range(steps):
        module.pde_step(dt=dt)
    if module.phi1.is_cuda:
        torch.cuda.synchronize(module.phi1.device)
    end = time.perf_counter()
    return end - start


def clone_states(src: FieldEnhancedSNN, dst: FieldEnhancedSNN):
    dst.phi1.copy_(src.phi1)
    dst.phi5.copy_(src.phi5)
    dst.Phi.copy_(src.Phi)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FieldEnhancedSNN PDE step with optional CUDA kernel")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA device.")
        return

    device = torch.device("cuda")

    # Baseline: force fallback path
    baseline = FieldEnhancedSNN(size=args.size, device=device)
    baseline.use_cuda_pde = False

    # Candidate: custom CUDA kernel path (if extension loads)
    candidate = FieldEnhancedSNN(size=args.size, device=device)
    clone_states(baseline, candidate)

    # Warmup
    run_steps(baseline, steps=20, dt=args.dt)
    run_steps(candidate, steps=20, dt=args.dt)

    # Timed runs
    baseline_t = run_steps(baseline, steps=args.steps, dt=args.dt)
    candidate_t = run_steps(candidate, steps=args.steps, dt=args.dt)

    # Numerical drift check
    phi1_diff = (baseline.phi1 - candidate.phi1).abs().mean().item()
    phi5_diff = (baseline.phi5 - candidate.phi5).abs().mean().item()

    print("PDE Benchmark Results")
    print(f"  grid={args.size}x{args.size} steps={args.steps} dt={args.dt}")
    print(f"  fallback_seconds={baseline_t:.6f}")
    print(f"  cuda_kernel_seconds={candidate_t:.6f}")
    if candidate_t > 0:
        print(f"  speedup_x={baseline_t / candidate_t:.3f}")
    print(f"  mean_abs_diff_phi1={phi1_diff:.6e}")
    print(f"  mean_abs_diff_phi5={phi5_diff:.6e}")


if __name__ == "__main__":
    main()
