# gpu_benchmark

Tests the parallel-computation claim from the planner analysis: `CubePlanner`'s
grasp-feasibility check (`RobotModel.solve_grasp`, called 24 orientations × 6
faces = 144 times in `feasible_grasps`) is independent per `(face,
orientation)` pair, so it can be vectorized and run on GPU instead of as a
Python loop.

## Files

- `grasp_math.py` — constants (from `../config.yaml`) + `reference_loop_numpy()`,
  a literal per-item Python loop identical to `MockRobot.solve_grasp`
  (`../examples/mock_robot.py`). This is the correctness baseline.
- `vectorized_grasp.py` — `batched_solve_grasp(R_batch, device)`: the same math,
  vectorized with `torch.einsum` over the batch and face dimensions. Runs
  unchanged on `"cpu"` or `"cuda"`.
- `benchmark.py` — sweeps batch size, checks the vectorized CPU/GPU outputs
  exactly match the reference loop, and times all three.

## Running

```bash
conda activate ai   # torch + cuda already installed there
conda run -n ai python3 benchmark.py
```

## Results (RTX A2000 8GB Laptop GPU, torch 2.11.0+cu128)

```
     batch |  cpu-loop (s) |  cpu-vec (s) |  gpu-vec (s) |  cpu-loop/gpu |  cpu-vec/gpu
---------------------------------------------------------------------------------------
       144 |       0.00261 |      0.00061 |      0.00081 |           3.2 |          0.7
      1440 |       0.02520 |      0.00074 |      0.00066 |          38.0 |          1.1
     14400 |       0.26251 |      0.00191 |      0.00122 |         214.7 |          1.6
    144000 |           nan |      0.01480 |      0.00634 |           nan |          2.3
   1440000 |           nan |      0.20794 |      0.05381 |           nan |          3.9
```

`cpu-loop` is what `CubePlanner.feasible_grasps` actually does today (a plain
Python loop, one `solve_grasp` call at a time). `cpu-vec`/`gpu-vec` batch all
combinations into one set of tensor ops.

## Takeaways

- **At the real workload size (144 calls)**: vectorizing on CPU alone gives
  ~4x over the current Python loop. GPU is *slower* than CPU-vectorized here
  (0.7x) — kernel-launch and host↔device transfer overhead dominates; there
  isn't enough work to amortize it.
- **The Python loop is what actually doesn't scale**: at just 14,400 items
  (100x the real workload) it already takes 260ms, ~215x slower than the GPU
  version, because it pays Python-level dict/attribute overhead per call
  instead of one batched tensor op.
- **GPU wins only past ~100k-item batches**, reaching 3.9x over CPU-vectorized
  at 1.44M items — relevant only if `solve_grasp` were used across a much
  larger sampling space (e.g. many candidate orientations/positions for a
  real IK-backed `RobotModel`), not for the fixed 24-orientation table this
  repo actually needs.
- **Conclusion**: for this repo's actual problem size, switching
  `feasible_grasps` to a batched CPU tensor op would be a real (~4x), free
  win. GPU is not worth the added dependency unless the feasibility oracle's
  input space grows by 3+ orders of magnitude (e.g. sampling many grasp
  candidates per face for a real IK solver instead of one closed-form check).
