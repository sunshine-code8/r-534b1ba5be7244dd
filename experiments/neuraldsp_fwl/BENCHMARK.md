# Full-frame candidate benchmark

Run from the Ghost-FWL repository root in your existing PyTorch environment.
No dataset, checkpoint, training configuration, or original NeuralLidarDSP
checkout is required for the new candidates. They reuse the local NeuralDSP
attention primitives; training models are unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_original_neuraldsp.py \
  --candidate all --batch-size 1 --precision fp32 \
  --warmup 20 --iterations 50 --repeats 5 --skip-flops \
  --csv output/neuraldsp_benchmark/fp32.csv
```

The default input is `[B,336,400,256,1]`, and the output is contiguous
`[B,336,400,256,4]` **logits**, not probabilities or argmax labels.

| Candidate | Temporal patch | Channels at full, half, quarter, eighth resolution | Five stage depths |
| --- | --- | --- | --- |
| `preferred` | 16 | 16, 16, 32, 64 | 0, 1, 2, 1, 0 |
| `fast` | 16 | 16, 16, 32, 64 | 0, 1, 1, 1, 0 |
| `patch32` | 32 | 16, 16, 32, 64 | 0, 1, 1, 1, 0 |

Replace `all` with one candidate name to measure only that model. The `CANDIDATES`
dictionary near the top of `benchmark_original_neuraldsp.py` can also be edited
directly. CLI overrides take precedence over that dictionary:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_original_neuraldsp.py \
  --candidate preferred --patch-size 16 \
  --channels 16 16 32 64 --depths 0 1 2 1 0 \
  --warmup 20 --iterations 50 --repeats 5 --skip-flops
```

- `--channels C0 C1 C2 C3` independently sets the widths at spatial scales
  `1, 1/2, 1/4, 1/8`. Alternatively, `--dim D` sets `(D,D,2D,4D)`.
- `--depths E1 E2 BOT D2 D1` independently sets Transformer counts at scales
  `1/2, 1/4, 1/8, 1/4, 1/2`. Zero skips the Transformer but retains merging,
  expansion, and additive skip fusion. Decoder attention starts with a shifted
  window, even when its depth is one.
- `--patch-size P` must divide `--time-bins` (default 256). Spatial dimensions
  must be divisible by 8. Channels must be positive and even, and widths of active
  attention stages must be divisible by `--heads` (default 2).
- `--overall-only` measures only the complete forward pass.
- `--precision fp16` or `--precision bf16` casts both the model and its input;
  these are explicit reduced-precision runs, not autocast runs.
- The default `--tf32 off` disables TF32 for both matrix multiplication and
  cuDNN. `--tf32 on` enables both. Printed settings and CSV record both flags;
  check them when comparing with older FP32 results that may have enabled TF32.
- Remove `--skip-flops` to use PyTorch's operator FLOP counter (requires a
  PyTorch version with `torch.utils.flop_counter`, such as 2.5). FLOP counting is
  outside timing; multiply-add is two FLOPs, and some elementwise ops are omitted.
- `--csv PATH` writes all stage results and actual configurations, overwriting
  that file if it already exists. Use a different filename for each ablation.

`Overall` includes matched filtering, tokenization, the full backbone, recovery
to full spatial resolution, and the dense four-class head. Timing uses warmed-up
eager inference and CUDA synchronization. It excludes input generation, data I/O,
host/device transfer, softmax, argmax, and other postprocessing. The model uses
random weights, so these results do not evaluate classification quality.

Overall timing happens before stage capture hooks are installed; isolated stage
timings are diagnostic and should not be summed to claim an end-to-end speed.
FPS is `batch_size * 1000 / mean_ms_per_batch`. For batch size one, 50 FPS means
at most 20 ms per forward pass. Standard deviation is across repeated mean
latencies, not individual-frame latency percentiles. Peak allocated CUDA memory
is measured during overall timing and includes model and input allocations.

For the default candidates, attention operates at `84x100` and `42x50`.
The latter is padded to `42x52` *inside spatial attention only*. Padded keys are
masked and discarded afterward; no real input pixels or bins are cropped.

The previous point-return-head benchmark remains available with
`--candidate original --neuraldsp-repo /path/to/NeuralLidarDSP`; it retains its
previous hardcoded architecture and FP32 settings and requires `einops`.
Candidate architecture overrides do not apply to that legacy mode.

Correctness checks (small CPU inputs and full-frame meta shapes):

```bash
OMP_NUM_THREADS=1 python -m experiments.neuraldsp_fwl.test_benchmark_candidates
```
