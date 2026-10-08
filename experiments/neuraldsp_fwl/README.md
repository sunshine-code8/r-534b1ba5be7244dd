# NeuralDSP/FWL isolated experiments

This directory contains the three requested NeuralDSP experiments. It imports
the original dataset readers but does not modify `src/`, `configs/`, or the
original launch scripts. All checkpoints are written below
`output/neuraldsp_fwl/`.

Production commands support either one GPU or DDP with multiple GPUs. Run them from
the Ghost-FWL repository root. The configured effective batch size stays at 32;
gradient accumulation is derived from the number of processes automatically.

Single-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 \
  -m experiments.neuraldsp_fwl.train \
  --config experiments/neuraldsp_fwl/configs/exp1_pretrain.yaml
```

## Experiment 1: reconstruction pretraining, then frozen-backbone classification

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m experiments.neuraldsp_fwl.train \
  --config experiments/neuraldsp_fwl/configs/exp1_pretrain.yaml

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m experiments.neuraldsp_fwl.train \
  --config experiments/neuraldsp_fwl/configs/exp1_frozen_head.yaml
```

The second command loads `output/neuraldsp_fwl/exp1_pretrain/best.pt`, freezes
the complete NeuralDSP backbone, and trains only the classification head.

## Experiment 2: scratch training with experiment-1 stage-2 hyperparameters

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m experiments.neuraldsp_fwl.train \
  --config experiments/neuraldsp_fwl/configs/exp2_scratch_same.yaml
```

## Experiment 3: scratch training with a tuned schedule

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m experiments.neuraldsp_fwl.train \
  --config experiments/neuraldsp_fwl/configs/exp3_scratch_tuned.yaml
```

Every run saves `best.pt` and `last.pt` under its own output directory.
