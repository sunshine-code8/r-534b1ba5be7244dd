# FAST/FINE：纯 reconstruction MAE 预训练

本入口独立于 Ghost-FWL 原有预训练/二阶段训练，以及 FAST/FINE 的监督训练。
默认使用当前 `configs/config_pretrain.yaml` 的 voxel 目录和确定性预处理参数；
不读取其中的 peak 配置、模型参数、随机 crop 尺寸和训练超参数，也不要求 peak 文件存在。
预训练配置分别为 `configs/fast_mae.yaml`、`configs/fine_mae.yaml`（相对于本目录）。

以下命令都从仓库根目录运行。

## 1. 一次性生成共用缓存

```bash
uv run --no-sync python -m experiments.wavedsp.preprocess_mae_cache \
  --config experiments/wavedsp/configs/fast_mae.yaml \
  --workers 1
```

默认原始目录是 `data/mae_dataset`。缓存实际写入
`~/datasets/Ghost-fwl/mae_dataset_cache_v1`，项目中的
`data/mae_dataset_cache_v1` 是指向该目录的软链接（当前工作区已创建）。
配置继续使用项目内的软链接路径，上面的生成命令无需改动。
FAST 和 FINE 使用同一份缓存，只需要生成一次。脚本不使用 GPU。
可以使用 `--data-root`、`--cache-root` 覆盖路径；生成和训练时要使用一致的缓存路径。
例如希望缓存放在其他磁盘时，对生成、检查和训练命令都追加 `--cache-root /path/to/cache`，
或同时修改两个新 YAML 的 `cache_root`。

缓存处理顺序与原有波形预处理一致：

1. Y 轴从 `y_crop_bottom` 裁到 `Y-y_crop_top`。
2. 时间轴从 `z_crop_front` 裁到 `T-z_crop_back`。
3. 用 `np.linspace(0, T-1, downsample_z, dtype=int)` 抽样。
4. 按源 dtype、连续 `[X,Y,T]` 保存 `.b2`。默认结果为 `[400,332,256]`。

不缓存随机 mask、不进行随机空间裁剪、不缓存峰参数或标签。
索引以源文件相对路径标识样本，跨序列的同名文件不会覆盖。
train/valid 重叠、重复文件、预处理形状不匹配会直接报错。

脚本支持原命令续建；已原子写入的文件会跳过。
`meta.json` 只在全部文件完成后发布，未完成缓存不能用于训练。
源文件大小/修改时间、目录清单或裁剪/抽样配置改变时，必须使用新缓存目录。
`--workers` 是同时解压和写入的线程数；默认 1，可根据 CPU、磁盘与内存余量增加。
同一缓存目录请只启动一个生成进程。

生成完成后检查：

```bash
uv run --no-sync python -m experiments.wavedsp.pretrain \
  --config experiments/wavedsp/configs/fast_mae.yaml \
  --check-data
```

也可在生成前用 `--raw-data --check-data` 只读检查原始数据。
2026-10-01 的当前目录检查得到 train 7,433 帧、valid 1,500 帧；
两个 split 的首帧预处理结果均为 `[400,332,256]`、uint16。
这些是 waveform-only 文件数，未按 peak 文件是否存在筛选。

## 2. 启动 FAST 预训练

选择空闲 GPU（下面以 0 为例）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m experiments.wavedsp.pretrain \
  --config experiments/wavedsp/configs/fast_mae.yaml
```

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m experiments.wavedsp.pretrain \
  --config experiments/wavedsp/configs/fast_mae.yaml
```

FINE 只需改为 `experiments/wavedsp/configs/fine_mae.yaml`。
两个配置都默认每卡 batch 8、有效 batch 32：单卡累积 4 次，双卡累积 2 次。
如果显存不足，在对应 YAML 中减小 `batch_size_per_gpu`，保持有效 batch 32。
有效 batch 必须是“每卡 batch × GPU 数”的整数倍；最后不足一个累积组时按实际微批次数归一化。
默认 100 epochs、AdamW、LR 1e-4、weight decay 0.01、无 scheduler、
BF16 主干前向、FP32 重建损失、梯度裁剪 1.0。
DataLoader 使用连续 XYT batch、源 dtype、pinned memory 和 persistent workers，
在 GPU 上转 float32 并恢复与监督训练一致的 YXT 轴序。

需要比较原始读取与缓存读取时，可追加 `--raw-data`。
该路径同样只加载 waveform，训练任务和预处理保持一致。

## 3. 任务定义与权重

`mae_model.ReconstructionWaveDSP` 只在自身实例中移除监督分类头，在完整
`forward_features()` 输出上安装分组 1×1 幅值重建头。
没有修改 `FastWaveDSP`、`FineWaveDSP` 的定义、普通 forward、
已有 embedding 辅助重建头，或监督 checkpoint 的参数结构。

默认在输入 embedding 之前遮挡空间 16×16 块，每块覆盖整条 256-bin 波形。
70% 指遮挡块比例；边缘块裁回有效图像范围，日志记录实际体素遮挡比例。
所有 skip connection 接收的也是遮挡后输入。
每个训练 epoch 重新生成 mask；验证 mask 按相对样本 ID 和固定 seed 生成，
不受 batch 大小、worker 或分布式 rank 分配影响。
验证不补齐/重复样本，并按全体验证样本的遮挡体素数量汇总 MSE。

损失仅为遮挡位置的 FP32 MSE，不计算分类、peak 参数或原 embedding 辅助重建损失。
重建梯度经过 embedding、下采样、瓶颈（含 FINE 时间注意力）和全部上采样模块。
稠密主干不会因为 mask 比例增加而自动减少计算量。

每次启动单独生成运行目录：

- FAST：`output/wavedsp/fast_mae/<运行时间>/`
- FINE：`output/wavedsp/fine_mae/<运行时间>/`

其中 `last.pt` 保存完整 epoch 的训练状态，`best.pt` 按最低验证 masked MSE 选择；
checkpoint 包含模型、优化器、完整配置、数据指纹、指标及独立 `backbone` 字段。
`backbone` 保持原 WaveDSP 主干键名，不含分类头和重建头，
供后续冻结主干的第二阶段使用。它不能直接作为现有监督入口的 `--resume` 输入：
监督阶段还需要新分类头和新的优化器。

同一预训练阶段恢复：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m experiments.wavedsp.pretrain \
  --config experiments/wavedsp/configs/fast_mae.yaml \
  --resume output/wavedsp/fast_mae/<运行时间>/last.pt
```

恢复时仍创建新运行目录。模型、mask 配置、数据指纹、优化器设置、
每卡/有效 batch 和 GPU 数必须一致；可更改 workers、缓存所在路径，
也可在等价 raw/cache 路径之间切换。随机训练 mask 和样本顺序按 epoch 确定。
中断在 epoch 中途时，从上一个完整 epoch 恢复。

## 4. 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m unittest \
  experiments.wavedsp.test_mae \
  experiments.wavedsp.test_model \
  experiments.wavedsp.test_training \
  experiments.wavedsp.test_cache \
  experiments.wavedsp.test_runs \
  experiments.wavedsp.test_evaluate
```

测试覆盖缓存逐值一致性、续建与错误配置拒绝、跨序列样本 ID、mask 边缘与稳定性、
遮挡输入防泄漏、两个模型的全部参数梯度、主干权重迁移、训练/验证循环、
梯度累积尾批、checkpoint 读写，以及双进程 CPU DDP 的不等长验证分片。
既有监督模型、损失、缓存、恢复目录及评估也包含在回归中。
这些检查不代替实际 GPU 全幅吞吐/显存测量。

## 5. 加载 MAE 主干并冻结，训练二阶段分类头

已完成的 FAST 预训练结果：
`output/wavedsp/fast_mae/20261001_161554_727455/best.pt`。
该 checkpoint 的 best 和 last 均为第 100 epoch，验证 masked MSE 为 26.08478764070982。
模型使用时间 patch_size=16、patch_dim=8（幅值 4 + 差分 4）、
stem_channels=96、bottleneck_mixer=true、time_bins=256。
空间 mask 的 16×16 与时间 embedding patch_size 是不同参数。

新配置 `experiments/wavedsp/configs/fast_mae_frozen.yaml` 显式固定了匹配的模型参数，
读取现有监督缓存 `data/ghost_dataset_cache_v1`，并设置：

```yaml
transfer:
  pretrained_checkpoint: output/wavedsp/fast_mae/20261001_161554_727455/best.pt
  freeze_backbone: true
```

该选项加载并冻结 embedding、下采样、瓶颈和全部上采样模块；
只训练新初始化的逐 bin 四分类头。FAST 冻结 311,680 个参数，训练 5,120 个参数。
冻结主干使用 eval/no_grad，优化器仅包含分类头参数。
每次加载会检查 checkpoint 类型、模型结构、数据预处理，以及完整主干键名和形状；
任何不匹配都会报错。普通监督配置不包含 transfer 时仍按原路径训练。

只检查加载和冻结（CPU，不训练）：

```bash
uv run --no-sync python -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_mae_frozen.yaml --check-transfer
```

启动单卡训练（将 GPU 0 换成空闲 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_mae_frozen.yaml
```

双卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_mae_frozen.yaml
```

输出位于 `output/wavedsp/fast_mae_frozen/<运行时间>/`。
二阶段从 epoch 1 和新优化器开始，不继承预训练优化器或 epoch。
后续中断恢复才传入二阶段目录内的 `--resume .../last.pt`；
不能将 MAE checkpoint 作为监督入口的 --resume 参数。
恢复二阶段会检查 transfer 策略一致，并直接使用二阶段 checkpoint，
不再读取原始预训练权重。

原有 `fast_ghost.yaml` 保留全监督基线用途。
其中注释的 patch_size=8、stem_channels=128 没有生效，也不匹配本次预训练，
不能用于这份 checkpoint 的二阶段训练。
`configs/config_train.yaml` 内 Ghost-FWL 自身的 patch_size/freeze_encoder/
pretrained_model_path 不控制 FAST/FINE 模型；WaveDSP 的结构和迁移选项来自上面的实验配置。
