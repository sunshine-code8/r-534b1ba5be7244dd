# FastWaveDSP 和 FineWaveDSP

仓库中的这两个模型现在只保留增强后的结构。原有的 `FastWaveDSP`、`FineWaveDSP` 类名及
`benchmark_wavedsp.py --model fast|fine|all` 入口不变；已移除 `variant` 参数和旧版实现。

|  | FastWaveDSP | FineWaveDSP |
| --- | --- | --- |
| 时间 patch | P=16，16 个 patch | P=4，64 个 patch |
| 局部 embedding | 4 维幅值 + 4 维 patch 内差分 | 4 维，带原始 4-bin 残差 |
| Embedding → U-Net | 128→96 通道 | 256→128 通道 |
| U-Net E1/E2/E3 | 96/128/128 | 128/160/256 |
| 时间注意力 | 无 | 仅在 42×50 瓶颈，2 heads，带相对距离偏置 |
| 分类输出 | 逐 bin 四分类 logits | 逐 bin 四分类 logits |

Fast 的差分只计算同一 patch 内相邻 bin 的差值。Fine 的 embedding 为
`SiLU(grouped_projection(x)) + x`，之后用 1×1 卷积压回 128 通道。
后续 U-Net 和分组分类头保持前次增强版的宽度；没有全分辨率多通道时间隐藏特征。

默认接口：`[B,H,W,T,1] → [B,H,W,T,4]`，T=256。也支持
`layout="channels_first"` 的 `[B,1,H,W,T] → [B,4,H,W,T]`。
`model.predict(x)` 返回 GPU 上的 `[B,H,W,T]` uint8 类别。

可选的 `auxiliary_reconstruction=True` 会加入训练用的 waveform 重建头。
只在显式调用 `forward_with_aux(x)` 时使用；普通前向和测速不运行辅助头。
`waveform_auxiliary_loss` 可计算 waveform L1 与时间差分 L1。

## 测速

在仓库根目录选择空闲 GPU，运行：

```bash
python benchmark_wavedsp.py --model all --device cuda:7 --precision fp32 \
  --batch-size 1 --warmup 5 --iterations 20 --repeats 5 \
  --csv output/wavedsp/fast_fine_fp32.csv
```

FP16 将 `--precision` 改为 `fp16`，并使用不同的 CSV 文件名。
`--model fast` 或 `--model fine` 可单独测试。
`--include-neuraldsp user_patch32` 可加入此前 P=32、depths=(0,1,2,1,0) 的对照模型。
测速包括模型前向、输入和输出布局转换；不包括数据加载与主机传输。
脚本按论文表格的单位打印 `Params [M]`（参数个数 ÷ 10⁶）和 `FLOPs [G]`
（单个输入样本的前向运算次数 ÷ 10⁹）。batch size 大于 1 时额外显示整个 batch 的计算量。
FPS 仍为每秒处理的输入样本数；指定 `--height 332 --width 400 --time-bins 256`
时就是完整帧 FPS，无需换算。CSV 的 `parameters_m`、`gflops_per_sample`、`fps`
对应这三项；同时保留原始参数个数、可训练参数和 batch 计算量字段。

`parameter storage` 是按当前 dtype 计算的参数张量存储量，`peak allocated` 是推理期间
PyTorch 分配的峰值显存，单位均为 MiB（2²⁰ 字节），与参数量的 M（10⁶ 个）不同。
延迟单位为 ms/batch，± 表示各轮重复测试平均延迟的标准差。
计算量使用与 NeuralDSP 基准相同的
PyTorch 算子计数器，乘加按 2 FLOPs，部分逐元素操作、布局拷贝和 labels 的 argmax
不在统计范围内。因此它是统一口径的前向算子估计，不能用于推算训练 FLOPs。
与其他论文比较时，除单位外还需核实 FLOPs 的输入尺寸、乘加计数规则和算子覆盖范围；
Ghost-FWL 表格截图只明确两列 FPS 的输入尺寸，不能据此确认 FLOPs 的输入尺寸。

只看模型规模，不做重复 FPS 测速；Ghost 实际高度为 332，Fine 的训练配置使用
`attention_chunk_size=256`：

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python benchmark_wavedsp.py \
  --model all --device cuda:0 --precision fp32 \
  --height 332 --width 400 --time-bins 256 --batch-size 1 \
  --attention-chunk-size 256 --metrics-only \
  --csv output/wavedsp/complexity_332x400x256_fp32.csv
```

原测速命令也会输出这些规模信息；若只需测速，可加 `--skip-flops`。

此前增强版的实测结果保存在 [BENCHMARK.md](BENCHMARK.md)。本次仅清理旧版实现和入口，
未重新测速；旧 CSV 保留为历史数据。

## Ghost 全幅监督训练与一次性缓存

Fast 和 Fine 共用相同的 Ghost 预处理缓存。预处理保持现有 `configs/config_train.yaml`
里的 Y/T 裁剪和 `np.linspace(..., dtype=int)` 时间抽样规则，结果尺寸为
`[X,Y,T]=[400,332,256]`。缓存保留体素原始 dtype（通常为 uint16）和标签整数 dtype；
训练时体素搬到 GPU 后转成 float32。缓存按原始 `scene/data|annotation_v1_expand/hist`
目录结构保存成 `.b2` 文件，并在根目录生成 `build_state.json`、
`train_index.json`、`valid_index.json` 和完成标志 `meta.json`。

项目软链接 `data/ghost_dataset_cache_v1` 指向
`/home/fanrundi/datasets/Ghost-fwl/ghost_dataset_wavedsp_cache_v1`。
在手动生成前，软链接的目标尚不存在。请从仓库根目录运行一次：

```bash
uv run python -m experiments.wavedsp.preprocess_cache \
  --config experiments/wavedsp/configs/fast_ghost.yaml \
  --data-root data/ghost_dataset \
  --cache-root data/ghost_dataset_cache_v1 \
  --workers 1
```

缓存生成与正在运行的训练会争用磁盘/CPU；如果当前训练仍在跑，建议等其结束后再执行。
脚本显示 train/valid 帧数进度；中断后用同一命令重跑，会跳过已写好的成对文件。
`meta.json` 只在全部帧完成后写入，因此未完成的缓存不会被训练误用。
若数据文件或预处理设置发生变化，应使用新的缓存版本/目录；脚本会拒绝混用。
`--workers` 可按磁盘吞吐和内存容量适当增加。

完成后检查缓存的首帧及两个划分：

```bash
uv run python -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_ghost.yaml --check-data
```

两个实验配置现在还启用 `training.native_cache_layout: true`：DataLoader 按缓存原本连续的
`[X,Y,T]` 形状拼 batch，然后在 GPU 上恢复网络和损失函数需要的轴顺序。
原始 `.b2` 缓存无需重新生成。训练用 focal loss 也改成固定形状的掩码求和，避免
CUDA 布尔索引带来的逐 batch 同步；现有 checkpoint 可以继续使用。
已经启动的进程不会自动加载这些代码改动，需在下次启动或 `--resume` 后生效。
如需对照原布局，在配置中设 `native_cache_layout: false`。
`training.prefetch_factor` 默认仍是 1，可在实验时单独设置。

可在空闲 GPU 上重现不写 checkpoint 的短程分析：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync python -m experiments.wavedsp.profile_training \
  --config experiments/wavedsp/configs/fine_ghost.yaml \
  --device cuda:0 --batch-size 8 --workers 8 --native-layout \
  --warmup 1 --steps 12 --loss current
```

`profile_loader.py` 可用 `--native-layout`、`--no-pin-memory` 和 `--prefetch-factor`
分别比较拼 batch、固定内存和预取。阶段计时每批同步一次 GPU，属于诊断数据，
不能直接等同于双卡 DDP 的整轮训练速度。

两个实验配置都默认从 `data/ghost_dataset_cache_v1` 读取，可分别启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_ghost.yaml

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fine_ghost.yaml
```

多卡使用 `CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run --standalone --nproc_per_node=2 -m experiments.wavedsp.train --config <配置文件>`。
每卡 batch、有效 batch、worker 数以各自 YAML 的 `training` 字段为准。
如果需要继续使用原始读取路径，可加 `--raw-data --data-root data/ghost_dataset`；
`--cache-root` 可以覆盖配置中的缓存路径。已有 checkpoint 可通过 `--resume <last.pt>` 从下一个 epoch 继续；模型、优化器、调度器和最佳指标会恢复。
每次新启动都会在配置的 `output_dir` 下创建 `YYYYMMDD_HHMMSS_ffffff` 子目录，
该目录单独保存 `run_info.json`、`last.pt`、`best.pt`。恢复训练也会创建新目录，
并先复制来源 `last.pt`/`best.pt`，因此不覆盖上一次的文件。
注意已在运行的旧进程仍使用启动时加载的旧代码，会继续写入旧的固定路径。
建议等其当前 epoch 写完 checkpoint 后再停掉，确认文件修改时间，然后用如下命令恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  -m experiments.wavedsp.train \
  --config experiments/wavedsp/configs/fast_ghost.yaml \
  --resume output/wavedsp/fast_ghost/last.pt
```

Fine 使用 `fine_ghost.yaml` 和对应的 `fine_ghost/last.pt`。
恢复时可以从原始数据读取切换到等价的预处理缓存；请保持模型结构和目标尺寸配置一致。
checkpoint 只在完整 epoch 结束后保存，中途停止当前 epoch 会从上一个完整 epoch 重新运行。


## Fast/Fine 独立测试

从仓库根目录使用训练完成的某次运行目录中的 `best.pt`。下例中的路径换成实际运行目录；
Fast 和 Fine 分别运行，结果默认保存在该 checkpoint 所在目录的 `evaluation/` 下。
原有 FWLMAE 的 `scripts/run_estimate.py`、`scripts/run_test.py` 和配置不受影响。

```bash
uv run python -m experiments.wavedsp.evaluate all \
  --model fast --checkpoint output/wavedsp/fast_ghost/<运行目录>/best.pt \
  --device cuda:0

uv run python -m experiments.wavedsp.evaluate all \
  --model fine --checkpoint output/wavedsp/fine_ghost/<运行目录>/best.pt \
  --device cuda:0
```

`all` 按顺序运行 `estimate`、`recall`、`pcd`、`ghost-removal`；也可以将 `all`
替换为任一阶段单独运行。`estimate` 生成官方命名的
`evaluation/estimate/*_prediction_voxel.b2`；`pcd` 调用现有点云脚本生成
`evaluation/pcd/*.pcd` 和 `*_gt.pcd`；`ghost-removal` 调用现有评估脚本并保存
`evaluation/ghost_removal.txt`。`recall` 独立推理，保存 `evaluation/recall.json`，
包含全体有效体素和峰位置的分类指标及逐场景结果。

`ghost-removal` 会实时显示官方评估脚本的日志、进度条和错误，同时逐段刷新写入
`evaluation/ghost_removal.txt`（每次启动覆盖该文件）。子进程以无缓冲模式运行，
无需额外命令参数。该文件在运行期间是部分日志，只有命令成功结束并显示
`Saved report` 后才代表本次评估完成；失败或中断时保留已写入的日志并返回错误。
此修改不改变官方点云匹配算法，也不会更新已经启动的进程。

`recall`（包括 `all` 中的该阶段）保存 JSON 后，会自动独立导出
`evaluation/confusion_matrix/`：其中 `peak/`、`voxel/` 和
`peak_by_scene/<场景>/` 分别保存四分类原始计数、按行归一化的 CSV 与 PNG/SVG
热力图，以及四个类别各自的 one-vs-rest 2×2 矩阵。
行是真实类别，列是预测类别，类别顺序为 noise、object、glass、ghost；
归一化矩阵的对角线是各类别 recall，零 support 行显示为 0。
2×2 矩阵的行列顺序均为“其他类别、当前类别”，即 `[[TN, FP], [FN, TP]]`。
汇总计数和数据来源保存在 `confusion_matrix/matrices.json`。
绘图使用环境中的 matplotlib；若导出失败，已保存的 recall 结果仍保留，后续阶段继续执行，
终端会报告错误，可通过下述独立命令重试。

已有 `recall.json` 可以直接离线导出，无需 checkpoint、GPU、estimate 或重新推理：

```bash
uv run --no-sync python -m experiments.wavedsp.confusion_matrix \
  --recall-json output/wavedsp/fast_ghost/fast_16patch/evaluation/recall.json
```

默认写入该 JSON 同级的 `confusion_matrix/`，可用 `--output-dir` 指定其他目录。
导出不会修改源 JSON；重复运行会更新对应的导出文件。

入口按 checkpoint 的 `target_size: [400,332,256]` 做全幅推理，并检查测试配置
的裁边和时间抽样是否与训练一致。默认读取 `configs/config_estimate.yaml`、
`configs/config_test.yaml` 和原有点云配置。estimate 的低置信度标签沿用原脚本的
`-1`，recall 沿用原测试的 noise 类 `0`。默认逐帧推理，
CUDA 使用 BF16；需要 FP32 可传 `--precision fp32`。


## FAST/FINE 纯 reconstruction MAE 预训练

独立的 waveform-only 缓存、完整主干重建预训练、单卡/双卡启动和断点恢复命令见 [MAE.md](MAE.md)。
FAST/FINE 共用 `data/mae_dataset_cache_v1`，该软链接指向
`~/datasets/Ghost-fwl/mae_dataset_cache_v1`，缓存实际存储在数据集目录下。
不依赖 peak 文件。原有监督训练入口不变。

若运行目录在评估进程启动后被移动，旧进程仍使用启动时的路径，会出现
`FileNotFoundError`。请在移动完成后，使用新路径和 `--output-root` 单独续跑
`pcd`；该阶段会跳过已有的完整预测/GT 点云对，只处理缺失帧。完成后单独运行
`ghost-removal`。运行期间保持结果目录路径稳定。
