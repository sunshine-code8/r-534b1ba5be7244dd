# FastWaveDSP / FineWaveDSP 增强版历史实测

以下结果来自此前增强版的同轮测试。本次移除旧版实现和 `variant` 入口后，
**未重新运行测速**；这些数字不是本次修改后的新测量。

条件：NVIDIA L40S `cuda:7`，batch=1，输入 `336×400×256`，完整连续 logits 输出，
PyTorch 1.13.1+cu117，TF32 off，cuDNN autotune=False，warmup=5，iterations=20，repeats=5。
不含数据加载或 CPU/GPU 传输。

| 模型 | FP32 延迟 | FP32 FPS | FP16 延迟 | FP16 FPS |
| --- | ---: | ---: | ---: | ---: |
| FastWaveDSP | 9.924 ms | 100.77 | 6.039 ms | 165.58 |
| FineWaveDSP | 12.666 ms | 78.95 | 7.098 ms | 140.88 |

原始记录：[FP32 CSV](../../output/wavedsp/plus_comparison_fp32.csv)、
[FP16 CSV](../../output/wavedsp/plus_comparison_fp16.csv)。这两份 CSV 是当时对照实验的
历史文件，仍含已移除的配置行；当前测速脚本只会运行 FastWaveDSP 和 FineWaveDSP。
随机权重推理速度不能说明分类精度。
