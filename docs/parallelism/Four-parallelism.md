## 第 1 章 四种 Parallelism

| 技术 | 全称 | 核心原理 | 主要解决的问题 |
| --- | --- | --- | --- |
| TP | Tensor Parallelism | 把同一层的权重矩阵切到多张 GPU | 单层计算和显存压力 |
| PP | Pipeline Parallelism | 不同 GPU 存不同 Transformer 层 | 整个模型权重放不下 |
| DP | Data Parallelism | 复制模型，分别处理不同请求 | 提高吞吐量 |
| EP | Expert Parallelism | 不同 GPU 存不同 Expert 权重 | MoE 专家权重太多 |
