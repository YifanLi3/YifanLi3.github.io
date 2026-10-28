# 第 3 章 PP (Pipeline Parallelism)
## 1. PP 到底在切什么？

PP = 按模型的层（Layer）切分，让不同 GPU 负责不同的 Transformer Layers。

和 TP 不同：

- TP：GPU 0 和 GPU 1 都参与第 1 层、第 2 层……的计算，只是各自负责层内不同部分。
- PP：GPU 0 完整计算第 1～4 层，GPU 1 完整计算第 5～8 层。

## 2. GPU 之间传的是什么？

传的是 Activation / Hidden States，而不是整层权重。

## 3. PP 下的 KV Cache 存在哪里？

这是 Inference 中非常重要的问题。先回顾：每个 Transformer Layer 都有自己的 KV Cache。假设模型有 8 层：

- Layer 1 有自己的 KV Cache。
- Layer 2 有自己的 KV Cache。
- ……
- Layer 8 也有自己的 KV Cache。

PP=2 后：

| GPU 0 | GPU 1 |
| --- | --- |
| Layer 1–4 的权重 | Layer 5–8 的权重 |
| Layer 1–4 的 KV Cache | Layer 5–8 的 KV Cache |

这和 TP 有很大区别：

- PP：KV Cache 通常按照 Layer 分布。每个 Stage 存储自己负责的 Layers 的 KV Cache。
- TP：KV Cache 可以在同一层内按 KV Heads 等维度切分，具体取决于 Attention 结构和实现。

所以 PP 不只是减少单 GPU 的权重显存，也可以减少每张 GPU 需要存储的 KV Cache 层数。但它不会自动让整个系统的 KV Cache 总量减少，只是把它分布到了不同 GPU 上。

## 4. 为什么 PP 会导致 GPU 闲置？

这其实是 PP 最大的缺点之一。

假设：

- GPU 0 负责 Layer 1–4，耗时 10ms。
- GPU 1 负责 Layer 5–8，耗时 10ms。
- 暂时忽略通信时间。

只有一个 Request A, 单个 Request 的总计算延迟就是20ms。虽然有两张 GPU，但没有让这个 Request 的两个 Stage 同时计算。因为 Layer 5 的输入依赖 Layer 4 的输出。这叫 Pipeline Dependency。所以 PP=2 不一定能降低单个 Request 的 Latency，这是一个很重要的结论。

PP 的主要价值往往是：

1. 让模型放得进多张 GPU。
2. 通过多个 Request 或 Microbatch 的流水线执行提高整体吞吐。

而不是单纯让单个 Request 的 Forward 快两倍。

## 5. Pipeline 为什么需要 Microbatch？

因为我们不希望 GPU 0 在计算完 Request A 后就一直闲着。当 GPU 0 把 A 交给 GPU 1 时，它可以开始计算 Request B。这就是流水线。这里提升的是 整体吞吐能力，不是每个 Request 的计算量减少了。

什么是 Microbatch？

Microbatch 就是把一批工作拆成较小的执行单元，让不同 Stage 能同时处理不同单元。

例如有 32 个 Requests：

可以划分为 4 个 Microbatches，每个 Microbatch 包含 8 个 Requests，GPU 0 处理 Microbatch 2 时，GPU 1 可以处理 Microbatch 1。

Inference 中也可以直接把不同 Request Batches 或 Token Groups 作为流水线调度单元，具体取决于 serving engine 的实现。

## 6. Pipeline Bubble 是什么？

## 7. PP 在 Prefill 和 Decode 阶段有什么不同？

## 8. PP 的 Stage 为什么不一定平均切？
