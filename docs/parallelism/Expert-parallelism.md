# 第 5 章 EP (Expert Parallelism)
## 1. EP 的完整工作流程

假设 Token A 在 GPU 0 上，Router 选择了 Expert 2 和 Expert 5。

流程如下：

1. GPU 0 先计算 Attention 和 Router。
2. Expert 2 在 GPU 0，本地计算。
3. Expert 5 在 GPU 1，因此将 Token A 的 hidden state 发送到 GPU 1。
4. GPU 1 使用 Expert 5 计算。
5. Expert 5 的输出返回 GPU 0。
6. GPU 0 按 Router 权重合并两个 Expert 的结果，继续后续计算。

这里有两个我们专门澄清过的知识点。

第一：Router 不是只存在于 GPU 0。

在这个简化配置中，两张 GPU 都可以运行 Router。哪个 GPU 当前持有 Token 的 hidden state，就可以在当地计算路由决策。

第二：通信传输的是 Token 的激活数据，不是每次传 Expert 权重。

Expert 权重提前加载并常驻对应 GPU 的显存。

## 2. EP 的 Load Balancing（负载均衡）

假设有 100 个 Tokens，其中 80 个都被路由到 GPU 0 上的 Expert 1。

那么：

- GPU 0 非常忙。
- GPU 1 可能相对空闲。

即使 GPU 1 有空闲算力，也不能随便处理 Expert 1 的 Tokens，因为它没有 Expert 1 的权重。

两种解决方式。

1. Expert Replication

把热门 Expert 复制到多张 GPU。

例如 Expert 1 同时放在 GPU 0 和 GPU 1。

那么 80 个 Tokens 可以分成两批，各自在一张 GPU 上计算。

代价是占用更多显存。

对比 TP：

- TP：两张 GPU 合作计算同一个 Expert 的一次运算。
- Expert Replication：两张 GPU 都拥有完整 Expert，各自处理不同 Tokens。

2. Dynamic Expert Placement / Rebalancing

根据运行中的负载情况，周期性调整 Expert 的放置。

> 系统怎么知道哪个 Expert 更忙？
> 

两个指标：

- Router 的路由统计：每个 Expert 被分配了多少 Tokens。
- 实际执行指标：每个 Expert 的执行时间、等待情况，以及对应 GPU 的负载。

系统可以据此识别热门 Experts，再决定是否需要复制或重新分布。
