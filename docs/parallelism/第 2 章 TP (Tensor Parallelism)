# 第 2 章 TP (Tensor Parallelism)

## 1.1 TP 到底是什么？为什么需要它？

TP = Tensor Parallelism，张量并行。

它的核心思想是：

把同一层中的权重矩阵切成几部分，放在不同 GPU 上，让多张 GPU 合作完成这一层的计算。

> 如果模型有 100 层，每层都要切开，那岂不是需要 200 张 GPU？
> 

答案是：不需要。

假设模型有 100GB 权重，两张 GPU，每张只有 80GB 显存。

TP=2 时：

- GPU 0：存第 1～100 层各自的一部分权重，合计约 50GB。
- GPU 1：存第 1～100 层剩下的权重，合计约 50GB。

两张 GPU 一起计算第 1 层，然后一起计算第 2 层，一直到第 100 层。因此每层计算过程中可能需要 GPU 之间通信。

例如，一个 Linear Layer 的权重可以是矩阵：

$$
W=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8
\end{bmatrix}
$$

它可以沿着不同维度切分。

这就进入了我们讨论最多的 横切和竖切。

## 1.2 TP 的横切和竖切：Row Parallel vs Column Parallel

Column Parallel：按列切

每张 GPU 负责不同的输出维度，因此各自产生不同位置的结果。如果需要完整输出，就用 All-Gather 拼接。

Row Parallel：按行切

每张 GPU 负责同一组输出的部分计算，因此必须把部分结果求和。如果所有 GPU 都需要完整结果，就用 All-Reduce。

注意：这是基于Y = XW 的矩阵方向约定。PyTorch Linear 的权重存储通常是 `[out_features, in_features]`，所以不能只凭画面上的“横切/竖切”判断，要看切的是输入维度还是输出维度。

举例假设：

$$
Y=XW
$$

$$
X=[1,2]
$$

$$
W=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8
\end{bmatrix}
$$

正常计算：

$$
X=[11,14，17，20]
$$

现在让 TP=2，两张 GPU 合作计算。

按列切：

!image.png

结果：两张卡得到不同位置的输出。

如果下一步需要完整 \(Y\)，就把两部分拼起来：

`[11,14] + [17,20]` → `[11,14,17,20]`

这里的 `+` 是拼接，不是数值相加。可以使用 All-Gather。

按行切：

!image.png

结果：两张卡算的是同一组输出位置的不同贡献。

需要逐元素求和：

`[1,2,3,4] + [10,12,14,16] = [11,14,17,20]`

如果两张卡都需要完整结果，可以使用 All-Reduce。

## 1.3 All-Reduce 究竟发生在哪张 GPU 上？

GPU 0 算了一半，GPU 1 算了一半。All-Reduce 是在哪张卡上完成的？是不是先合起来，再重新分给两张卡？

不是。All-Reduce 是多张 GPU 共同参与的通信操作。

例如：

GPU 0：A=[1,2,3,4]

GPU 1： B=[10,12,14,16]

All-Reduce（Sum）完成后：

GPU 0：[11,14,17,20]

GPU 1：[11,14,17,20]

两张 GPU 最后都拥有完整结果。

不需要先把所有数据集中到 GPU 0，然后再从 GPU 0 分发出去。

实际可以通过 Ring、Tree 等不同算法完成分布式通信，具体由 NCCL 等通信库和硬件环境决定。

> 结果合起来以后，下一层不是又要切开吗？
> 

这里要区分两件事：

- 权重切片：已经提前存放在不同 GPU 上，不需要每一层重新切权重。
- 计算结果：All-Reduce 后，各 GPU 已经拿到下一步需要的完整结果。

## 1.4 为什么 Transformer 里要交替使用 Column Parallel 和 Row Parallel？

用 Transformer 的 MLP 来解释。

一个典型的 MLP 包含两个 Linear 运算，中间有激活函数。以简化的非门控 MLP 为例：第一层 Column Parallel 的输出，刚好就是第二层 Row Parallel 需要的输入分片。

#### 1.1.5 TP 的优点、缺点，以及为什么不全部使用 TP

TP 的好处

- 可以把大矩阵分散到多张 GPU，降低单卡权重显存需求。
- 多张 GPU 可以同时参与同一层计算。
- 在计算规模和通信条件合适时，可以降低单个请求的计算延迟。

TP 的代价

需要频繁跨 GPU 通信。

例如一个 Transformer 有 100 层，每层 Attention 和 MLP 中可能涉及多次 collective communication。

如果 GPU 之间通信很慢，就可能出现：

- GPU 计算完成后等待其他 GPU。
- All-Reduce 占用大量时间。
- GPU 数量增加，但性能没有同比提升。

所以 TP 通常更适合 GPU 之间有高速互联的环境，例如 NVLink / NVSwitch。

> 为什么有时选择 PP？
> 

PP 可以把不同层放到不同 GPU 或 GPU 组，主要在 stage 边界传输 hidden states。

它通常减少了 TP 那种层内频繁的 collective communication，但代价是流水线等待、负载不均衡，以及单请求不能同时使用所有 stage 计算同一层。

因此两者经常组合使用。
