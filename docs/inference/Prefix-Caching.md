# 第 4 章 Prefix Caching and Radix Attention

## 1. 为什么需要Prefix Caching？

传统的KV Cache有局限性，它的生命周期和单个请求严格绑定，一旦请求结束，KV Cache就全部释放掉了。没有解决跨请求重复计算的问题，比如在线LLM推理中，大量请求往往共享system prompt，如果每个请求都独立计算一遍完整的prefill，就会造成大量冗余。

所以prefix caching存在的意义就是改变kv cache的所有权，不在把kv cache绑到某一个request，而是归属到可复用的前缀序列上，即便一个请求结束，对应的kv cache可以继续保留在共享换缓存池中，由淘汰策略决定何时释放

## 2. Radix Attention的核心原理

Radix Attention的核心数据结构是Radix Tree，将各请求中的token id序列组织成前缀树，相同prefix仅对应一份KV Cache

### 2.1 基本组成

```python
class RadixTreeNode:
    def __init__(self, key_fn: KEY_FN) -> None:
        self._parent = None
        self.children = {}       # 子节点字典
        self.key_fn = key_fn     # 从 key tensor 提取“子节点序列索引”的函数
        self.ref_count = 0       # 引用计数
        self.uuid = counter
        self.timestamp = tic or time.monotonic()  # 节点访问时间戳

        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
```

### **2.2 Radix Tree 与 KV Cache Pool 的联系**

| **对象** | **保存内容** | **所在位置与作用** |
| --- | --- | --- |
| Radix Tree 的 key | token id 序列 | 逻辑索引，用于判断两个请求是否共享前缀 |
| Radix Tree 的 value | token slot / page indices | 地址索引，指向 KV Cache Pool 中对应的存储位置 |
| KV Cache Pool | 每一层的 K、V 张量 | GPU 上真正参与 Attention 计算的数据 |

### 2.3 **从到达到请求结束的缓存生命周期**

假设 Radix Tree 结构已稳定，历史请求已在 Radix Tree 中形成共享前缀 ，且每个 token id 对应一个物理槽位。后续过程主要参考 mini-SGLang 的实现（辅助参考 SGLang 的部分代码）：s = [900, 10, 11, 12]，需要处理的两个请求R1和R2。

R1 = [900, 10, 11, 12, 20, 21]
R2 = [900, 10, 11, 12, 41, 42, 43]

**Step1 Match 产生候选信息**。

R1/R2 的匹配结果包含 `cache_hit_len = 4`、节点 handle 以及物理索引 [p0, p1, p2, p3]。此时 R1/R2 尚未对这些页加锁，这些路径理论上仍可能被淘汰，因此“匹配到”不等于“已经安全持有”。

**Step2 Schedule 决定谁先运行**。

在 mini-sglang 中，调度按 pending_list 的到达顺序依次尝试添加请求，直到 token budget 或资源不足为止，没有 prefix-aware 的复杂策略。

而在完整 SGLang 中，调度器会比较命中长度、后缀长度与资源需求。SGLang 的调度策略分为两类：

- **前缀感知**：lpm、dfs-weight、hrrn、shortest-prefill-first 等，优先长共享前缀、按树 DFS 权重、或让更短未缓存工作先跑；
- **不感知前缀缓存**：fcfs（先到先服务）、lof（最长输出优先）、random、routing-key 等。

这里假设采用 shortest-prefill-first，两者共享前缀相同，R1 未命中后缀更短（只需两个新槽位），于是优先选中处理 R1，R2 则是留在等待队列。调度阶段还可能做 in-batch 前缀检查，避免同一 batch 内重复计算相近前缀。

需要注意的是，Scheduler 先选 R1，再选 R2，只表示它先后把两个请求加入 Batch。最终 GPU 可以一次处理整个 Batch。

> 那如果都是变成一个batch送入gpu去做forward，各种调度策略如 shortest-prefill-first还有什么用呢？
> 

GPU 的资源有限，不是所有请求都能进入同一轮 Forward，假如**GPU Token Budget 不够，就需要**调度策略。Scheduling 不是在决定 GPU 如何逐个执行 Batch 里面的请求，而是在决定**每一轮有限的 GPU 计算和 KV Cache 资源，应该分配给哪些请求，以及每个请求分配多少。**

**Step3 Lock 把候选变成受保护的引用**。R1 进入批次计算前，对匹配节点调用 lock 相关函数：

- **从该节点向根逐级 ref_count += 1** ；
- 对应 token id 序列从 `evictable_size` 移入 `protected_size`，并从可选淘汰叶子集合中更新状态。

共享前缀 [900, 10, 11, 12] 因此受到保护。命中前缀的索引写入请求的 page table，*但加锁本身只负责保护，不负责分配新的 page* 。

**Step4 Allocate 为未命中部分只申请缺口**。R1 的未命中后缀长度为 2，于是从 **free list** 分配 [p6, p7]，page table 变为 [p0, p1, p2, p3, p6, p7]。若空闲不足，分配器会先请求 CacheManager ，拿到一定量的释放物理空间后再继续分配。

**Step5     Insert 把新计算结果变成共享缓存**。Prefill 完成后，用 R1 的 token id 序列与 page table 更新树。已有的S节点继续复用，只在其下新增：[20, 21] —> [p6, p7]

**Step6 Unlock 与请求结束处理**。完成 R1 的计算后，对匹配路径调用 unlock 的相关函数（ref_count -= 1）：

- 若某节点 ref_count 降为 0 ，则它直接变为可淘汰状态（进入 evictable_size）；
- 若 ref_count > 0，说明还有其他请求在用，不能被淘汰。

此时，对应的前缀部分引用次数变成 0 ，也不会被立即从共享前缀树中摘除以及释放相应的物理槽位。

**Step7 空间不足时的 evict 与请求结束后的 free**

- **evict**（空间不足时）：当后续请求（如 R2）分配时发现 free 不够，调用 `prefix_cache.evict`。它只会挑选当前引用次数为 0 的叶子节点，**按策略从树中摘除**，并把对应 page 索引物理空间归还。（例如此时若 [31,32] 节点已无引用，就可能被选中，[p6, p7]对应就会被释放）
- **free**（无策略归还）
