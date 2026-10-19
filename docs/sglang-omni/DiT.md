# 第 1 章 DiT

## 1. DiT是什么？

DiT 是 Diffusion Transformer 的缩写，指"用 Transformer 当主干网络的扩散模型"。它来自 2023 年的论文 *Scalable Diffusion Models with Transformers*，Sora 和 Stable Diffusion 3 这类模型都是这个结构。

### 1.1 扩散模型（或 flow matching）

是怎么生成东西的：

1. 从一团随机噪声开始。
2. 用一个神经网络预测"这团噪声应该往哪个方向改"。
3. 沿着这个方向走一小步，再预测、再走一步。重复 N 步，噪声就逐渐变成想要的东西。

这个负责做预测的网络就叫 estimator（估计器）。早期 estimator 多用 U-Net（卷积结构），DiT 把它换成了 Transformer。

放到sglang-omni的 TTS 里： CosyVoice3 和 MiniCPM-o 的 vocoder 都是这样一条流水线：

```
语音 token → Flow（DiT 做 N 步去噪）→ mel 频谱 → HiFT → 波形
```

- Flow 就是上面说的 flow matching 生成过程。MiniCPM-o 默认 `n_timesteps=10`，也就是走 10 步。
- DiT 是 Flow 里的 estimator，每一步都要把整个 DiT 跑一遍。再加上 CFG（classifier-free guidance，一种让生成结果更贴合条件的技巧），batch 还要翻倍。

### 1.2 “每一步都要把整个 DiT 跑一遍"什么意思

Flow 生成 mel 的过程，就像从一团噪声出发，沿着一条路一小步一小步地走到目标：

```
噪声 x₀ ──走一步──> x₁ ──走一步──> x₂ ── … ──> x₁₀ = 最终的 mel
```

每走一步之前，都要先问 DiT："在当前位置，我应该往哪个方向走、走多快？"DiT 的回答就是代码里的 `derivative`（导数，也就是速度场）。拿到方向后，往前走一小步：

\[
x_{\text{new}} = x + dt \times v
\]

各部分的意思是：

- **\(x\)**：更新前的数值。在这里，是当前正在生成的声学特征。
- **\(v\)**：模型预测的**变化速度**，包含变化的方向和大小。我之前只写“方向”，不够准确。
- **\(dt\)**：这一步前进多少，也叫步长。
- **\(x_{\text{new}}\)**：更新后的数值，作为下一轮计算的输入。

例如，当前 \(x=0.2\)，模型预测 \(v=3\)，步长 \(dt=0.1\)，那么：

\[
x_{\text{new}}=0.2+0.1\times3=0.5
\]

这种"算出方向、走一小步、再算方向、再走一步"的数值方法，叫 Euler 法，所以函数名是 `solve_euler`。对应的代码：

```python
for step in range(1, len(t_span)):
    paired_sample = torch.cat([x, x], dim=0)
    paired_timesteps = torch.cat([t, t], dim=0)
    if states is None:
        conditional_derivative = self.estimator.forward(
            paired_sample,
            paired_mask,
            paired_mu,
            paired_timesteps,
            paired_speaker_embeddings,
            paired_mel_conditioning,
        )
    else:
        // ... 流式分支省略 ...
    conditional_derivative, unconditional_derivative = torch.split(
        conditional_derivative, [x.size(0), x.size(0)], dim=0
    )
    guided_derivative = (
        (1.0 + self.inference_cfg_rate) * conditional_derivative
        - self.inference_cfg_rate * unconditional_derivative
    )
    x = x + dt * guided_derivative
    t = t + dt
```

其中 `self.estimator` 就是 DiT。MiniCPM-o 默认 `n_timesteps=10`，所以这个 `for` 循环会跑 10 次，DiT 被完整调用 10 次。而每次调用都要把所有 block（每个 block 里有 attention、卷积、MLP）全部过一遍。

所以生成一段音频的 DiT 计算量，大约是"DiT 跑一次的量 × 10"。这就是 DiT 成为 vocoder 里最耗时部分的原因。

### 1.3 CFG 是干什么的

DiT 预测方向时会参考一些"条件"：

- `mu`：从语音 token 编码出来的内容信息，即"要说什么"；
- `speaker_embeddings`：说话人特征，即"用谁的声音"；
- `mel_conditioning`：参考音频的 mel，即"参照什么音色和语气"。

CFG（classifier-free guidance）的做法是同时算两个方向：

- 有条件的方向： 把这些条件都给 DiT，得到"按要求应该往哪走"；
- 无条件的方向： 把条件全换成 0，得到"什么都不知道时会往哪走"。

然后把两者的差异放大：

最终方向=(1+w)×有条件−w×无条件\text{最终方向} = (1 + w) \times \text{有条件} - w \times \text{无条件}

最终方向=(1+w)×有条件−w×无条件

这里的 www 就是代码里的 `inference_cfg_rate`（0.7）。直观地说，就是"往'按要求'的方向多推一点，离'随便生成'远一点"，这样生成的语音更贴合内容和说话人。

为什么是 batch 翻倍？

两个方向都要跑 DiT，最直接的做法是每一步调用两次 DiT。但每次调用都有固定开销，更好的办法是把两份输入拼成一个 batch，只调用一次。代码就是这样做的。

循环开始前，先把条件拼成"真条件 + 全零条件"两份：

```python
paired_mask = torch.cat([mask, mask], dim=0)
paired_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
paired_speaker_embeddings = torch.cat(
    [speaker_embeddings, torch.zeros_like(speaker_embeddings)], dim=0
)
paired_mel_conditioning = torch.cat(
    [mel_conditioning, torch.zeros_like(mel_conditioning)], dim=0
)
```

每一步里，把当前位置 `x` 也复制成两份：`torch.cat([x, x], dim=0)`。

举个例子，假设一次同时处理 8 条音频：

```
原本 batch:  [音频1, 音频2, …, 音频8]                       → 8 行
CFG 之后:    [音频1~8（真条件）, 音频1~8（全零条件）]          → 16 行
```

DiT 一次算完这 16 行，再用 `torch.split` 切回前 8 行（有条件的结果）和后 8 行（无条件的结果），按上面的公式合成最终方向。

所以 DiT 是 vocoder 里最耗计算的部分，有几个 PR 都在想办法让它跑得更快：

- #2424：让 Inductor 能融合 DiT 里的 LayerNorm；
- #2516：给流式 DiT 用 CUDA Graph；
- #2397：给 MiniCPM-o 的 DiT 用 FP16 加 compile。

### 1.4 MiniCPM-o 的 DiT 长什么样

它由 depth 个结构相同的 `DiTBlock` 堆叠而成（默认值是 28，实际层数以 checkpoint 的配置为准）。每个 block 里按顺序是：

- LayerNorm 和 AdaLN 调制：AdaLN 会根据"当前是第几步"来调整归一化后的数值
- attention
- 因果卷积
- MLP

前面说的"给每个 block 单独 compile"，就是对这些 `DiTBlock` 逐个做 compile。

## 2. Dense DiT and Packed DiT

一个 batch 里的多条音频长度不一样时，有两种组织数据的方式。

dense（补齐）： 把所有序列都 pad 到最长的那条，排成一个规整的矩形张量。

```
请求A: ■■■■■■■■■■      (长度 10)
请求B: ■■■■□□□□□□      (长度 4，补 6 个空位)
请求C: ■■■■■■■□□□      (长度 7，补 3 个空位)
→ 形状 [3, 10, C]
```

attention 时再传一个 mask 告诉模型哪些位置是 padding。优点是形状规整，好写，也好编译。缺点是 padding 部分也在白白参与计算。

packed（拼接）： 把有效帧首尾相连拼成一条长序列，再额外记录每条序列从哪里开始、到哪里结束。

```
■■■■■■■■■■|■■■■|■■■■■■■   → 形状 [21, C]，并记录分界点 [0, 10, 14, 21]
```

attention 用专门的变长 kernel（`varlen_attn`）来算。优点是没有浪费的计算。缺点是代码更复杂，形状依赖数据本身（每次长度都不同），编译起来更麻烦。
