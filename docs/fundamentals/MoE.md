# **第 4 章** MoE

## 1. MoE做什么？

要计算的就是每个expert的FFN。

普通Transformer中，每个token都经过同一个FFN：

```
x → Linear → 激活函数 → Linear → y
```

MoE层中有很多个FFN，每个FFN叫一个expert。router会为每个token选择其中几个expert，例如`top_k=2`：

```
token 0 → expert 3、expert 7
token 1 → expert 7、expert 1
token 2 → expert 3、expert 5
```

每个expert都有自己的一套权重矩阵：

```
expert 1: W1_up, W1_down
expert 3: W3_up, W3_down
expert 7: W7_up, W7_down
```

所以`token 0`需要计算：

```
token0 × expert3权重
token0 × expert7权重
```

## 2. MoE是如何在gpu里计算的

最简单的写法是每个token单独计算：

```
token0 × W3
token2 × W3
token0 × W7
token1 × W7
```

每次都要从显存读取一次expert权重。一个expert矩阵可能有几十MB，但一次只算1个token，计算量很小，GPU大部分时间都在等数据读入。这就像为了运一个箱子，开一辆大卡车往返一次。

更好的方法是把选中同一个expert的token放在一起：

```
[token0, token2] × W3
[token0, token1] × W7
```

W3只需要读取一次，就可以同时计算多个token。GPU有几千个计算核心，适合批量做矩阵乘法：

```
[多个token的输入] × [一个expert权重]
```
