# **第 4 章 Kernel**

## 1. 什么是kernel？

kernel就是一次在GPU上运行的函数。CPU告诉GPU：用这几千个线程，执行这个函数

## 2. 什么是fused？

fused是“融合”的意思，指把原本多个小步骤合并在一个kernel里完成。

举个MoE不融合的例子：几百个expert可能就要启动几百上千个kernel，每个kernel都有启动开销。

```python
kernel 1：取出expert 3的token
kernel 2：计算expert 3矩阵乘法
kernel 3：取出expert 7的token
kernel 4：计算expert 7矩阵乘法
...
```

fused MoE kernel只启动一次，在同一个kernel里同时处理所有expert：

```
一个kernel：
  GPU线程组0 → 计算expert 3的一部分token
  GPU线程组1 → 计算expert 7的一部分token
  GPU线程组2 → 计算expert 1的一部分token
  ...
```

## 3. 为什么用固定大小的输入到kernel？

GPU一般把任务拆成固定大小的小块，每块由一组线程处理。例如`block_size=64`：

```
block 0：处理64个token位置
block 1：处理64个token位置
block 2：处理64个token位置
```

固定大小有几个好处：

1. 每个线程组的工作量和形状一样，代码可以写得很高效。
2. GPU矩阵计算单元（Tensor Core）适合固定形状，例如16、32、64、128。
3. 每个block只需要知道一个expert ID。
