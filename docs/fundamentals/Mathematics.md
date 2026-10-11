## 1. Arithmetic Intensity是什么，如何定义的？

!image.png

**运算强度**定义就是"每搬运一个字节，能做多少次浮点运算"

下面的例子是一次矩阵乘法（GEMM）的运算强度：

### **1.1为什么 FLOPs = 2MKN**

输出矩阵 Y， 一个点积需要：

- K 次乘法
- K-1 次加法

合计约 2K 次浮点运算（1 次乘法 + 1 次加法 = 一个 "乘加"，通常算作 2 FLOPs）。

输出矩阵 Y 共有 M x N 个元素，所以总浮点运算数为：

FLOPs=M×N×2K=2MKNFLOPs=*M*×*N*×2*K*=2*MKN*

### **1.2 为什么 bytes = b(MK + KN + MN)**

要完成这次 GEMM，最少需要从内存中：

- **读取** X：M * K 个元素
- **读取** W：K * N 个元素
- **写回** Y：M * N 个元素

每个元素占 b 字节（比如 FP32 是 4 字节，FP16/BF16 是 2 字节），所以总的内存搬运量是：

bytes=b(MK+KN+MN)bytes=*b*(*MK*+*KN*+*MN*)

（这里假设的是理想情况——每个元素只读/写一次，没有考虑 cache 未命中重复读取等实际硬件因素，是一个"下限"估计。）

### 1.3 用 Arithmetic Intensity 解释为什么 Batching 有用

LLM Inference 中一个非常重要的优化：Batching

随着 Batch Size 增加：

- GPU 的计算量增加。
- 但是模型权重读取量不需要同比增加。
- 每读取一个 Byte，GPU 能做更多计算。

这也是为什么 **Continuous Batching 是 vLLM、SGLang 等推理框架非常重要的优化技术**。

## 2. 推理和训练一次的计算量

推理：2ND （主要统计模型参数参与的矩阵乘法）

其中：

- **2**：每个参数参与计算时，近似需要一次乘法和一次加法
- **N**：模型参数数量
- **D**：处理的 token 数量

训练：每个 token 的训练计算量 = 前向2ND + 反向4ND ，则总计算量表示为6ND

## 3. Flops和FLOPS的区别

这是 AI Infra 面试中非常重要的区别。
|  | FLOPs | FLOPS |
| --- | --- | --- |
| 含义 | Floating Point Operations浮点运算次数 | 每秒浮点运算次数 |
| 衡量什么 | 计算任务有多大 | 硬件计算速度有多快 |
| 单位 | FLOPs | FLOPs/s |
| 举例 | 100 GFLOPs | 100 TFLOPS |
