Arithmetic intensity (AI) is FLOPs per byte moved from memory. A kernel with low AI is memory-bound; high AI is compute-bound.

下面的例子是一次矩阵乘法（GEMM）的运算强度：

**1. 为什么 FLOPs = 2MKN**

输出矩阵 Y， 一个点积需要：

- K 次乘法
- K-1 次加法

合计约 2K 次浮点运算（1 次乘法 + 1 次加法 = 一个 "乘加"，通常算作 2 FLOPs）。

输出矩阵 Y 共有 M x N 个元素，所以总浮点运算数为：

FLOPs=M×N×2K=2MKNFLOPs=*M*×*N*×2*K*=2*MKN*

**2. 为什么 bytes = b(MK + KN + MN)**

要完成这次 GEMM，最少需要从内存中：

- **读取** $X$：$M \times K$ 个元素
- **读取** $W$：$K \times N$ 个元素
- **写回** $Y$：$M \times N$ 个元素

每个元素占 $b$ 字节（比如 FP32 是 4 字节，FP16/BF16 是 2 字节），所以总的内存搬运量是：

bytes=b(MK+KN+MN)bytes=*b*(*MK*+*KN*+*MN*)

（这里假设的是理想情况——每个元素只读/写一次，没有考虑 cache 未命中重复读取等实际硬件因素，是一个"下限"估计。）
