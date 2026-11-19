# **第 3 章 Profilling**

## 1. Benchmark和Profilling的关系

Benchmark：描述的是请求侧的表现，可以把 benchmark 看成黑盒测量

profilling：Profiling 则记录程序内部的执行，CPU 在运行哪些 operator、GPU 执行了哪些 kernel、时间花在哪里，可以把profiling 看成打开内部过程的白盒观察。

两者要配合：先用指标明确问题，再用内部记录检验原因，修改后回到 benchmark 看结果。

## 2. Profilling

值得关注的其实就是三件事情：**耗时长的事件、频繁重复的事件、事件之间的空隙**。

**明确症状 → 提出假设 → 用 trace 检查 → 修改并验证**：

1. **明确症状。** 固定模型、硬件、版本、输入输出长度和并发，看慢的是 TTFT、TPOT，还是吞吐；先排除负载变化造成的差异。
2. **提出可以检查的假设。** 
    1. TTFT 高，就检查排队和 prefill；
    2. TPOT 抖，就检查 decode 是否被 prefill 插入或其他工作阻塞；
    3. 吞吐上不去，就检查实际 batch、并发和计算 / 通信等待。
    4. 指标决定从哪里查，但本身不是原因的证明
3. **验证假设和改动。** 在相同 rank、相同执行阶段采集短 trace，把可疑事件对应到代码。每次集中改一个问题，确认目标开销减少，再关闭 profiler，重复跑同条件 benchmark，并补正确性检查。
