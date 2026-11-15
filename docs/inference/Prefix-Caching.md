# 第 4 章 Prefix Caching and Radix Attention
## 1. 为什么需要Prefix Caching？

传统的KV Cache有局限性，它的生命周期和单个请求严格绑定，一旦请求结束，KV Cache就全部释放掉了。没有解决跨请求重复计算的问题，比如在线LLM推理中，大量请求往往共享system prompt，如果每个请求都独立计算一遍完整的prefill，就会造成大量冗余。

所以prefix caching存在的意义就是改变kv cache的所有权，不在把kv cache绑到某一个request，而是归属到可复用的前缀序列上，即便一个请求结束，对应的kv cache可以继续保留在共享换缓存池中，由淘汰策略决定何时释放

## 2 Radix Attention的核心原理

Radix Attention的核心数据结构是Radix Tree，将各请求中的token id序列组织成前缀树，相同prefix仅对应一份KV Cache

### 2.1 基本组成
