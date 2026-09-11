# 原生 vLLM 的 LBHND 使用场景

## 对话一：LBHND 适合什么场景

用户：

我们再回到原生 vLLM，现在是 LBHND，D 就是 C。有哪些 backend，或者说哪些使用场景会更喜欢用这样的存储呢？比如对层做修改时？假设我要直接删除某一层，我就喜欢用这样。讲一下它的使用场景，擅长什么，不擅长什么。

助手：

`LBHND` 可以理解为：

```text
[Layer, Block, Head, Token, Dimension]
```

其中：

```text
L = layer
B = block
H = KV head
N = block 内 token
D/C = 每个 head 的 content 或 dimension
```

物理顺序是：

```text
先 Layer，再 Block，再 Head，再 Token，再 Dimension
```

因此同一层的所有 block 连续放在一起：

```text
Layer 0:
    Block 0
    Block 1
    Block 2
    ...

Layer 1:
    Block 0
    Block 1
    Block 2
    ...
```

这种布局适合按 layer 访问、复制、导出、迁移、压缩和量化。比如要把某一层的 KV 从 GPU 搬到 CPU，可以直接处理这一层的连续区域：

```text
GPU: Layer 5 = [Block 0][Block 1][Block 2][Block 3]
CPU: Layer 5 = [Block 0][Block 1][Block 2][Block 3]
```

如果要修改、替换或逻辑删除某一层，也容易找到该层对应的 view。vLLM 的 per-layer KV cache 接口和 offload worker 都是按 layer name 组织 cache 的，因此 layer-oriented layout 对这类操作比较自然。

但它不适合以 block 为中心、同时访问多个 layer 的算法。`LBHND` 下，同一个 physical block 在不同 layer 的数据相距较远：

```text
Layer 0 的 Block 7
Layer 1 的 Block 7
Layer 2 的 Block 7
```

这些地址并不挨在一起。如果算法要做跨层 KV 相似度、跨层 KV 共享、跨层压缩，或者希望一次 DMA 搬走同一个 block 的多层数据，更适合 block-first 的布局，例如 `BLHND`：

```text
Block 0:
    Layer 0
    Layer 1
    Layer 2

Block 1:
    Layer 0
    Layer 1
    Layer 2
```

因此可以这样判断：

```text
要拿一整层的很多 block：LBHND
要拿一个 block 的很多层：BLHND
```

原生 vLLM 中，CPU attention backend 偏好 `LBHNC`；FlashInfer 的某些路径支持 `LBHNC` 和 `BLHNC`；FlexAttention 偏好 `LBNHC`；MLA indexer 的某些路径支持 `BLHNC` 和 `BLNHC`。最终 layout 由 backend 支持列表和全局 resolve 逻辑共同决定，而不是单纯由用户随意选择。

需要注意，LBHND 让“按层找到和处理”容易，但不等于物理删除一层后 buffer 自动压缩。真正删除一层并压紧后续内存，仍然需要搬移后续数据、修改 offset，并重新绑定 view。

## 对话二：跨层算法与按层 offload

用户：

你说的是真实场景，但我想知道，有没有一种解决方案，比如某种算法做跨层怎么怎么，或者某种卸载方案做按层 offload。

助手：

有，而且可以分成两类：一类是系统按 layer 逐层处理、搬运或修改 cache；另一类是算法需要同一个 block 的多个 layer 同时参与。

### 按 layer offload KV cache

这是 `LBHND` 最直接的应用。vLLM 的 offload worker 接收 per-layer GPU KV caches。使用 `LBHND` 时，同一层的多个 block 连续放置，因此可以按 layer 做批量 DMA：

```python
for layer in layers:
    copy(gpu_layer_cache[layer][block_ids], cpu_layer_cache[layer][block_ids])
```

实际的管理粒度仍然可以是 block；`LBHND` 只是让这一层的 block 在地址上连续，便于批量搬运。

### KV connector 与 P/D disaggregation

在 prefill 节点生成 KV、decode 节点接收 KV 的场景中，可以按 layer 分批传输：

```text
Layer 0 的多个 block
Layer 1 的多个 block
Layer 2 的多个 block
```

这类 layer-oriented transfer 适合 `LBHND`。vLLM 的 cache view 会记录 layer offset、layer stride 和 block stride，系统可以据此计算某一层某一 block 的地址。

### 按 layer 做 KV 压缩或量化

例如：

```text
Layer 0 使用 FP16
Layer 1 使用 INT8
Layer 2 保持 FP16
```

或者浅层保留完整 KV、深层保留压缩 KV。按层组织时，可以独立读取、压缩、量化、替换和重新绑定某个 layer 的 view。

这和 vLLM 的 layerwise weight reload 思路相似，但要区分：vLLM 现成的 layerwise reload 主要针对模型权重；按层 KV 压缩或 offload 是同样受益于 `LBHND` 的系统设计。

### 跨层 KV 算法

标准 Transformer 是逐层执行的：

```text
Layer 0 → Layer 1 → Layer 2 → ...
```

普通 attention kernel 通常只需要当前 layer 的 KV，因此不需要跨层同时读取。

但如果算法需要：

```text
跨层 KV 相似度计算
跨层 KV 共享
跨层 KV 压缩
按 block 同时分析多个 layer
```

那么 `LBHND` 不理想，因为同一个 block 的不同 layer 相距较远。此时更适合 `BLHND` 或其他 block-first 布局。

最终判断方法是：

```text
操作对象是“一整层的很多 block”吗？
    是：偏向 LBHND

操作对象是“一个 block 的很多层”吗？
    是：偏向 BLHND
```
