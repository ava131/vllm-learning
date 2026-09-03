# vLLM KV Cache Layout Notes

## Enum

用户：

```python
class KVCacheLayout(Enum):
"""Physical layout descriptor for a KV cache group.
```makefile
The logical shape is always [L, B, H, N, <content>] (RFC #42082).
Each member's value is a stride permutation that maps logical axes
to physical (memory) order.
"""

LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)
LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]
LHBNC = (0, 2, 1, 3, 4)  # [L, H, B, N, C]
BLHNC = (1, 0, 2, 3, 4)  # [B, L, H, N, C]
BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]
BHLNC = (1, 2, 0, 3, 4)  # [B, H, L, N, C]
```
还有这种东西，什么时候用哪种？什么叫然后用不同的 stride/view，把它解释成“每层自己的 KV cache”
这句话到底怎么理解，怎么体现
```

助手：

对，就是这段。它其实在说一件很具体的事：

**同一块连续 HBM，可以用不同的“走路规则”解释。这个走路规则就是 stride。**

先把维度翻译成人话：

```text
L = layer
B = block
H = KV head
N = block 内 token/state
C = 每个 state 的内容 bytes
```

比如逻辑上我们想访问：

```text
layer = 2
block = 5
head = 3
token_in_block = 7
content_byte = 10
```

无论物理 layout 是什么，逻辑坐标都是：

```text
[L=2, B=5, H=3, N=7, C=10]
```

但它在 HBM 里的真实地址，要靠 stride 算：

```text
addr = base
     + L * stride_L
     + B * stride_B
     + H * stride_H
     + N * stride_N
     + C * stride_C
```

`KVCacheLayout` 决定的就是：

```text
stride_L / stride_B / stride_H / stride_N / stride_C 谁大谁小
```

**举个小例子**

假设：

```text
L = 3 层
B = 2 个 block
每个 page = 100 bytes
```

先不管 H/N/C 细节，只看 L 和 B。

如果是 `LBHNC`，也就是 L 在外，B 在里：

```text
物理内存：
L0B0 L0B1 | L1B0 L1B1 | L2B0 L2B1
```

那：

```text
layer_stride = 2 blocks * 100 = 200 bytes
block_stride = 100 bytes
```

访问 `L2B1`：

```text
addr = base + 2 * 200 + 1 * 100
```

如果是 `BLHNC`，也就是 B 在外，L 在里：

```text
物理内存：
B0: L0B0 L1B0 L2B0 | B1: L0B1 L1B1 L2B1
```

那：

```text
layer_stride = 100 bytes
block_stride = 3 layers * 100 = 300 bytes
```

访问 `L2B1`：

```text
addr = base + 2 * 100 + 1 * 300
```

所以同一个逻辑坐标 `L2B1`，物理地址算法不同。这就是 layout。

**什么叫 view？**

vLLM 不是给每层真的复制一份新 tensor。它是：

```text
先 malloc 一整块大 buffer
然后给 layer0 创建一个 view
然后给 layer1 创建一个 view
然后给 layer2 创建一个 view
```

这些 view 都指向同一块底层内存，只是：

```text
起点 offset 不同
stride 不同
shape 不同
```

所以“每层自己的 KV cache”不是说每层一定独立 malloc，而是说每层拿到一个 tensor 视角。

比如物理内存是 `BLHNC`：

```text
B0: L0 L1 L2 | B1: L0 L1 L2 | B2: L0 L1 L2
```

给 layer1 的 view 会长得像：

```text
layer1_view[0] -> B0 里的 L1 page
layer1_view[1] -> B1 里的 L1 page
layer1_view[2] -> B2 里的 L1 page
```

它看起来像：

```text
layer1_view = [L1B0, L1B1, L1B2]
```

但底层真实内存中，这几个 page 中间隔着别的 layer：

```text
raw buffer:
L0B0 [L1B0] L2B0 L0B1 [L1B1] L2B1 L0B2 [L1B2] L2B2
```

这个“隔着走”的距离就是 `block_stride`。

源码里对应的是这段：

```python
Layer layers[l]'s page for block b starts at
offset + l * layer_stride + b * block_stride
```

也就是：

```text
第 l 层、第 b 个 block 的地址 =
起点 + l * layer_stride + b * block_stride
```

这就是“用 stride/view 解释成每层自己的 KV cache”。

**什么时候用哪种？**

看 backend 支持和全局 resolve。

原生 vLLM 会先问每个 attention backend：

```text
你支持哪些 KVCacheLayout？
```

然后取一个全模型都能接受的 layout。

大致规律：

```text
L 开头：layer compact，每层 cache 更连续
B 开头：block outermost，每个 block 内聚合多层 page
```

具体例子：

```text
flex_attention: 只支持 LBNHC
CPU attention: 只支持 LBHNC
FlashInfer: 支持 LBHNC 和 BLHNC
MLA indexer: 支持 BLHNC 和 BLNHC
```

所以不是用户普通情况下手动选，而是 backend 报支持列表，vLLM 自动选。用户可以用环境变量强制：

```bash
VLLM_KV_CACHE_LAYOUT=BLHNC
```

但如果 backend 不支持，会报错。

**最关键的理解**

`BLHNC` 这种确实是：

```text
物理上：一个 block 里放多个 layer 的 KV page
```

但 layer 调 kernel 时看到的是：

```text
自己这一层的 KV cache view
```

所以两句话同时成立：

```text
物理 HBM 可以是 block-major，block 里包含多层。
代码接口仍然是 per-layer view，每层拿自己的 kv_cache tensor。
```

这就是原生 vLLM 这套 layout 设计的核心。

## Page

用户：

```text
每个 page = 100 bytes
```

一个page指的是啥呢？

助手：

这里的 **page** 可以先粗略理解成：

```text
某一层、某一个 block 的 KV 数据块
```

也就是：

```text
page = layer L 的 block B 里，所有 KV head、block 内所有 token/state、所有 content 的那一坨数据
```

对应逻辑形状：

```text
[H, N, C]
```

因为完整逻辑形状是：

```text
[L, B, H, N, C]
```

固定了 `L` 和 `B` 以后，剩下的就是：

```text
[H, N, C]
```

这一块就叫一个 page。

举例：

```text
L = 第 3 层
B = 第 7 个 block
```

那么：

```text
page(L=3, B=7) = 第 3 层第 7 个 block 的全部 KV 内容
```

如果是普通 attention，里面大概包含：

```text
block 内所有 token 的 K
block 内所有 token 的 V
所有 KV heads
每个 head 的 head_dim
```

所以它不是“一个 token”，而是：

```text
一个 layer 的一个 block
```

更具体一点，普通 KV cache 可以想成：

```text
page_size_bytes =
block_size * num_kv_heads * 每个 token 每个 head 的 KV 数据大小
```

如果 K/V 合在同一个 state/content 里，page 里会同时含 K 和 V；如果 backend 拆 K/V，概念上还是“这个 layer 这个 block 的 KV page”，只是实现上可能拆成 K page 和 V page/view。

所以我前面说：

```text
每个 page = 100 bytes
```

只是为了举例，真实代码里 page 大小来自 `spec.page_size_bytes`。

最短版：

```text
block_id 是调度单位；
page 是某个 layer 对应这个 block_id 的实际 KV 存储内容。
```

所以：

```text
block B
  layer0 page
  layer1 page
  layer2 page
```

这里每个 `layerX page` 就是“该 layer 的 block B 的 KV 数据”。

## BHLNC Page

用户：

这样，你先把上面两轮对话，一次不差的写进一个md里，从那个enum类开始。
写完之后，我还有一个疑问，那如果是BHLNC,这个的page又该怎么确定呢？

助手：

写好了：[vllm_kv_cache_layout_notes.md](/Users/Admin/Documents/Codex/2026-09-03/lai/outputs/vllm_kv_cache_layout_notes.md)

关于 `BHLNC` 的 page：**page 仍然是固定 `L` 和 `B` 后的那一块 KV 内容**，也就是逻辑上的 `[H, N, C]`。这个概念不因为物理 layout 变成 `BHLNC` 就改变。

区别在于，`BHLNC = [B, H, L, N, C]` 的物理顺序是：

```text
先按 block
  再按 head
    再按 layer
      再按 token/state
        再按 content
```

所以一个 block 内部不是：

```text
B0: L0_page L1_page L2_page
```

而更像：

```text
B0:
  H0: L0 的 H0 部分, L1 的 H0 部分, L2 的 H0 部分
  H1: L0 的 H1 部分, L1 的 H1 部分, L2 的 H1 部分
  H2: L0 的 H2 部分, L1 的 H2 部分, L2 的 H2 部分
```

也就是说，在 `BHLNC` 里，**一个 layer 的 page `[H,N,C]` 在物理上可能不是一整段连续内存**。它被按 head 切开，散在 block 内不同 head 区域里。

例子：

```text
逻辑 page = page(L1, B0) = L1B0 的所有 H/N/C
```

在 `BLHNC` 里它大概连续：

```text
B0: [L0 all heads][L1 all heads][L2 all heads]
```

`L1B0 page` 是一整坨。

但在 `BHLNC` 里：

```text
B0:
  H0: [L0H0][L1H0][L2H0]
  H1: [L0H1][L1H1][L2H1]
```

`L1B0 page` 等于：

```text
[L1H0] + [L1H1] + ...
```

这些片段逻辑上组成一个 page，但物理上中间夹着别的 layer 的 head 数据。

所以最关键的一句是：

```text
page 的逻辑定义不变：固定 L 和 B 后的 [H,N,C]。
layout 只决定这个 page 在 HBM 里是不是连续、怎么跨 stride 走。
```
