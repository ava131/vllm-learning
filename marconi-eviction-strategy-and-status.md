# Marconi 的淘汰策略：机制、例子与实现现状

> 记录日期：2026-09-20
> 范围：只讲 Marconi 的第二部分——**淘汰（缓存满了扔谁）**。包括机制、具体例子、以及 vLLM / vllm-ascend 的实现现状与对应 PR。

---

## 0. 一句话

> **不扔"最久没碰的"，扔"占地方大、又没帮我省下多少计算"的。**
> 判的对象不是 block，而是**前缀树上的一个条目**；判的依据是"新旧 + α × 性价比"；淘汰时删分数最低的，删到够用为止。

---

## 1. 机制：四步

### 第 1 步：计分单位不是 block，是"树上的条目"

```
节点 = 分叉点（前缀树里序列分开的地方）
边   = 相邻两个节点之间的一段 token
条目 = 这条边的 KV ＋ 挂在它末端节点上的 SSM 状态
```

- 论文里的树：每条边带这段 token 的 KV；SSM 状态挂在边上/节点上；
- **不是固定大小的 block**，而是一段**长度可变**的 token；
- 不是每个节点都带状态：只有被"准入"过的节点才有（共享前缀分叉点 + 序列尾巴）。

### 第 2 步：算"性价比"

```
flop_eff(n) = 这个条目能省下的计算量 / 这个条目占的内存
```

| | 内容 |
|---|---|
| 分子 | 复用这条边能省下的计算（Attention + SSM + MLP 全算），**且只算"相对父节点多出来的那一段"** |
| 分母 | 这条边的 KV + 它带的 SSM 状态（固定大小） |

**"相对父节点"必须有**：否则父节点已经能省的那段会被重复计入，深层节点全被高估。

### 第 3 步：得到最终分数

```
S(n) = recency(n) + α · flop_eff(n)
```

- 两个量都**在整棵树的所有节点上做 min-max 归一化到 0~1**（一个是时间、一个是"算力/字节"，量纲不同）；
- `α = 0` → 退化成纯 LRU；
- **淘汰 = 反复挑分数最低的删掉，直到腾够空间**（按需，不搞全量大清洗）。

### 第 4 步：α 自动调

1. 启动时 α = 0（先按 LRU 跑）；
2. 第一次发生淘汰后拍快照，进入**观察期**（继续用 LRU，但记录请求）；
3. 观察期长度 ≈ 第一次淘汰前见过的请求数的 **5~15 倍**；
4. 后台**回放**这些请求，对多组 α 做网格搜索（多 CPU 核并行，几秒出结果）；
5. 选命中率最高的 α，之后固定使用。

### 两个容易忽略的实现细节

- **不只淘汰叶子**：只有 1 个孩子的中间节点也进候选（它代表"只被走过一次的中间路径"，不会再被复用，却白占一份状态）。
- **命中时不刷新祖先节点的时间戳**（传统系统会刷新）：祖先的状态不会被复用、KV 也会被子节点吸收，刷新只会让该走的不走。

---

## 2. 为什么"长前缀的缓存更值钱"（这是整个策略的地基）

| | 内存 | 复用它省下的算力 | 每字节省下的算力 |
|---|---|---|---|
| Attention KV（复印件） | ∝ 长度（`2·L·D·2` 字节） | ∝ 长度 | **基本恒定** |
| SSM 状态（笔记本） | **固定**（`D·N·2` 字节，与 L 无关） | ∝ 长度 | **∝ 长度** |

关键数字：**一个 SSM 状态 ≈ 单个 token KV 的 10~100 倍**。block size 16 时，一个 SSM 层的状态约是 attention 层 token block 里 KV 的 4 倍。

于是推出：

> **SSM 状态是一笔"贵的、固定的开销"。它划不划算，取决于这笔开销被摊在了多长的前缀上。**

由此得到三类条目的性价比排序：

| 条目 | 性价比 |
|---|---|
| 深位置的 checkpoint（花一份状态钱，能跳过一大段） | **最高** |
| 普通 KV 块（不带状态） | **基准线（恒定）** |
| 紧贴父节点下面的浅 checkpoint（多跳过几十个 token，却占一整份状态） | **最低** |

**所以结论不是"优先存 SSM"，而是"SSM 状态要惜着存、挑着留"**：
- 准入（论文第一部分）= 只在该花的地方花（分叉点 + 尾巴）；
- 淘汰（论文第二部分）= 已经花出去的多笔开销，先退掉摊得最薄的那笔。

---

## 3. 为什么 LRU 不够：一维 vs 二维

**先解开一个概念结：LRU 的"链表位置"本来就是分数。**

- 每次访问把条目移到链表尾部；淘汰时从头部取——等价于"分数 = 最近一次访问时间"，尾部分数最高，头部分数最低；
- 它能用链表实现，是因为只有**一个维度**，而且"刚被访问"这个事件天然告诉你往哪儿挪（O(1)）。

**一旦加第二个维度，链表就表达不了了**：一个条目可能"很新但不值钱"，也可能"很旧但很值钱"，链表排不出这种序。所以必须改成"给每个条目算一个数，挑最小的扔"。

> 顺带一提：ARC / 2Q / T-LRU 这些策略是"**用多个链表去近似第二个维度（频次）**"；Marconi 是直接算一个数值分数。

**LRU 的四个盲区**：

| 盲区 | 说明 |
|---|---|
| 1. 看不见"每字节的价值差异" | 状态块和 KV 块占地差不多，但代表长度可能差几十倍 |
| 2. 扫表污染（scan pollution） | 一批"只用一次"的新请求会把热前缀全挤出去（经典的 LRU 失效场景） |
| 3. 频次盲 | 被一万个请求共用的 system prompt 块，和只用过一次的块，在它眼里只差"新旧" |
| 4. 目标错位 | LRU 优化"命中次数"，但真正想要的是"省下的算力"——有长度差异时两者背离 |

---

## 4. 具体例子

### 例子 A：同一个分叉点下的长短分叉

```
root
 └──[2000 token]──▶ P（带状态）
                     ├──[64 token]────▶ A（带状态）
                     └──[4096 token]──▶ B（带状态）
```

约定：**1 token 的 KV = 1 单位内存；1 token 的计算 = 1 单位；1 份 SSM 状态 = 100 单位内存。**

因为是相对父节点 P 算增量：

| 条目 | 增量 token | 占用内存 | 能省的计算 | 性价比 |
|---|---|---|---|---|
| **A**（短分叉） | 64 | 64 + **100** = 164 | 64 | **0.39** |
| **B**（长分叉） | 4096 | 4096 + **100** = 4196 | 4096 | **0.98** |

看 A：它也挂了整整一份状态（100 单位），却只换来 64 个 token 的可跳过空间——**固定开销完全摊不开**。

再看新旧：A 刚被访问过（recency = 0.9），B 有一阵没用了（recency = 0.2）。
归一化后（假设这棵树性价比范围就是 0.39~0.98）：A 的 flop_eff = **0**，B 的 flop_eff = **1**。

| α | S(A) | S(B) | 淘汰谁 |
|---|---|---|---|
| **0**（纯 LRU） | 0.90 | **0.20** | **B（长的那个）** |
| **0.5** | **0.90** | 0.60 | **A** |
| **1** | **0.90** | 1.20 | **A** |

**后果**：假设接下来 8 个请求都走 B 分支。

| | Marconi（扔 A） | LRU（扔 B） |
|---|---|---|
| 请求能跳到哪 | 直接跳到 **6096**（P 的 2000 + B 的 4096 全跳过） | 只能跳到 **2000**（B 那段没了） |
| 每个请求重算 | **0** | **4096** |
| 8 个请求合计重算 | **0** | **32,768 个 token** |
| 若之后 8 个请求走 A | 每个重算 64，合计 512 | 0（A 保住了） |

**净账：Marconi 少算了约 32,000 个 token 的重复 prefill。**

而且这不是偶发失误，是**系统性**的：每次出现新的短分叉，都会把老的、长的条目往后挤，缓存会逐渐被"短而新"的条目占满。

### 例子 B：淘汰"中间节点"和淘汰"叶子"不一样

```
root
 └──[2000 token]──▶ P（带状态）
                     ├──[64 token]───▶ A（带状态，叶子）
                     └──[512 token]──▶ C（带状态，只有一个孩子）
                                        └──[4096 token]──▶ B（带状态，叶子）
```

| 淘汰谁 | 实际释放什么 | 之后还能复用到哪 |
|---|---|---|
| **C**（中间节点） | **只释放状态**（100）；512 token 的 KV **被孩子 B 吸收** | 还能跳到 B（路是通的），只是**不能在 C 处停下** |
| **A**（叶子） | 状态 + 自己那段 KV（164） | 走 A 的请求只能退回到 P |

**直觉**：**KV 是复用链上的"路"，状态是打开这条路的"钥匙"。**
淘汰中间节点 = 收走钥匙、路留着（并交给孩子）；淘汰叶子 = 钥匙和这段路一起没了。

所以像 C 这种"单孩子的中间节点"是**最理想的淘汰目标**：收走它那份昂贵的状态，几乎不影响任何人继续走路。这正是论文说"不只淘汰叶子"的原因。

### 什么时候这个策略不起作用（所以它不会更差）

| 情况 | 结果 |
|---|---|
| 两个候选一样长 | 性价比相同 → 只由新旧决定 → **退化成 LRU** |
| 候选都不带状态（纯 KV） | 性价比都等于基准线 → 退化成 LRU |
| 缓存没满 | 根本不淘汰 |
| 序列长度分布很窄 | 所有条目性价比接近 → 退化成 LRU |

**性质**：只在"条目之间的'覆盖长度 ÷ 占用内存'差异很大，且缓存压力大到必须做选择"时才起作用；否则自动退化成 LRU。

---

## 5. 收益场景（简述）

**有收益的前提（三条缺一收益就掉）**

1. 缓存里同时存在"固定大块（SSM 状态）"和"线性小块（KV）"——即存在**密集 checkpoint**（`all` 模式 / 长 retention / 多个 checkpoint 常驻）；
2. 缓存**紧张到需要择优**（论文实测：**中等竞争收益最大**，60→140GB 时相对 LRU 提升 24.3% / 51.5% / 68.3% / 30.0% / 10.0%）；
3. **序列长度分布宽**（有"值得偏心"的对象）。

**收益大的场景**：agent / 编码类（长度跨度最大，SWE-bench 命中率 16.4%→32.7%、FLOP saved +90.3%、P95 提升 219.7%）、1 个共享 anchor + N 个候选（RL rollout / 多候选）、多租户 / RAG（防 burst 污染）、长输出对话（LMSys，45.6%）。

**收益小的场景**：纯 Transformer（论文说三套系统表现一样）、缓存宽裕、短序列为主（ShareGPT 仅 19.0%）、当前默认的稀疏保留配置（缓存里主要剩 KV 块）。

**代价**：短序列命中率最多 **-3.0%**；P5 TTFT 差 6.3%（**绝对只差 2.1ms**）；换来长序列命中率 **+25.5%**、P50/P95 TTFT 改善 **13.4% / 22.0%**（74.2ms / 274.9ms）。

---

## 6. 实现现状（截至 2026-09-20）

把"驱逐"拆成三层看：**框架（能不能换策略）→ 策略（用什么规则）→ 混合感知的价值核算（SSM vs KV 的性价比）**。

### 第一层：框架

| | vLLM 上游 | vllm-ascend |
|---|---|---|
| **CPU / offload 层** | ✅ 已合：[#27039](https://github.com/vllm-project/vllm/pull/27039) 引入 ARC（2025-11-12）；[#49114](https://github.com/vllm-project/vllm/pull/49114) 加 `CachePolicyFactory`，**外部包可按名字注册策略**（2026-07-29） | 🟠 **没搬过来**：[#7125](https://github.com/vllm-project/vllm-ascend/pull/7125)、[#7469](https://github.com/vllm-project/vllm-ascend/pull/7469)（NPU offloader 支持 ARC）**都 open、都带 merge-conflicts** |
| **GPU / HBM 层**（真正决定显存里留什么） | 🔴 **仍是单队列 LRU**：[#40270](https://github.com/vllm-project/vllm/pull/40270)（open，needs-rebase）写明 `BlockPool` 用 `FreeKVCacheBlockQueue`，没有频率信号、会被 scan 污染；它提出 `GPUCachePolicy` 抽象，**至今没合** | 🔴 跟随上游（单队列 LRU） |

### 第二层：策略

| 策略 | 状态 |
|---|---|
| ARC（自适应 新旧↔频次） | ✅ 已合，但**只在 CPU offload 路径** |
| T-LRU [#37825](https://github.com/vllm-project/vllm/pull/37825) / SLRU [#38984](https://github.com/vllm-project/vllm/pull/38984) | 🟠 open |
| Session-Aware [#50422](https://github.com/vllm-project/vllm/pull/50422) / Priority-Aware [#47837](https://github.com/vllm-project/vllm/pull/47837) | 🟠 open |
| Block 优先级 [#55403](https://github.com/vllm-project/vllm/pull/55403) | 🟠 open（needs-rebase） |
| **频次 + 成本**（思路与论文几乎一致）[#23641](https://github.com/vllm-project/vllm/issues/23641) | ❌ **被 stale bot 关闭，全程没有人类回复** |
| 昇腾自有策略 | ❌ 无 |

### 第三层：混合感知的价值核算

**上游和昇腾都是 0。** 没有任何 PR 实现"能省的计算 ÷ 占的内存"这种评分。

但 vLLM 用**另一条路**达到了部分目的——**稀疏保留（retention interval）**：

- 机制：把大部分块**打上掩码**（不分配哈希、不参与匹配，**按 FIFO 淘汰**），只保留少数被认为值得留的位置（prompt 尾部边界 + 共享前缀分叉点，即 [#47782](https://github.com/vllm-project/vllm/pull/47782) 的 `reachable_boundaries`）；
- 本质：**把"价值函数"硬编码成人工规则**，而不是算出来的。它回答"哪些位置值得留"，但不回答"多个候选里先扔哪个"；
- 相关：准入 [#37898](https://github.com/vllm-project/vllm/pull/37898)（2026-06-10 合）；retention interval [#43447](https://github.com/vllm-project/vllm/pull/43447) / [#45845](https://github.com/vllm-project/vllm/pull/45845)；正式参数化 [#52216](https://github.com/vllm-project/vllm/pull/52216)（默认 0，2026-08-17 合）；Mamba+EAGLE 默认值修复 [#55760](https://github.com/vllm-project/vllm/pull/55760)（2026-09-08 合）。

混合模型上已合的、"释放"相关的工作（是"把不该占的放开"，不是"在候选里择优"）：

- [#28047](https://github.com/vllm-project/vllm/pull/28047)：Mamba2 hybrid 的 prefix cache blocks 在不再被运行中请求需要时**允许被淘汰**（之前被钉住，缓存涨不上去）；
- [#35219](https://github.com/vllm-project/vllm/pull/35219)：修"SSM cache block 释放数为 0"的 bug。

### 现在实际能调的旋钮

| 旋钮 | 作用 |
|---|---|
| `--enable-prefix-caching` | 开/关前缀缓存 |
| `--mamba-cache-mode align` | 块对齐保存 SSM 状态（混合模型前缀缓存的前提） |
| `--prefix-cache-retention-interval` | 默认 0 = 稀疏保留（只留最新 replay 边界 + 共享前缀分叉点） |
| CPU offload 的 `eviction_policy`（`lru` / `arc`） | 只影响 **offload 层**，不影响 HBM 里留谁 |
| **GPU 侧的驱逐算法** | **没有这个旋钮** |

### 结论

> **上游 vLLM：框架在半条路上通了（offload 层已有 ARC + 可插拔注册），但 GPU/HBM 侧至今是单队列 LRU，改造卡在 [#40270](https://github.com/vllm-project/vllm/pull/40270)；混合模型的性价比核算完全没有——它用"人工规定只保留哪些边界"代替了"自动算性价比"。**
>
> **vllm-ascend：连上游已合的 ARC 都还没搬（两个 PR 挂着 merge-conflicts），HBM 侧跟随上游 LRU，混合感知的驱逐是 0。** 另外两个直接影响 hybrid 复用覆盖的开放项：[#14468](https://github.com/vllm-project/vllm-ascend/pull/14468)（v0.25.1rc 上 Marconi 准入不生效）、[#13300](https://github.com/vllm-project/vllm-ascend/pull/13300)（AscendStore 未适配 `reachable_boundaries`）。

**一句话**：**"把 SSM 状态和 KV 放在一起按价值择优淘汰"这件事，vLLM 和 Ascend 都还没开始做。**

### 自查命令（确认你手上的版本）

```bash
# 1) GPU 侧是否还是单队列 LRU（有 GPUCachePolicy 说明插槽已打开）
grep -rn "FreeKVCacheBlockQueue\|GPUCachePolicy" $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/v1/core/ | head

# 2) offload 层是否已有可插拔策略与 ARC
grep -rn "CachePolicyFactory\|ARCCachePolicy" $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/v1/kv_offload/ | head

# 3) 昇腾：Marconi 准入信号是否接通（返回三元组 / 设置了 num_uncached_common_prefix_tokens）
grep -rn "num_uncached_common_prefix_tokens\|longest_hit_length" \
  $(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))") | head
```

---

## 7. 如果要落地：最小可行方案

**不需要完整复刻论文的 FLOP 公式。** 最小记账是：

1. 给每个缓存条目（或每个块）额外记一个整数：**"它比父节点多代表了几个 token"**；
2. 性价比 = `增量 token 数 ÷ 占用字节数`；
3. 最终分数 = `归一化的新旧 + α × 归一化的性价比`，淘汰分数最低的。

| 条目类型 | 这个近似的表现 |
|---|---|
| 纯 KV 块 | 比值恒定 → 自动退化成"按内存归一"的 LRU，不会出错 |
| 带 SSM 状态的条目 | 比值直接反映"固定开销摊开了多少" |

**接入路径**

- 上游：先等/参与 [#40270](https://github.com/vllm-project/vllm/pull/40270) 把 GPU 侧 `GPUCachePolicy` 插槽打开，再把上面这个度量接进去；offload 层现在就可以用 [#49114](https://github.com/vllm-project/vllm/pull/49114) 的工厂直接注册。
- 昇腾：先把 [#7125](https://github.com/vllm-project/vllm-ascend/pull/7125) / [#7469](https://github.com/vllm-project/vllm-ascend/pull/7469) 的 ARC 适配救活作为最小验证，再考虑 hybrid 感知的度量。

**前置条件**：先有度量（命中率 / 被丢弃的复用量：上游 [#52527](https://github.com/vllm-project/vllm/pull/52527)、昇腾 [#16639](https://github.com/vllm-project/vllm-ascend/issues/16639)），再做离线 trace 复现，证明"目标场景能赢、默认场景不退化"。
