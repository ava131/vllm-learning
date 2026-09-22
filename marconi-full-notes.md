# Marconi 完整梳理：原理 · 数据 · 实现现状 · Ascend 落地路线

> 整理日期：2026-09-20
> 说明：本文合并并取代之前两份笔记（`marconi-prefix-caching-vllm-ascend-chat.md`、`marconi-eviction-strategy-and-status.md`）。
> 所有"实现现状"结论均以 GitHub PR/Issue 原文和仓库文件内容为依据，并在文中标注 PR 编号；标注"推断"的地方是推断，不是事实。

---

## 0. 一页速览

| 项目 | 结论 |
|---|---|
| **论文** | Marconi: Prefix Caching for the Era of Hybrid LLMs，Princeton + AWS，MLSys 2025，[arXiv:2411.19379](https://arxiv.org/abs/2411.19379)，代码 [github.com/ruipeterpan/marconi](https://github.com/ruipeterpan/marconi) |
| **解决什么** | 混合模型（Attention + SSM，如 Mamba / Jamba / Qwen3.5）上，前缀缓存几乎失效：SSM 状态无法回滚 → 只能精确匹配 → 要支持任意前缀就得密集存档 → 又大又几乎不被命中 → 缓存被撑爆 |
| **两大部分** | ① **存（准入）**：只缓存"共享前缀分叉点"和"对话尾部"两处；② **删（驱逐）**：缓存满时按 `新旧 + α × 性价比` 打分，先淘汰分低的 |
| **论文收益** | 命中率最高 34.4×（vs vLLM+）；P95 TTFT 降幅再多 36.1%~71.1%（103~617ms）；单看驱逐（vs LRU）命中率 +19.0%~219.7% |
| **上游 vLLM 现状** | **准入已落地**（[#37898](https://github.com/vllm-project/vllm/pull/37898) 2026-06-10、[#47782](https://github.com/vllm-project/vllm/pull/47782) 2026-07-13）；**驱逐的"性价比评分"完全没人做**（唯一的同类 RFC [#23641](https://github.com/vllm-project/vllm/issues/23641) 被 stale bot 关掉） |
| **Ascend 现状** | 基础 hybrid/Mamba 前缀缓存已上（[#7103](https://github.com/vllm-project/vllm-ascend/pull/7103) 等）；**但 v0.23.0 基线里没有 Marconi 准入**（已核实：vLLM v0.23.0 无该代码）；**驱逐是 0** |
| **能不能做** | 能。而且路径清晰：0.23.0 上先补"准入"（有实测数据支撑，[#14468](https://github.com/vllm-project/vllm-ascend/pull/14468) 证明 0%→34~69% 命中率），再做"驱逐"（全社区空白，价值高但需要在 0.23.0 上先开 GPU 侧策略插槽） |

---

# 第一部分 · 原理

## 1.1 问题：混合模型上"前缀缓存"卡在哪

**前缀缓存**：很多请求的开头一模一样（系统提示词、few-shot 示例、多轮对话的历史），这段算过一次就把中间结果留着，下次直接跳过，省掉重复的"读题"时间。

混合模型有两类层，中间结果的形态完全不同：

| | 中间结果是什么 | 能不能"只留前面一段" |
|---|---|---|
| Attention 层 | 每个 token 一份 KV（"原话复印件"） | **能**，沿序列维度切片即可 |
| SSM 层 | 一本固定大小的状态（"读书笔记"），逐 token 就地改写 | **不能**，第 100 页的笔记退不回第 80 页 |

于是形成死结：

- 想支持任意前缀 → 必须**每 x 个 token 存一份 SSM 状态**（密集存档）；
- SSM 状态又大又固定 → 存一堆**以后再也没人翻**的状态 → 缓存抖动。

论文给出的量化证据：

| 事实 | 数字 |
|---|---|
| block size 32 时：KV 块复用率 vs SSM 状态复用率 | **25.0% vs 0.4%**（差 **65.3×**） |
| block size 128 时 SSM 状态复用率 | 仍只有 **3.3%** |
| 7B 模型、单条 10K token 序列的缓存占用 | **17.4 GB**，是同规模 Transformer 的 **3.3×** |
| 一个 SSM 状态 vs 单个 token 的 KV | 大约 **10~100 倍** |
| block size 16 时，一个 SSM 层的状态 vs 该 token block 里 Attention 层的 KV | 约 **4 倍** |

**SSM 状态的三条性质**（论文原文归纳）：
1. 大小固定，与代表多少 token 无关；
2. 就地更新，无法回滚成前缀；
3. 比单个 token 的 KV 大几个数量级。

**核心矛盾**：SSM 状态是"全有或全无"的——要复用一个前缀，必须**所有 Attention 层的该前缀 KV + 每个 SSM 层的一个精确匹配状态**同时存在，复用机会被"复用机会最少的层"卡住。

---

## 1.2 存（准入）：只在两个地方存

论文分析了真实 trace 里所有被复用的前缀，归纳为两类：

| 类型 | 例子 | 复用形态 |
|---|---|---|
| **Purely input**（纯输入共享） | system prompt、指令、few-shot 示例、长文档 QA、自一致性采样 | 被**很多不同请求**共享 |
| **Input + output**（输入+输出共享） | 多轮对话历史、agent 的历史交互轨迹 | **从尾部往后接**，不分叉 |

对应两种做法：

### (a) 对话尾部状态（last decoded state）
每个请求解码结束后，**把最后一个 token 处的状态存下来**。因为下一轮的输入 = 上一轮的输入 + 上一轮的回答，正好从这个位置接上。这条路径**立即生效**（第一次就能用）。

### (b) 共享前缀分叉点（speculative insertion）
在真正 prefill 之前，先把这条请求**拿到前缀树里"试插一下"**：
- 如果它会插到某条已有边的中间、**把边劈成两段、产生一个新的中间节点** → 说明这条请求的前一截和以前某个请求重复了，这是一个"共享前缀分叉点"，**值得在这里存一份状态**；
- 否则什么都不存。

### 由此带来的时间语义（重要）

> **共享前缀从"第 3 次出现"开始才省时间。**

- 第 1 次：没人知道这段会重复；
- 第 2 次：这时才能"发现"重复（需要两次才能比出来），于是存下状态——但这次自己已经白算了；
- 第 3 次：直接从这里接着读，开始省。

对 system prompt 这种被成百上千个请求共享的内容，少省一次可以忽略。

### 结果：每条序列最多约 2 个状态

一个"分叉点"的 + 一个"尾部"的。KV 仍然全量缓存（与现有系统一致）。这是**用覆盖率换缓存效用**的取舍：牺牲"任意前缀的复用能力"，换来"缓存里几乎没有废纸"。

### 怎么拿到"中间那个状态"（技术细节）

| 模型类型 | 方法 |
|---|---|
| 支持 chunked state passing（Mamba2、RetNet、GLA 等） | 直接物化缓存**倒数第二个 chunk** 的状态（位置误差在一个 chunk 内，开销极小） |
| 不支持（Mamba1、Jamba 等） | **两遍 prefill**：第一遍只算到目标位置拿到状态，第二遍从该状态接着算剩下的 |

### 与 vLLM 实现的差异（关键）

论文是**提前猜**（投机插入 + radix tree）；vLLM 是**事后看**：

> 分别看 Attention 组的 KV 命中了多长、SSM 组的命中多长；如果 **KV 命中了但 SSM 没跟上**，就说明存在一个没被缓存的共享前缀——于是在**块对齐的位置**补一份状态。

所以 vLLM 是 **"Marconi-style"（效果类似、方法不同）**，不是论文的移植。

---

## 1.3 删（驱逐）：让"值钱"参与排序

### 1.3.1 计分单位：前缀树上的"条目"，不是 block

```
节点 = 分叉点（序列分开的地方）
边   = 相邻两个节点之间的一段 token
条目 = 这条边的 KV ＋ 挂在它末端节点上的 SSM 状态
```

- **不是固定大小的 block**，而是一段**长度可变**的 token；
- 不是每个节点都带状态：只有被"准入"过的节点才有（共享前缀分叉点 + 序列尾巴）。

在 vLLM 里没有这个"条目"概念（它的单位是固定大小的 block），这是**照搬不了的根本原因**。

### 1.3.2 公式三件套

**① 性价比（FLOP 效率）**

```
flop_eff(n) = 这个条目能省下的计算量 / 这个条目占的内存
```

| | 内容 |
|---|---|
| 分子 | 复用这条边能省下的计算（Attention + SSM + MLP 三层都算），**且只算"相对父节点多出来的那一段"** |
| 分母 | 这条边的 KV + 它带的 SSM 状态（固定大小） |

"相对父节点"必须有：否则父节点已经能省的那段会被重复计入，深层节点全被高估。

内存口径（论文附录）：Attention 层 KV = `2 · L · D · 2` 字节；SSM 层状态 = `D · N · 2` 字节；conv1d 状态 = `in_channels · conv_kernel · 2` 字节（在 7B 混合模型里占 6.1%，实验中计入、公式里省略）。

**② 归一化**：`recency` 和 `flop_eff` 都在**整棵树的所有节点**上做 min-max 归一化到 0~1（一个是时间、一个是"算力/字节"，量纲不同，不归一化无法相加）。

**③ 最终分数**

```
S(n) = recency(n) + α · flop_eff(n)
```

- `α = 0` → 退化成纯 LRU（这是它的安全网）；
- **淘汰 = 反复挑分数最低的删掉，直到腾够空间**（按需，不做全量大清洗）。

**④ α 自适应**

1. 启动时 α = 0（先按 LRU 跑）；
2. 第一次发生淘汰后拍快照，进入**观察期**（继续用 LRU，但记录请求）；
3. 观察期长度 ≈ 第一次淘汰前见过的请求数的 **5~15 倍**；
4. 后台**回放**这些请求，对多组 α 做网格搜索（多 CPU 核并行，**几秒出结果**，常数比一个请求的完整 prefill+decode 还短）；
5. 选命中率最高的 α，之后固定使用。

### 1.3.3 优先级规则（回答"是不是有属性上的优先级"）

**有，但要分两层。**

**第一层：结构规则——谁有资格进候选**

| 规则 | 说明 |
|---|---|
| **孩子数 ≥ 2 的节点：豁免** | 它代表"被多个请求共享的公共骨架"，淘汰它会让多个分支同时失去复用能力，代价最大 |
| **孩子数 ≤ 1 的节点：进候选** | 包括叶子，也包括"单孩子的中间节点"——后者代表"只被走过一次的中间路径"，不会再被复用，却白占一份状态 |
| 连锁降级 | 当某个节点的孩子被淘汰到只剩 1 个时，它自己也会**从豁免变成候选**；但如果它本身很值钱，分数高，仍然活下来 |

**第二层：分数排序——候选里谁先走**

分数 = `新旧 + α × 性价比`，两个维度合成为一个数，**分低的先走**。

| 维度 | 含义 | 高分的含义 |
|---|---|---|
| 新旧（recency） | 最后一次被访问的时间，全树归一化 | 越新越高 |
| 性价比（flop_eff） | 覆盖长度 ÷ 占用内存，全树归一化 | 越"摊得开"越高 |

**对比 LRU**：LRU 的链表位置**本身就是一种分数**（分数 = 最近访问时间，只有一个维度）。加第二个维度后，链表排不出正确的序（一个条目可能"很新但不值钱"或"很旧但很值钱"），所以必须改成"算一个数、挑最小"。

> 补充：ARC / 2Q / T-LRU 等策略是"**用多个链表去近似第二个维度（频次）**"；Marconi 是直接算数值分数。

### 1.3.4 两个容易忽略的实现细节

1. **淘汰中间节点和淘汰叶子不一样**

| 淘汰的节点 | 释放什么 | 为什么 |
|---|---|---|
| **叶子节点** | 自己的 KV + 自己的状态 | 没有更深的复用在用这些 KV |
| **中间节点**（≤1 孩子） | **只释放状态**，KV **被孩子吸收** | 更深的复用要从根一路读到孩子，缺了这段 KV 就走不通 |

**直觉**：**KV 是复用链上的"路"，状态是打开这条路的"钥匙"。** 淘汰中间节点 = 收走钥匙、路留着（并交给孩子）；淘汰叶子 = 钥匙和这段路一起没了。

2. **命中时不刷新祖先节点的时间戳**（传统系统会刷新）：祖先的状态不会被复用、KV 也会被子节点吸收，刷新只会让该走的不走。

### 1.3.5 为什么"大小"不能当代理（对比 GDSF）

| | 内存大小 | 与"省下的计算"的关系 |
|---|---|---|
| Attention KV | `2 · L · D · 2` 字节 | **线性成正比** → 大小可以当代理 |
| SSM 状态 | `D · N · 2` 字节（**与 L 无关**） | **完全脱钩** → 大小不能当代理 |

所以一个"代表 200 token"的 KV 条目和一个"代表 20000 token"的状态条目，占地可能差不多，价值差 100 倍。LRU 和"按大小淘汰"都看不见这个差别。

### 1.3.6 地基：为什么"长前缀的缓存每字节更值钱"

| | 内存 | 复用它省下的算力 | 每字节省下的算力 |
|---|---|---|---|
| KV | ∝ 长度 | ∝ 长度 | **基本恒定** |
| SSM 状态 | **固定** | ∝ 长度 | **∝ 长度** |

结论：

> **SSM 状态是一笔"贵的、固定的开销"。它划不划算，取决于这笔开销被摊在了多长的前缀上。**

由此得到三类条目的性价比排序：

| 条目 | 性价比 |
|---|---|
| 深位置的 checkpoint（花一份状态钱，能跳过一大段） | **最高** |
| 普通 KV 块（不带状态） | **基准线（恒定）** |
| 紧贴父节点下面的浅 checkpoint（多跳过几十个 token，却占一整份状态） | **最低** |

**所以不是"优先存 SSM"，而是"SSM 状态要惜着存、挑着留"**：
- 准入 = 只在该花的地方花（分叉点 + 尾巴）；
- 驱逐 = 已经花出去的多笔开销，先退掉摊得最薄的那笔。

论文的微基准也印证：SSM 层占比越高（1:2 → 1:8）、状态维度越大（16 → 128），这套策略的优势越大。

---

## 1.4 三个例子

统一约定：**1 token 的 KV = 1 单位内存；1 token 的计算 = 1 单位；1 份 SSM 状态 = 100 单位内存。**

### 例 1：同一个分叉点下的长短分叉

```
root
 └──[2000 token]──▶ P（带状态）
                     ├──[64 token]────▶ A（带状态）
                     └──[4096 token]──▶ B（带状态）
```

因为是相对父节点 P 算增量：

| 条目 | 增量 token | 占用内存 | 能省的计算 | 性价比 |
|---|---|---|---|---|
| **A**（短分叉） | 64 | 64 + **100** = 164 | 64 | **0.390** |
| **B**（长分叉） | 4096 | 4096 + **100** = 4196 | 4096 | **0.976** |

A 也挂了整整一份状态（100 单位），却只换来 64 个 token 的可跳过空间——**固定开销完全摊不开**。

新旧：A 刚被访问过（0.9），B 有一阵没用了（0.2）。归一化后（假设这棵树里性价比范围就是 0.390~0.976）：A 的 flop_eff = **0**，B 的 flop_eff = **1**。

| α | S(A) | S(B) | 淘汰谁 |
|---|---|---|---|
| **0**（纯 LRU） | 0.90 | **0.20** | **B（长的那个）** |
| **0.5** | **0.90** | 0.60 | **A** |
| **1** | **0.90** | 1.20 | **A** |

**后果**：假设接下来 8 个请求都走 B 分支。

| | Marconi（扔 A） | LRU（扔 B） |
|---|---|---|
| 请求能跳到哪 | 直接跳到 **6096**（P 的 2000 + B 的 4096） | 只能跳到 **2000** |
| 每个请求重算 | **0** | **4096** |
| 8 个请求合计重算 | **0** | **32,768 个 token** |
| 若之后 8 个请求走 A | 每个重算 64，合计 512 | 0（A 保住了） |

**净账：Marconi 少算了约 32,000 个 token 的重复 prefill。**
而且这不是偶发失误，是**系统性**的：每次出现新的短分叉，都会把老的、长的条目往后挤。

### 例 2：淘汰"中间节点"和淘汰"叶子"不一样

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

C 是**最理想的淘汰目标**：收走它那份昂贵的状态，几乎不影响任何人继续走路。这正是论文说"不只淘汰叶子"的原因。

### 例 3：又来一个和 B 一样长的新条目，P 会被淘汰吗？

```
root
 ├──[2000 token]──▶ P（带状态，2 个孩子）
 │                   ├──[64 token]────▶ A（带状态）
 │                   └──[4096 token]──▶ B（带状态）
 └──[4096 token]──▶ N（带状态，新来的）
```

**答案：P 不会被淘汰。**

**第一步看候选资格**：P 有 2 个孩子 → 代表被多个请求共享的公共开头 → **豁免，不在候选名单里**。A、B、N 是叶子，都在候选里。

**第二步算分数**（P 的增量是相对 root 的 2000 token）：

| 条目 | 段长 | 占用 | 能省 | 性价比 | 归一化 eff | recency | 分数（α=1） |
|---|---|---|---|---|---|---|---|
| **P**（2 孩子，**非候选**） | 2000 | 2100 | 2000 | 0.952 | 0.959 | 0.7 | 1.66 |
| **A** | 64 | 164 | 64 | **0.390** | **0** | 0.9 | **0.90** ← 最低 |
| **B** | 4096 | 4196 | 4096 | 0.976 | 1.0 | 0.2 | 1.20 |
| **N**（新来） | 4096 | 4196 | 4096 | 0.976 | 1.0 | 1.0 | 2.00 |

**Marconi 的淘汰顺序**：先 **A**。若还需继续腾空间，B 和 N 性价比**完全相同** → 这时才由"新旧"决定 → **B 走**（它更旧）。
**LRU 的顺序**：直接淘汰 **B**（0.2 最旧），留着性价比只有 1/4 的 A。

**第三步看连锁效应**：A 被淘汰后，P 的孩子数从 2 变成 **1** → **P 从此进入候选名单**；但 P 的分数（1.66）比 B（1.20）还高（段长 2000、性价比 0.952、更新），所以**它仍然安全**。

> 这个例子同时说明两件事：① "多孩子节点豁免"保护的是**公共骨架**；② "单孩子中间节点也可淘汰"清理的是**降级后没人再用的节点**，而不是骨架本身。

### 什么时候这套策略不起作用（所以它不会更差）

| 情况 | 结果 |
|---|---|
| 候选一样长 | 性价比相同 → 只由新旧决定 → **退化成 LRU** |
| 候选都不带状态（纯 KV） | 性价比都等于基准线 → 退化成 LRU |
| 缓存没满 | 根本不淘汰 |
| 序列长度分布很窄 | 所有条目性价比接近 → 退化成 LRU |

**性质**：只在"条目之间'覆盖长度 ÷ 占用内存'差异很大，且缓存压力大到必须做选择"时才起作用；否则自动退化成 LRU，不会更糟。

---

# 第二部分 · 数据与收益

## 2.1 实验设置

| 项目 | 内容 |
|---|---|
| **数据集** | ① **LMSys**（多轮对话，输出较长，常达数千 token）；② **ShareGPT**（多轮对话，输出简短，几十~几百 token）；③ **SWE-Agent on SWE-Bench**（agent 工作负载，输入长度分布最宽，几百 ~ 几万 token） |
| **到达模式** | 变化 inter-session 与 inter-request 到达间隔，模拟人类打字/IDE 交互与排队延迟 |
| **模型** | 主实验：7B 混合模型，{4 Attention, 24 SSM, 28 MLP} 层；TTFT 洞察：**Jamba-1.5-Mini**（12B active / 52B total，state dim 128，vLLM 实现），4×A100-40GB |
| **硬件** | AWS p4d.24xlarge：8×A100-40GB、96 vCPU、1152GB DDR4；FP16 |
| **Baseline** | ① **Vanilla**（不开前缀缓存）；② **vLLM+**（细粒度存档：每个 token block 存一份状态，block size 32，是 vLLM 支持的最大值）；③ **SGLang+**（**用 Marconi 的准入 + 纯 LRU 驱逐**——用来隔离出"驱逐"单独的贡献） |
| **指标** | **token 命中率**（跳过的 prefill token 数 ÷ 总输入 token 数，近似省下的算力）；**TTFT** 的 P5/P50/P95；FLOP saved |
| **不做的事** | 不评估下游精度指标——前缀复用是精确复用，不改变模型输出 |

## 2.2 收益：准入 + 驱逐（vs vLLM+）

| 指标 | LMSys | ShareGPT | SWEBench |
|---|---|---|---|
| Token 命中率提升（平均） | **4.5×** | **7.3×** | **34.4×** |
| P95 TTFT 降幅（vs 无前缀缓存） | 36.9%（281.4ms） | 73.2%（106.3ms） | 46.8%（**617.0ms**） |
| P95 TTFT 降幅再多（vs vLLM+） | 36.1%（275.4ms） | 71.1%（103.3ms） | 46.8%（617.0ms） |
| P95 TTFT 降幅再多（vs SGLang+） | 17.2%（131.1ms） | 12.8%（18.5ms） | 24.7%（325.7ms） |

## 2.3 收益：只看"驱逐"（vs SGLang+，准入相同）

| 指标 | 结果 |
|---|---|
| Token 命中率提升（P95） | **19.0% ~ 219.7%** |
| SWEBench（长度分布最宽） | 命中率 **16.4% → 32.7%**（+99.4%），**FLOP saved +90.3%** |
| 分场景 P95 提升 | SWEBench **219.7%** / LMSys **45.6%** / ShareGPT **19.0%** |
| 命中率按长度分桶 | <7K token：最多 **-3.0%**；>7K token：最多 **+25.5%** |
| TTFT | P5 差 6.3%（**绝对只差 2.1ms**）；P50 **-13.4%**（74.2ms）；P95 **-22.0%**（274.9ms） |

**这是最有价值的一组数**：它证明"驱逐策略"本身（在准入相同的前提下）能带来一倍量级的 token 命中率提升，且代价只是短序列几毫秒。

## 2.4 微基准（什么条件下收益变大）

| 变量 | 结果 |
|---|---|
| **缓存竞争度**（60 / 80 / 100 / 120 / 140 GB） | 相对 SGLang+ 提升 **24.3% / 51.5% / 68.3% / 30.0% / 10.0%** → **中等竞争时收益最大**；高竞争时命中率基数掉到 <10%，低竞争时随便扔都行 |
| **层配比**（Attention:SSM = 1:2 / 1:4 / 1:8） | 相对 vLLM+：13.5% → 66.6% → **2.6×**；相对 SGLang+：**5.8% → 26.0% → 59.7%**；**纯 Transformer 上三套系统完全一样** |
| **SSM 状态维度**（16 → 128，Mamba1 → Mamba2） | 相对 vLLM+：**5.7× → 35.4×** |
| **到达模式** | 会话率 0.5→2/s：命中率 48.7%→43.0%，但相对 SGLang+ 的优势 **1.4× → 1.6×**（竞争加剧，择优更重要） |

## 2.5 代价与边界

**代价（有意为之）**：
- 短序列命中率最多 **-3.0%**，P5 TTFT 差 6.3%（**绝对值 2.1ms**）；
- 换来长序列命中率 **+25.5%**，P50/P95 TTFT 改善 **13.4% / 22.0%**；
- 共享前缀要**第 3 次出现**才开始省；
- 每条序列最多约 2 个状态 → 牺牲了"任意前缀"的覆盖能力。

**没收益的场景**：纯 Transformer；缓存宽裕；短序列为主（ShareGPT 只有 19.0%）；当前默认的稀疏保留配置（缓存里主要剩 KV 块，性价比近似常数）。

**论文自身的边界**：不评估下游精度（合理，因为是精确复用）；two-pass prefill 有额外开销；评估基于自己的原型系统（radix tree 实现），不是生产引擎。

## 2.6 这个工作的价值有多大（我的判断，标注为推断）

| 层次 | 价值 | 依据 |
|---|---|---|
| **工程价值** | 高 | 混合模型是长上下文服务的主流方向；不做这件事，前缀缓存命中率就是 0%——这是**功能性缺陷**，不是优化 |
| **学术价值** | 已兑现（MLSys 2025） | 但"驱逐"这一半在工业引擎里仍是空白 |
| **社区影响力** | 中高 | 上游只实现了"准入"那一半，且方法不同；"驱逐"的同类 RFC（[#23641](https://github.com/vllm-project/vllm/issues/23641)）被 stale bot 关掉，**谁先做谁就是第一份** |
| **可落地性** | 中 | 需要新的记账（每个条目代表多长），且 GPU 侧策略插槽（[#40270](https://github.com/vllm-project/vllm/pull/40270)）还没合——但 offload 侧已有可插拔工厂（[#49114](https://github.com/vllm-project/vllm/pull/49114)）可以立刻用 |

---

# 第三部分 · 实现现状（vLLM / Ascend）

## 3.1 把 Marconi 拆成 12 个能力块

| 编号 | 能力（白话） | 归属 |
|---|---|---|
| A1 | KV 与 SSM 状态绑定管理（复用时要一起有） | 基础 |
| A2 | 前缀匹配（找最长公共开头） | 基础 |
| A3 | 状态存在块边界上（避免改 kernel） | 基础 |
| B1 | 对话尾部状态（多轮对话复用） | **存** |
| B2 | 共享前缀识别（第 2 次发现、第 3 次命中） | **存** |
| B3 | 在共享前缀分叉点物化 SSM 状态 | **存** |
| B4 | 获取中间状态的手段（chunk 对齐 / two-pass） | **存** |
| C1 | 性价比（FLOP 效率）效用函数 | **删** |
| C2 | α 自适应调优 | **删** |
| C3 | 中间单孩子节点可淘汰 + 命中不刷祖先 | **删** |
| D1 | 部分命中（FA 命中 / SSM 未命中）与数值一致 | 集成 |
| D2 | PD / MTP / EAGLE / MRv2 / 多模态 / TP 组合正确性 | 集成 |
| D3 | 指标与可观测性 | 集成 |

## 3.2 上游 vLLM：每个 PR 干了什么

### 基础（A1–A3）

| PR / Issue | 状态 | 干了什么 |
|---|---|---|
| [#25752](https://github.com/vllm-project/vllm/pull/25752)、[#26377](https://github.com/vllm-project/vllm/pull/26377)、[#27289](https://github.com/vllm-project/vllm/pull/27289) | ✅ 已合 | Mamba2 / Mamba1 的自动前缀缓存、缓存粒度可配（tracking issue [#26201](https://github.com/vllm-project/vllm/issues/26201) 里勾为完成） |
| [#26807](https://github.com/vllm-project/vllm/pull/26807) | ✅ 已合 | GDN 的 `all` 模式（在块边界物化循环状态） |
| [**#30877**](https://github.com/vllm-project/vllm/pull/30877) | ✅ 2026-01-23 | **块对齐（align）模式的 Mamba 前缀缓存**：调度按 block_size 对齐切分，让状态能映射到块哈希上；不改底层 kernel，兼容投机解码/MTP/EAGLE。**这是整个混合模型前缀缓存的地基** |
| [#28047](https://github.com/vllm-project/vllm/pull/28047) | ✅ 2025-12-25 | Mamba2 的 prefix cache 块在不再被运行中请求需要时**允许被淘汰**（之前被钉住，导致缓存涨不上去） |
| [#35219](https://github.com/vllm-project/vllm/pull/35219) | ✅ 2026-03-10 | 修"SSM cache block 释放数为 0"的 bug |
| [#40172](https://github.com/vllm-project/vllm/pull/40172)、[#42406](https://github.com/vllm-project/vllm/pull/42406)、[#42792](https://github.com/vllm-project/vllm/pull/42792) | ✅/🟠 | align 模式的后处理改用 GPU kernel（避免 CPU-GPU 同步）、Model Runner V2 支持 |

### 存（B1–B4）

| PR / Issue | 状态 | 干了什么 | 对应能力块 |
|---|---|---|---|
| [**#37898**](https://github.com/vllm-project/vllm/pull/37898) | ✅ 2026-06-10 | **"Marconi-style 准入"**。只改 3 个文件、+89 行：coordinator 里算出 `longest_hit_length - hit_length`（= 有共享前缀但 SSM 状态没跟上），调度器在 `_mamba_block_aligned_split` 里判断该值 `>= block_size` 就把本次 prefill 的 chunk 切到共享前缀结束处，让状态在块边界被物化。**用"看"代替论文的"猜"** | **B2 + B3** |
| [**#47782**](https://github.com/vllm-project/vllm/pull/47782) | ✅ 2026-07-13 | 把它从"调度器特判"重构成通用机制：`find_longest_cache_hit` 直接返回第三个值，并引入 `reachable_boundaries`（prompt 尾部边界 + 共享前缀 junction），让稀疏保留和 Marconi 准入统一。**9 个文件** | B2 + B3 + 稀疏保留 |
| [#43447](https://github.com/vllm-project/vllm/pull/43447)、[#45845](https://github.com/vllm-project/vllm/pull/45845) | ✅ | retention interval 机制；并让 Mamba / linear attention 也遵守它 | 稀疏保留 |
| [#52216](https://github.com/vllm-project/vllm/pull/52216) | ✅ 2026-08-17 | `prefix_cache_retention_interval` 升级为正式参数，**默认 0**（最省内存的稀疏保留） | 稀疏保留 |
| [#55760](https://github.com/vllm-project/vllm/pull/55760) | ✅ 2026-09-08 | 修默认值坑：稀疏保留 + EAGLE 会让 Mamba 组几乎永不命中（生产报过 0 命中），改成按是否显式设置来区分处理 | D2 |
| [#52789](https://github.com/vllm-project/vllm/pull/52789) | ✅ 2026-08-22 | 内部 prefill checkpoint，省掉"为了存档而额外计算"的开销，**TTFT 改善 9%~25%** | B4 |
| [#50551](https://github.com/vllm-project/vllm/pull/50551) | 🟠 **open** | 把"最后一个 decode token"的状态也保留下来（Marconi 的 last decoded state），只在稀疏保留路径下补齐 | **B1（缺）** |
| [#55697](https://github.com/vllm-project/vllm/issues/55697) + [#55873](https://github.com/vllm-project/vllm/pull/55873)/[#55875](https://github.com/vllm-project/vllm/pull/55875)/[#55876](https://github.com/vllm-project/vllm/pull/55876) | 🟠 open | 另一条路线：应用用 `<|mamba_checkpoint|>` 显式声明边界 + 调度器 Producer/Consumer 协调；声称 L40S 上吞吐 +110%、kernel 7.58× | B2/B3 替代路线 |

### 删（C1–C3）

| PR / Issue | 状态 | 干了什么 | 对应能力块 |
|---|---|---|---|
| [**#23641**](https://github.com/vllm-project/vllm/issues/23641) | ❌ **关闭（stale）** | RFC：`Frequency and Cost Aware Eviction Policy for Prefix Caching`——`retention benefit = freq × compute_cost`，`cost = size^α`（α=2）。**思路和论文几乎一样**；2025-12-10 被 stale bot 标记，2026-01-10 自动关闭，**全程只有 2 条 bot 评论，没有一个 maintainer 回复** | C1（最接近的一次） |
| [#27039](https://github.com/vllm-project/vllm/pull/27039) | ✅ 2025-11-12 | **ARC 驱逐策略**（自动在"新旧"和"频次"之间切换）——但只在 **CPU offload 路径** `vllm/v1/kv_offload/cpu/` | C3 的近似（通用维度，非混合感知） |
| [#49114](https://github.com/vllm-project/vllm/pull/49114) | ✅ 2026-07-29 | **`CachePolicyFactory`**：外部包可以按名字注册自己的 `CachePolicy`（同样只在 offload 路径） | 驱逐框架（offload 层） |
| [#40270](https://github.com/vllm-project/vllm/pull/40270) | 🟠 **open（needs-rebase）** | **GPU BlockPool 可插拔驱逐**：指明现在 `BlockPool` 用单队列 LRU（`FreeKVCacheBlockQueue`），没有频率信号、会被 scan 污染；提出 `GPUCachePolicy` + lru/two_queue 等策略 | **驱逐框架（GPU 层）—— 关键闸门** |
| [#37825](https://github.com/vllm-project/vllm/pull/37825) / [#38984](https://github.com/vllm-project/vllm/pull/38984) / [#50422](https://github.com/vllm-project/vllm/pull/50422) / [#47837](https://github.com/vllm-project/vllm/pull/47837) / [#55403](https://github.com/vllm-project/vllm/pull/55403) | 🟠 open | T-LRU / SLRU / Session-Aware / Priority-Aware / Block 优先级（机会性命中） | C3 的近亲，全部未合 |
| — | ❌ 无 | **性价比（覆盖长度 ÷ 占用内存）评分** | **C1/C2 完全空白** |

### 集成（D1–D3）

| PR / Issue | 状态 | 干了什么 |
|---|---|---|
| #42524 / #44243 / #46384 一系 | ✅ | 混合模型的部分命中（FA 命中但 Mamba 未命中时的处理） |
| [#52527](https://github.com/vllm-project/vllm/pull/52527) | 🟠 open | 上报"因为缺 checkpoint 而丢掉的共享前缀 token 数"（度量） |
| [#37003](https://github.com/vllm-project/vllm/issues/37003) | 🟠 open | RFC：带优先级的 KV 保留 API（无人推进） |
| #53912 / #57266 / #57267 / #43587 / #51250 / #54504 / #52317 / #53749 / #45238 | 🟠 open | 各类组合场景 bug（MTP 输出损坏、`--mamba-block-size` 组合崩、多模态增量请求、0% 命中、断言阻塞等） |

## 3.3 vllm-ascend：每个 PR 干了什么

| PR / Issue | 状态 | 干了什么 | 对应能力块 |
|---|---|---|---|
| [**#7103**](https://github.com/vllm-project/vllm-ascend/pull/7103) | ✅ 2026-03-15 | 把 align 模式的 hybrid 前缀缓存搬到昇腾（跟随上游 [#30877](https://github.com/vllm-project/vllm/pull/30877)）。**PR 自述三个坑**：① align + PD 分离当时不支持；② 多机并行时 block_size 被抬到 **2048**（Qwen3.5-35B-A3B、TP2），短于该长度的前缀永不缓存；③ 打了 triton kernel 补丁绕开昇腾上的 bug | A1–A3 |
| [#7372](https://github.com/vllm-project/vllm-ascend/pull/7372) / [#9514](https://github.com/vllm-project/vllm-ascend/pull/9514) | ✅ | 310P 上的 Mamba cache / Prefix Mamba Cache | 基础 |
| [#7796](https://github.com/vllm-project/vllm-ascend/pull/7796) / [#7814](https://github.com/vllm-project/vllm-ascend/pull/7814) | ✅ 2026-03-31 | layerwise connector 支持 Mamba prefill 前缀缓存 | A1 + PD |
| [#9533](https://github.com/vllm-project/vllm-ascend/pull/9533) | ✅ 2026-06-03 | AscendStore 里的 Hybrid & Mamba align 前缀缓存 | D4/存储 |
| [#10009](https://github.com/vllm-project/vllm-ascend/pull/10009) | ✅ 2026-06-16 | PD 场景下的部分组缓存（Mamba 组没命中时，不要放弃 FA 组已命中的部分） | **D1** |
| [#10393](https://github.com/vllm-project/vllm-ascend/pull/10393) | ✅ 2026-07-02 | `AscendStoreCoordinator`：按组的 reachable mask 协调（对齐上游 Mooncake 的设计） | D4 |
| [#12038](https://github.com/vllm-project/vllm-ascend/pull/12038) | ✅ 2026-07-15 | 修 v0.23.0 上"EAGLE 查找把 hybrid 命中长度撑大"的问题 | D2 |
| [#15406](https://github.com/vllm-project/vllm-ascend/pull/15406) / [#16562](https://github.com/vllm-project/vllm-ascend/pull/16562) | ✅ 2026-09 | MRv2 / 310P 的前缀缓存适配、MRV2 Mamba block-table 容量修复 | D2 |
| [#15763](https://github.com/vllm-project/vllm-ascend/pull/15763) / [#15674](https://github.com/vllm-project/vllm-ascend/pull/15674) | ✅ 2026-09-05 | AscendStore hybrid Mamba cache 释放修复 | D4 |
| [**#14468**](https://github.com/vllm-project/vllm-ascend/pull/14468) | 🟠 **open**（创建 2026-08-18，只有 bot 评论） | **Marconi 准入的 backport（针对 `releases/v0.25.1rc`）**：v0.25.1 上游用的是实例属性 `self.num_uncached_common_prefix_tokens`，Ascend 覆写 `find_longest_cache_hit` 时漏了赋值 → 调度器永远读到 0 → 准入永不触发。**加 1 行修复**，并附实测（见 3.5） | **B2 + B3** |
| [**#13300**](https://github.com/vllm-project/vllm-ascend/pull/13300) | 🟠 **open**（merge-conflicts） | AscendStore 适配上游 [#47782](https://github.com/vllm-project/vllm/pull/47782) 的 `reachable_boundaries` 接口——不改的话新接口下抛 `TypeError`，回退逻辑会把**共享前缀边界和 retention interval 一起丢掉** | B3 / D4 |
| [#7125](https://github.com/vllm-project/vllm-ascend/pull/7125) / [#7469](https://github.com/vllm-project/vllm-ascend/pull/7469) | 🟠 open（merge-conflicts） | NPU offloader 支持 ARC 驱逐（跟随上游 [#27039](https://github.com/vllm-project/vllm/pull/27039)） | 驱逐框架（offload 层） |
| [#16449](https://github.com/vllm-project/vllm-ascend/pull/16449) / [#16528](https://github.com/vllm-project/vllm-ascend/pull/16528) | 🟠 open | PD + MTP 下的 hybrid/Mamba 前缀缓存修复、细粒度命中 | D2 |
| [#12176](https://github.com/vllm-project/vllm-ascend/pull/12176) | ❌ 关闭未合 | hybrid 部分前缀命中（"要等 0.25.0 之后"），最后因冲突关闭 | D1 |
| [#16639](https://github.com/vllm-project/vllm-ascend/issues/16639) | 🟠 open issue | `vllm:prefix_cache_hits_total` 恒为 0（缓存明明在工作） | **D3** |
| [#11711](https://github.com/vllm-project/vllm-ascend/issues/11711) | 🟠 open issue | Qwen3.5-35B 在 910B 上开前缀缓存后反而比 Qwen3-30B 慢近 2 倍 | D2/性能 |

## 3.4 覆盖矩阵

| 能力块 | 上游 vLLM | Ascend |
|---|---|---|
| A1 KV+SSM 绑定管理 | ✅ #30877 一系 | ✅ #7103 等 |
| A2 前缀匹配 | ✅ block hash + coordinator | ✅ 跟随（monkey-patch） |
| A3 块对齐存储 | ✅ #30877 | ✅ #7103（但 block_size 会膨胀） |
| B1 对话尾部状态 | 🟡 稀疏保留只保 replay 边界；通用化 🟠 #50551 | 🟡 跟随 |
| B2 共享前缀识别 | ✅ #37898 | **v0.23.0 ❌；main ✅；v0.25.1rc ❌（🟠 #14468）** |
| B3 分叉点物化状态 | ✅ #37898 + #47782 | 🟡 本地 ✅；AscendStore/PD ❌（🟠 #13300） |
| B4 取中间状态 | ✅ 块对齐 + #52789 | 🟡 跟随 |
| C1 性价比评分 | ❌ 空白 | ❌ 空白 |
| C2 α 自适应 | ❌ 空白 | ❌ 空白 |
| C3 中间节点淘汰 / 不刷祖先 | 🟡 近似（#28047、🟠 #40270/#55403） | 🟡/❌ 跟随 |
| D1 部分命中 | 🟡 有实现，仍有 bug | 🟡 #10009；#12176 关闭 |
| D2 组合场景 | 🟡 一堆 open bug | 🟡 一堆 open PR |
| D3 指标 | 🟠 #52527 | ❌ #16639 只有 issue |
| 驱逐框架（GPU） | 🟠 #40270（未合） | 🟠 跟随 |
| 驱逐框架（offload） | ✅ #27039 + #49114 | 🟠 #7125/#7469 未合 |

## 3.5 关键代码事实（已核实）

| 事实 | 证据 |
|---|---|
| vLLM **v0.25.1** 里 Marconi 准入是**实例属性**形式 | 该版本 `kv_cache_coordinator.py:737` 有 `self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length` |
| vLLM **v0.23.0**（2026-06-15 发布）**没有** Marconi 准入 | 该版本 `kv_cache_coordinator.py` 里搜不到 `num_uncached_common_prefix_tokens` / `longest_hit_length` |
| vLLM v0.23.0 的 retention interval **只对 sliding-window 生效** | 源码注释原文：`Retention only sparsifies sliding-window checkpoints for now; ... TODO: Support Mamba/linear attention.` |
| Ascend **releases/v0.23.0** 分支**没有** Marconi 准入 | 该分支 `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`（526 行）里搜不到相关字段；`find_longest_cache_hit` 只有两个定义位置，无 `longest_hit_length` |
| Ascend **main** 已接通 | main 上同一文件第 423 行：`return cache_hit_blocks, hit_length, longest_hit_length - hit_length`（第 330/398 行计算 `longest_hit_length`） |
| Ascend **#14468 的实测数据**（910C/A3，Qwen3.6-27B，TP=4，max-num-seqs=32，max-num-batched-tokens=20480，开 `--enable-prefix-caching`） | `releases/v0.25.1rc`：8192/5120/4 → TTFT 982.9ms、命中 0.0%；8192/7168/4 → 979.2ms、0.0%；20480/9216/8 → 2333.9ms、0.0%；20480/12288/8 → 2347.1ms、0.0%；32768/13824/4 → 3242.6ms、12.2%。**修复后**：737.9ms/34.3%；406.4ms/**69.2%**；1692.2ms/34.7%；1434.1ms/48.4%；2508.0ms/33.8% |

**结论**：

> **上游 vLLM：准入已落地（近似实现，剩 #50551 等尾巴）；驱逐的"性价比评分"完全空白，GPU 侧驱逐至今是单队列 LRU（#40270 卡着）。**
>
> **vllm-ascend：v0.23.0 基线里连 Marconi 准入都没有（因为上游 v0.23.0 就没有）；main 上已接通；offload 侧的 ARC 还挂在两个 merge-conflicts 的 PR 上；混合感知的驱逐是 0。**

---

# 第四部分 · 在 Ascend 0.23.0 上实现完整 Marconi

## 4.1 起点盘点（v0.23.0 有什么、没有什么）

**有**：
- `AscendHybridKVCacheCoordinator`（继承上游 `HybridKVCacheCoordinator`），已处理 Mamba 组、EAGLE lookahead margin、PD 分离下的 Mamba 特殊逻辑、16k 对齐（带压缩比的模型）；
- align 模式的 hybrid/Mamba 前缀缓存（#7103 及其后续修复）；
- AscendStore 的 hybrid 缓存与协调（#9533、#10393）；
- 部分组缓存（#10009）。

**没有**：
- Marconi 准入（共享前缀识别）——上游 v0.23.0 就没有这段代码，Ascend 分支自然也没有；
- Mamba 的稀疏保留（上游 v0.23.0 的 retention 只对 sliding-window 生效，源码里是 TODO）；
- GPU 侧可插拔驱逐（`BlockPool` 单队列 LRU）；
- 任何"性价比"驱逐评分；
- 可靠的命中率指标（#16639）。

## 4.2 板块与实现顺序

> 依赖关系：**P0 → P1 → P2** 是"存"，彼此基本独立；**P3 → P4** 是"删"，P4 依赖 P3；**P5** 可与 P3/P4 并行。

| 阶段 | 板块 | 依赖 | 优先级 |
|---|---|---|---|
| **P0** | 度量与基线 | — | 必做，最先 |
| **P1** | 准入：共享前缀识别（B2/B3） | P0 | **最高性价比** |
| **P2** | 准入：尾部状态 + Mamba 稀疏保留（B1/B4） | P0、P1 | 高 |
| **P3** | 驱逐框架：GPU 侧策略插槽（C 的前置） | P0 | 中高 |
| **P4** | 驱逐策略：性价比评分（C1/C2/C3） | P3 | 中（价值高、风险也高） |
| **P5** | 组合场景加固（D1/D2/D4） | 视具体项 | 持续 |

## 4.3 每个板块：做什么、改哪里、怎么验证、预期收益

### P0 · 度量与基线（先能看见）

**做什么**：让"前缀缓存到底命中了多少"变得可观测。

**依据**：Ascend [#16639](https://github.com/vllm-project/vllm-ascend/issues/16639)（`vllm:prefix_cache_hits_total` 恒 0）、[#15745](https://github.com/vllm-project/vllm-ascend/pull/15745)（BalanceScheduler 不报指标）、上游 [#52527](https://github.com/vllm-project/vllm/pull/52527)（上报因缺 checkpoint 丢掉的共享前缀 token）。

**怎么做**：
1. 修 hit/miss 计数与上报路径，确保 Mamba 组也计入；
2. 加一个"token 命中率"指标（**注意：请求命中数和 token 命中数是两个不同的量**，论文用的是后者）；
3. 固定一套可复现的压测脚本（固定 prompt、前缀长度、并发、输入长度），作为后续所有验证的基线。

**怎么验证**：
- 同一个长 system prompt 发 ≥3 次，token 命中率应随次数上升（而不是恒 0）；
- 指标数值与日志里的 prefix cache hit rate 对得上；
- 关掉 `--enable-prefix-caching` 时指标归零。

**预期收益**：不直接产生性能，但没有它后面每一步都无法判断是否生效。

---

### P1 · 准入：共享前缀识别（**投入产出比最高**）

**做什么**：把上游 [#37898](https://github.com/vllm-project/vllm/pull/37898)（+ [#47782](https://github.com/vllm-project/vllm/pull/47782) 的接口重构）这套"看到 FA 命中、SSM 没跟上 → 在块边界补一份状态"的能力带到 0.23.0。

**两条路线（二选一）**：

| 路线 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 升级基线** | 把 Ascend 基线从 vLLM v0.23.0 升到 ≥ v0.25.1（最好 main），直接获得准入 + Mamba retention | 一次性拿到两块能力，长期维护成本低 | 升级量大，Ascend 的 monkey-patch 需要全面重测 |
| **B. 在 0.23.0 上 backport** | 把 `kv_cache_coordinator` 的返回值 + 调度器的 chunk 切分逻辑移植过来，并让 `AscendHybridKVCacheCoordinator.find_longest_cache_hit` 把该值带出来 | 改动局部、风险可控、见效快 | 需要自己维护一份 backport；上游后续改动会持续产生冲突 |

**改动点**（对应 #37898 的 3 个文件 + Ascend 的覆写层）：
1. coordinator：在按组求最长命中时，**额外记录"最长的单组命中长度"**，最终返回 `最长组命中 - 实际可用命中`；
2. scheduler：`_mamba_block_aligned_split` 里，若该值 `>= block_size`，把本次 prefill 的 chunk 切到共享前缀结束处；
3. **Ascend 侧**：`AscendHybridKVCacheCoordinator.find_longest_cache_hit` 必须把这个值**带出来**（这正是 [#14468](https://github.com/vllm-project/vllm-ascend/pull/14468) 修的那个漏赋值——Ascend 自己覆写了这个函数，很容易漏）。

**怎么验证**：
- **单测**：构造"FA 组命中 N 块、Mamba 组命中 0 块"的场景，断言返回值 > 0；再断言调度器会把 chunk 切在共享前缀边界上；
- **E2E**：同一个长 system prompt 发 3 次以上（第 1 次预热、第 2 次识别、第 3 次命中），看 token 命中率从 ~0% 升到 30%+，TTFT 相应下降；
- **回归**：纯 Transformer 模型上行为不变（论文说三套系统在纯 Transformer 上表现一致）。

**预期收益**（Ascend 实测，来自 #14468，910C/A3、Qwen3.6-27B、TP=4）：

| 输入 / 前缀 / 并发 | 修复前 TTFT / 命中率 | 修复后 TTFT / 命中率 |
|---|---|---|
| 8192 / 5120 / 4 | 982.9 ms / 0.0% | **737.9 ms / 34.3%** |
| 8192 / 5120 / 8 | 875.5 ms / 17.1% | 758.2 ms / 34.6% |
| 8192 / 7168 / 4 | 979.2 ms / 0.0% | **406.4 ms / 69.2%** |
| 8192 / 7168 / 8 | 673.0 ms / 50.9% | 426.5 ms / 69.4% |
| 20480 / 9216 / 8 | 2333.9 ms / 0.0% | **1692.2 ms / 34.7%** |
| 20480 / 12288 / 8 | 2347.1 ms / 0.0% | **1434.1 ms / 48.4%** |
| 32768 / 13824 / 4 | 3242.6 ms / 12.2% | 2508.0 ms / 33.8% |

> 这是**昇腾上已验证过的、最大的一块确定性收益**：多处从 0% 命中直接变成 30%~69%，TTFT 降 20%~59%。

---

### P2 · 准入：对话尾部状态 + Mamba 稀疏保留

**做什么**：让"多轮对话"这条路径也能复用——把上一个回答结尾处的状态保留下来；同时补上上游在 v0.23.0 里还缺的"Mamba 的稀疏保留"。

**依据**：[#50551](https://github.com/vllm-project/vllm/pull/50551)（保留 decode checkpoint，open）、[#47782](https://github.com/vllm-project/vllm/pull/47782)（`reachable_boundaries`）、[#43447](https://github.com/vllm-project/vllm/pull/43447)/[#45845](https://github.com/vllm-project/vllm/pull/45845)（retention for Mamba）、v0.23.0 源码里的 `TODO: Support Mamba/linear attention`。

**怎么做**：
1. 把 `reachable_boundaries` 的语义带进来：**prompt 尾部边界 + 共享前缀 junction** 两类位置都要保留；
2. 每个请求结束后，把"最后一个被调度器对齐的 decode 位置"的状态也纳入保留；
3. **注意 retention 的默认值**：上游在 [#52216](https://github.com/vllm-project/vllm/pull/52216) 把默认改成 0（最省内存），随后又因 Mamba+EAGLE 组合出问题在 [#55760](https://github.com/vllm-project/vllm/pull/55760) 打了补丁——**移植时要一起带上，否则会出现"缓存组永不命中"**。

**怎么验证**：
- 多轮对话 workload：第 2 轮起，除新增片段外的历史应全部命中；
- Mamba+EAGLE/MTP 组合：确认 Mamba 组不是 0 命中（这正是 #55760 修的问题）；
- 稀疏保留打开时，确认关键边界没有被误删。

**预期收益**：论文里 LMSys（长输出多轮对话）相对 vLLM+ 的 token 命中率提升为 **4.5×**；ShareGPT（短输出）为 **7.3×**。注意这两个数包含准入的整体效果，不能单独归给尾部状态。

---

### P3 · 驱逐框架：把 GPU 侧"扔谁"变成可插拔

**做什么**：现在 HBM 里块的释放顺序是固定的单队列 LRU，先把这个顺序参数化。

**依据**：[#40270](https://github.com/vllm-project/vllm/pull/40270)（GPU BlockPool 可插拔驱逐，open/needs-rebase）、[#49114](https://github.com/vllm-project/vllm/pull/49114)（offload 层的 `CachePolicyFactory` 已合，可参照它的接口设计）。

**怎么做**：
1. 在 Ascend 的 block pool / free queue 上引入策略接口（对齐上游 #40270 的 `GPUCachePolicy` 命名与语义，**便于将来回馈上游**）；
2. 实现两个策略：`lru`（默认，行为完全不变）+ 一个新的"分数策略"占位；
3. 保证开关默认关闭。

**怎么验证**：
- `lru` 路径下，与改动前的命中率、TTFT 完全一致（回归）；
- 人为切换到一个"总是优先淘汰最长块"的假策略，命中率应发生可观测变化（证明插槽真的生效）。

**预期收益**：不直接产生性能，但没有它，驱逐策略无处安放。

---

### P4 · 驱逐策略：性价比评分

**做什么**：实现论文第二部分的评分与淘汰。

**依据**：论文 4.2 节与附录 A.1；社区无实现（同类 RFC [#23641](https://github.com/vllm-project/vllm/issues/23641) 被 stale bot 关闭）。

**最小可行版本**（不需要完整复刻 FLOP 公式）：

1. **记账**：给每个缓存条目（或块组）记一个整数——**"它比父节点多代表了几个 token"**；
   - 对纯 KV 块，这个值恒定；
   - 对带 SSM 状态的条目，这个值正好反映"固定开销摊开了多少"。
2. **性价比** = `增量 token 数 ÷ 占用字节数`；
3. **分数** = `归一化的新旧 + α × 归一化的性价比`；
4. **候选规则**：孩子数 ≤ 1 的节点才可淘汰；**多孩子节点豁免**（这是保护公共骨架的关键，不要省）；
5. **α**：先做离线调参；在线自适应（论文的 bootstrap + 回放网格搜索）可作为第二步。

**怎么验证**（**必须先离线，再在线**）：

| 步骤 | 做法 | 看什么 |
|---|---|---|
| ① 离线复现 | 用论文 artifact 里的 trace（LMSys / ShareGPT / SWE-Bench，含 token 级信息）+ 策略仿真脚本，跑 LRU vs 评分策略 | token 命中率、FLOP saved、按长度分桶的命中率变化 |
| ② 在线 A/B | 同一 trace、同一缓存大小，跑 lru 与评分策略 | token 命中率、P50/P95 TTFT、短序列命中率的下降幅度 |
| ③ 回归 | 纯 Transformer、缓存宽裕、短序列场景 | 应当**没有差异**（退化成 LRU） |
| ④ 边界 | 缓存从宽到紧扫一遍 | 确认峰值出现在"中等竞争"，与论文一致 |

**预期收益**（论文在 NVIDIA 上的实测，**Ascend 上尚未验证**）：
- 相对 LRU：token 命中率 **+19.0% ~ +219.7%**；
- SWEBench：命中率 **16.4% → 32.7%**，**FLOP saved +90.3%**；
- 代价：短序列（<7K）命中率最多 **-3.0%**，P5 TTFT 差 6.3%（**绝对 2.1ms**）；换来长序列 **+25.5%**、P50/P95 改善 **13.4% / 22.0%**。

---

### P5 · 组合场景加固

按现成 issue/PR 推进，优先级由你们的业务场景决定：

| 场景 | 对应项 |
|---|---|
| PD 分离 / 外部存储 | 昇腾 [#13300](https://github.com/vllm-project/vllm-ascend/pull/13300)（`reachable_boundaries` 适配，open，merge-conflicts）、[#12781](https://github.com/vllm-project/vllm-ascend/pull/12781)、[#16044](https://github.com/vllm-project/vllm-ascend/pull/16044) |
| MTP / EAGLE / 投机解码 | 昇腾 [#16449](https://github.com/vllm-project/vllm-ascend/pull/16449)、[#16528](https://github.com/vllm-project/vllm-ascend/pull/16528)、[#16258](https://github.com/vllm-project/vllm-ascend/pull/16258) |
| block size 膨胀导致短前缀永不命中 | [#7103](https://github.com/vllm-project/vllm-ascend/pull/7103) 自述问题、[#15601](https://github.com/vllm-project/vllm-ascend/pull/15601)、上游 [#33194](https://github.com/vllm-project/vllm/pull/33194) |
| 部分命中与数值一致性 | 昇腾 RFC [#16451](https://github.com/vllm-project/vllm-ascend/issues/16451)（3 个 cold/warm 漂移根因，1 修 2 未修） |
| MRv2 / 310P | 已合 [#15406](https://github.com/vllm-project/vllm-ascend/pull/15406)、[#16562](https://github.com/vllm-project/vllm-ascend/pull/16562) |

**验证方式**：把这些场景做成 nightly 用例（固定模型 + 固定 trace + 固定并发），每次改动跑一遍。

## 4.4 风险与依赖

| 风险 | 说明 | 缓解 |
|---|---|---|
| **基线版本** | 0.23.0 的 vLLM 基线本身没有准入代码，backport 会长期产生冲突 | 优先评估"升级基线"路线；若必须留在 0.23.0，把 backport 做成一个小而清晰的 patch，并加回归测试 |
| **Ascend 覆写层** | Ascend 用 monkey-patch 覆写了 coordinator，**上游加的任何新返回值都很容易被漏掉**（#14468 就是这么来的） | 在覆写处加断言/单测：上游接口变了就必须显式处理 |
| **记账成本** | "每个条目代表多长"是新状态，会影响 KV cache manager 的核心路径 | 先只做"增量 token 数"这一个整数；先离线验证收益再上线 |
| **短序列回退** | 驱逐策略必然牺牲短序列命中率 | 默认关闭；上线前用真实 trace 评估 P5/P50 的变化，确认绝对值可接受 |
| **度量缺失** | 没有可靠指标就无法判断是否生效 | P0 先行 |
| **数值一致性** | 部分命中在昇腾上仍有 cold/warm 漂移（RFC #16451 还剩 2 个根因） | 与 P5 一起做，作为验收门槛 |

## 4.5 最终理想收益（区分"已验证"和"预期"）

| 层次 | 收益 | 证据强度 |
|---|---|---|
| **准入在 Ascend 上** | 命中率 0% → **34.3%~69.4%**；TTFT 从 982.9→737.9ms、979.2→406.4ms、2333.9→1692.2ms、3242.6→2508.0ms | **已实测**（昇腾 910C/A3，Qwen3.6-27B，TP=4；来源 [#14468](https://github.com/vllm-project/vllm-ascend/pull/14468)） |
| **准入 + 驱逐（上游论文）** | Token 命中率 **4.5× / 7.3× / 34.4×**（LMSys / ShareGPT / SWEBench，vs vLLM+）；P95 TTFT 降幅再多 **36.1% / 71.1% / 46.8%**（275.4 / 103.3 / 617.0 ms） | **论文实测（NVIDIA 8×A100，7B 混合模型 + Jamba-1.5-Mini）** |
| **单看驱逐（上游论文）** | 相对 LRU：token 命中率 **+19.0%~+219.7%**；SWEBench 16.4%→32.7%、**FLOP saved +90.3%** | **论文实测（NVIDIA）** |
| **完整 Marconi 在 Ascend 上** | **无法直接给出数字**。合理预期：准入贡献大头（已被 #14468 验证），驱逐在"长度分布宽 + 缓存中等紧张 + SSM 状态常驻多"的 workload 上再叠加几十个百分点 | **推断**——必须自己实测 |

> ⚠️ **不要把论文的 34.4× 直接当成 Ascend 的预期**：数据集、模型、硬件、缓存配置都不同。可复用的只是"趋势"——SSM 层占比越高、状态维度越大、序列长度分布越宽、缓存竞争越中等，收益越大。

---

# 附录 · PR / Issue 索引

| 编号 | 仓库 | 状态 | 一句话 |
|---|---|---|---|
| [#30877](https://github.com/vllm-project/vllm/pull/30877) | vllm | ✅ 2026-01-23 | align 模式 Mamba 前缀缓存（地基） |
| [#28047](https://github.com/vllm-project/vllm/pull/28047) | vllm | ✅ 2025-12-25 | Mamba2 prefix cache 块允许被淘汰 |
| [#35219](https://github.com/vllm-project/vllm/pull/35219) | vllm | ✅ 2026-03-10 | 修 SSM cache block 释放为 0 |
| [#37898](https://github.com/vllm-project/vllm/pull/37898) | vllm | ✅ 2026-06-10 | **Marconi-style 准入**（存） |
| [#47782](https://github.com/vllm-project/vllm/pull/47782) | vllm | ✅ 2026-07-13 | Marconi 准入 + 稀疏保留统一（`reachable_boundaries`） |
| [#52216](https://github.com/vllm-project/vllm/pull/52216) | vllm | ✅ 2026-08-17 | retention interval 成正式参数，默认 0 |
| [#55760](https://github.com/vllm-project/vllm/pull/55760) | vllm | ✅ 2026-09-08 | 修 Mamba+EAGLE 下稀疏保留 0 命中 |
| [#52789](https://github.com/vllm-project/vllm/pull/52789) | vllm | ✅ 2026-08-22 | 内部 prefill checkpoint，TTFT -9~25% |
| [#50551](https://github.com/vllm-project/vllm/pull/50551) | vllm | 🟠 open | 保留 decode checkpoint（对话尾部状态） |
| [#55697](https://github.com/vllm-project/vllm/issues/55697) + [#55873](https://github.com/vllm-project/vllm/pull/55873)/[#55875](https://github.com/vllm-project/vllm/pull/55875)/[#55876](https://github.com/vllm-project/vllm/pull/55876) | vllm | 🟠 open | 应用显式声明 checkpoint 边界 |
| [#23641](https://github.com/vllm-project/vllm/issues/23641) | vllm | ❌ 关闭(stale) | 频次+成本感知驱逐 RFC（最接近论文的一次） |
| [#27039](https://github.com/vllm-project/vllm/pull/27039) | vllm | ✅ 2025-11-12 | ARC 驱逐（仅 CPU offload 路径） |
| [#49114](https://github.com/vllm-project/vllm/pull/49114) | vllm | ✅ 2026-07-29 | `CachePolicyFactory`（offload 层可插拔） |
| [#40270](https://github.com/vllm-project/vllm/pull/40270) | vllm | 🟠 open | **GPU BlockPool 可插拔驱逐（关键闸门）** |
| [#55403](https://github.com/vllm-project/vllm/pull/55403) | vllm | 🟠 open | Block 优先级 / 机会性命中 |
| [#52527](https://github.com/vllm-project/vllm/pull/52527) | vllm | 🟠 open | 丢失复用量指标 |
| [#26201](https://github.com/vllm-project/vllm/issues/26201) | vllm | 🟠 open | 混合模型前缀缓存 tracking issue |
| [#7103](https://github.com/vllm-project/vllm-ascend/pull/7103) | ascend | ✅ 2026-03-15 | 昇腾 hybrid align 前缀缓存 |
| [#9533](https://github.com/vllm-project/vllm-ascend/pull/9533) | ascend | ✅ 2026-06-03 | AscendStore hybrid & Mamba align 前缀缓存 |
| [#10009](https://github.com/vllm-project/vllm-ascend/pull/10009) | ascend | ✅ 2026-06-16 | PD 下部分组缓存 |
| [#10393](https://github.com/vllm-project/vllm-ascend/pull/10393) | ascend | ✅ 2026-07-02 | AscendStoreCoordinator（按组 mask） |
| [#12038](https://github.com/vllm-project/vllm-ascend/pull/12038) | ascend | ✅ 2026-07-15 | 修 EAGLE 撑大 hybrid 命中 |
| [#14468](https://github.com/vllm-project/vllm-ascend/pull/14468) | ascend | 🟠 open | **v0.25.1rc 上打开 Marconi 准入（实测 0%→34~69%）** |
| [#13300](https://github.com/vllm-project/vllm-ascend/pull/13300) | ascend | 🟠 open | AscendStore 适配 `reachable_boundaries` |
| [#7125](https://github.com/vllm-project/vllm-ascend/pull/7125) / [#7469](https://github.com/vllm-project/vllm-ascend/pull/7469) | ascend | 🟠 open | NPU offloader 的 ARC 驱逐 |
| [#16449](https://github.com/vllm-project/vllm-ascend/pull/16449) / [#16528](https://github.com/vllm-project/vllm-ascend/pull/16528) | ascend | 🟠 open | PD+MTP 下 hybrid/Mamba 前缀缓存修复 |
| [#16639](https://github.com/vllm-project/vllm-ascend/issues/16639) | ascend | 🟠 open | 命中指标恒 0 |
| [#16451](https://github.com/vllm-project/vllm-ascend/issues/16451) | ascend | 🟠 open | 部分命中的 cold/warm 数值漂移 RFC |
