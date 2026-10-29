# GDN / MTP 的 "n+1 份 recurrent state" 问题：方案综述与 Ascend 落地选型

> **调研范围**：ReplaySSM、SpecLA、Bole、TreeWY、DeltaLog、DAMP、DASC 七个方案 + 2025–2026 年相关最新工作
> **调研目标**：找出一个适合在 **昇腾（Ascend NPU）** 上实现的方案，用于 GDN (Gated DeltaNet) 混合模型 + MTP/EAGLE 投机解码
> **成文日期**：2026-10-29　|　**状态**：初版（一手材料 = 论文全文 + GitHub PR/Issue 元数据，均已逐条核验，见文末"证据等级说明"）

---

## 目录与"需求覆盖"对照

| 你要求的字段 | 在哪里 |
|---|---|
| 论文地址 | §3 每节开头 `**论文**` + §9.1 汇总表 |
| 实现思路 | §3 每节 `核心思路` + §2 统一数学框架 |
| 解决什么问题 | §3 每节 `解决什么问题` |
| 有无开源代码仓（含链接） | §3 每节 `开源代码` + §9.3 |
| 实现情况（vllm / sglang / vllm-ascend） | §5（逐 PR 核验）+ §3 每节 `引擎落地` |
| vllm / ascend 的 PR 或 issue | §5.1 / §5.2 / §5.5 |
| Ascend 实现思路与可能遇到的问题 | **§6（重点）** + §3 每节 `Ascend 移植要点` |
| 其他最新/更好的方案 | §7（含 SketchSSM、ReaDy、Tandem、KVBuffer 等） |
| 选一个合适方案的建议 | §0.4 / §4 / §8 |

**阅读路径建议**：只想知道结论 → 读 §0 与 §8.2；要写方案 → 读 §2 + §6；要复现 → 读 §5.3 + §6.4。

---

## 0. 结论先行（TL;DR）

1. **"n+1 份 state" 不是架构的必然，而是当前实现的"物理表示"选择。** 2026 年 6–10 月这一批工作（ReplaySSM / TreeWY / Bole / SpecLA / DeltaLog）在互不知情的情况下收敛到了**同一个数学洞察**：
   > **不要把 dense recurrent state 当作"每个 token 一份"来存；把它存成"一个 checkpoint + 一段紧凑的 rank-1 因子日志"，在接受（commit）时用一次矩阵乘法重算唯一需要保留的状态。**
   这个变换对 GDN 是**代数精确**的（只是把 WY/UT 变换从 chunk 扩展到 verify 窗口 / 树），不是近似。

2. **树（tree）版本只需要一个三角求解**：把所有 draft 节点按 DFS 先序排列，祖先掩码使其严格下三角，于是
   `(I + diag(β)G) Ṽ = R` 一次前代 / 或 `D` 次矩阵乘（Neumann 级数）即可并行算出**所有**节点输出，无需任何中间 state。存储从 `(N+1)×d_k×d_v` 降到 `1×d_k×d_v + N×(d_k+d_v)`。

3. **工程上真正被验证、并已进生产的三条路线**（按成熟度排序）：
   | 路线 | 代表 | 落地状态（已核验） |
   |---|---|---|
   | checkpoint + 因子/输入日志（chain only） | **ReplaySSM**（上游叫 **RecoverSSM**，开关 `--use-replayssm`） | **Mamba2**：vLLM #48018 ✅、TensorRT-LLM #14203 ✅、FlashInfer #3975 ✅、SGLang ✅；**KDA**：vLLM #63045 ✅、TensorRT-LLM #16464 ✅、FlashInfer #4081 ✅、**vllm-ascend 已进 main**；**GDN**：SGLang ✅、FlashInfer #4081 ✅、TensorRT-LLM #16464 ✅，**vLLM 上游仍 open（#61309）**；**GLM-5.3-Flash / Kimi-K3 生产在用** |
   | 树上三角求解 / Neumann（tree，精确） | **TreeWY**、**Bole** | TreeWY = vLLM **私有 fork**（作者明言未上游）；Bole = 声称集成 SGLang 但**无公开代码** |
   | 拓扑感知的三路分流（chain 常驻 SRAM / tree 掩码 / 多链并行） | **SpecLA** | PyTorch+Triton 原型，未进引擎 |

4. **对 Ascend 的选型建议（详见 §6）**：
   - **不建议**第一版就做 tree：TreeWY 在 **vLLM** 上实测"非因果祖先掩码 ⇒ 掉出 CUDA Graph ⇒ 比不做树还慢"；Bole 在 SGLang 上也承认更宽的树"**还不是吞吐赢**"。**Ascend 的 ACL Graph 约束只会更严。**
   - **建议主路线**：`checkpoint + 因子日志 + 指针回滚`（即 ReplaySSM 的 chain 形态 + TreeWY 的 `mamba_state_commit="reconstruct"` 语义），**保持因果掩码**，可整图捕获、可落到固定 shape 的 ACL Graph。
   - **第二期**再上 tree：用 **Neumann 级数（D≤4 次 Cube matmul）** 替代三角求逆——这是 Ascend 最友好的形式（Cube 单元吃 matmul，不吃三角求解）。
   - **第三期**叠加 DAMP 式混合精度（FP16 关键 key channel + INT8 其余）与 DASC 式 checkpoint 压缩，把 HBM 占用打下来。

5. **Ascend 侧不是空白，而是"半落地"——这才是最好的切入点**：
   - **vllm-ascend main 已有 KDA 的 RecoverSSM 端到端路径**（`vllm_ascend/models/glm5next/recoverssm_config.py` 等）。
   - **GDN 的 checkpoint/replay PR 已经写完并测过**：[vllm-ascend#18765](https://github.com/vllm-project/vllm-ascend/pull/18765)，实测 *"48 value heads, K=V=128, MTP3 ⇒ SSM + records ≈ **3.188 MiB** vs **12 MiB** for four snapshots"*。
   - **RFC [vllm-ascend#19987](https://github.com/vllm-project/vllm-ascend/issues/19987)（2026-10-27，两天前）** 正是要把 GDN RecoverSSM 推进到 **MRV2**。
   - **真正的缺口**：① **图路径**（#18765 明说 "*Primitive graph support does not enable an engine graph path*"）；② **树**；③ **多卡/变长**；④ **checkpoint 自身的精度压缩**。
   ⇒ **不要从零开始造轮子，去补这四个缺口**（详见 §8.3）。

---

## 1. 问题定义：为什么 GDN / MTP 会需要 "n+1 份" 状态

### 1.1 GDN 的递推

Gated DeltaNet 每层每个 state head 维护一个 `d_k × d_v` 的矩阵状态 `S`，逐 token 更新：

```
S_t = α_t · S_{t-1} + β_t · (v_t − α_t · S_{t-1} k_t) k_t^T          (1)
```

- `α_t ∈ (0,1)`：标量衰减门（GDN 全 head 共享一个标量；KDA 是逐 key channel 的 `diag(exp(g_t))`）
- `β_t`：写入门（write strength）
- 括号项就是 **delta rule**：先把"状态已经预测出的对 `k_t` 的响应"减掉，再写入残差

三个致命性质：

1. **`S_t` 是整个前缀的有损摘要**，不像 KV cache 那样"按 token 可寻址"；
2. **更新几乎不可逆**（含 `1/β`、`α` 除法在数值上不可行）；
3. **单步算术强度极低**（一次 elementwise 衰减 + 一次 rank-1 外积），所以**状态读写是纯 HBM 带宽开销**。

### 1.2 投机解码为什么把它放大成 (n+1) 倍

投机解码的一次 target 前向要**并行**验证 `n` 个 draft token（EAGLE 树则 `N` 个节点），**在接受结果出来之前**，递推已经推进到了窗口末尾。而 commit 只需要被接受路径上的那**一个**状态。

于是"朴素正确"的做法是给每个 draft 位置都存一份完整状态的快照：

```
                ┌─────────────────────────────────────────┐
state 存储量 =  L × H × d_k × d_v × bytes  ×  (T+1)  [chain]
                                    ×  (N+1)  [tree, N = 节点数]
                └─────────────────────────────────────────┘
```

论文里把这个做法叫 **full-state snapshotting / `store_all`**，是 **vLLM 与 SGLang 当前的默认实现**（TreeWY §2 原文："*The current default in both vLLM (storeall in our figures) and SGLang*"）。

**这不是散文，是代码里的一行乘法**（vLLM main）：
```python
# vllm/model_executor/layers/mamba/abstract.py
num_speculative_blocks = 0 if use_replayssm else vllm_config.num_speculative_tokens

# vllm/v1/kv_cache_interface.py  —  MambaSpec
max_memory_usage_bytes = page_size_bytes * (1 + num_speculative_blocks)          # 默认
# align 模式：page_size_bytes * (2 + num_speculative_blocks + num_prefill_checkpoint_blocks)
```
> **`1 + num_speculative_blocks` = `1 + K` = n+1 份。** 这就是你要解决的问题在代码里的唯一出处。
> 打开 RecoverSSM 时 `num_speculative_blocks = 0`，**这一行就变成 `× 1`**。

补充两个易混淆点（来自 vLLM 实现）：
- **只有 SSM 矩阵状态被复制**；**卷积状态不复制** —— `gated_delta_net_state_shape` 用的是 `conv_kernel_size - 1 + num_spec`（一个加宽的 tensor，靠 `num_accepted_tokens` 原位回退）。
- 这些 slot 是 **block-table 的列**（`spec_state_indices_tensor` 形状 `[decode_bs, num_spec+1]`），采样后取第 `num_accepted_tokens - 1` 列。

### 1.3 具体量级（本报告按公开模型配置估算，非论文数字）

下表全部是 **B=256 序列的合计值**（除"状态/序列"一列）：

| 模型 | 线性层数 L | v heads H | d_k=d_v | 状态/序列（bf16） | 基线（1 份） | +3 draft（n+1=4） | 树 39 节点（N+1=40） |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 30 | 32 | 128 | **30 MiB** | 7.7 GiB | **30.7 GiB** | **≈307 GiB（单机不可行）** |
| Qwen3.5-397B-A17B | 45 | 64 | 128 | **90 MiB** | 23.1 GiB | **92 GiB** | **≈921 GiB（跨节点都紧张）** |
| Nemotron-3-Super-120B (Mamba-2, FP32) | — | — | — | 每层 **4 MiB**/序列（论文数字） | 每层 **1.07 GiB**（论文数字，B=256） | 每层 ×4 | — |

> 计算式：`state = L × H × d_k × d_v × bytes`，35B 为 `30×32×128×128×2B = 30 MiB`。树列 `(39+1)×30 MiB×256 ≈ 307 GiB`。

> 单卡 Ascend 800I A2 是 64 GB HBM。**光是 35B 模型 + 3 draft 的状态快照就要吃掉半张卡**，而这还是 bf16；GDN 状态通常要求 FP32 累加（DAMP 全文按 FP32 讨论，SGLang 下 Qwen3.6-35B / B=256 仅 recurrent state 就 **80 GB**）。
>
> 论文侧的实测印证：
> - Bole：Qwen3.5-122B-A10B、8 个 32 节点请求 → 快照 **36 GiB**；`B=... `时 **72 GiB**；而 Bole 的因子表示**< 1 GiB**。
> - TreeWY：每个 draft 位置一份快照，峰值 KV 占用被抬高 **2.0–2.45×**，直接压低并发与 TTFT。
> - ReplaySSM：把"每步快照"换成"checkpoint + 缓冲"，投机解码吞吐达到 vLLM 原生投机解码的 **≈1.8×**。

### 1.4 三个必须同时满足的约束

任何方案都要回答三件事（这也是 §3 横向对比的三条轴）：

| 约束 | 问题 | 方案给出的答案 |
|---|---|---|
| **C1 表示** | 不存 N+1 份 dense state，存什么？ | 紧凑因子（`k`, `v`, `β`, `α` 或伪值 `ṽ`） |
| **C2 回滚** | 被拒绝的分支怎么丢弃？ | 指针移动（日志截断）／重算唯一被接受状态 |
| **C3 并行** | 树的分支怎么在一个 kernel 里算完？ | 祖先掩码 → 严格下三角 → 一次求解 / D 次 matmul |

---

## 2. 统一数学框架：所有方案都是"同一件事"的不同工程切面

这一节是本报告的核心：**把五个方案写成同一个式子的不同特例**，后面 §3 的逐个分析都会引用这里的符号。

### 2.1 从递推到"伪值"（pseudo-value）

把 (1) 展开，定义**伪值**

```
ṽ_t = β_t (v_t − α_t S_{t-1} k_t)
```

递推就变成纯粹的**衰减累加**：

```
S_t = α_t S_{t-1} + ṽ_t k_t^T
     = g_t S_0 + Σ_{i≤t} (g_t/g_i) ṽ_i k_i^T ,      g_t = Π_{j≤t} α_j   (2)
```

**关键点**：给定 `S_0` 和所有 `ṽ_i`，**任意时刻的 state 和任意时刻的输出都能直接算出来，不需要走递推**。

```
o_t^T = g_t q_t^T S_0 + Σ_{i≤t} (g_t/g_i)(q_t^T k_i) ṽ_i^T                (3)
```

代价是：求 `ṽ_t` 本身依赖 `S_{t-1}`。把 (3) 代回 `ṽ` 的定义，得到**一个线性方程组**：

```
(I + diag(β) G) Ṽ = R
G[t,i] = (g_t/g_i)(k_t^T k_i) · [i ≺ t]        ← 严格下三角（i 是 t 的严格祖先）
R[t]   = β_t v_t − β_t g_t (S_0 k_t)
```

这就是 DeltaNet 的 **WY / UT 变换**（Yang et al., arXiv:2406.06484），原本用于 chunk 内并行；**TreeWY/Bole/SpecLA 做的事情就是：把"chunk"换成"整棵 draft 树"，把因果掩码换成祖先掩码。** 数学形式完全不变，额外的只有两点：

- **顺序**：节点必须按 **DFS 先序**（拓扑序）排列，保证祖先下标 < 后代下标 ⇒ `G` 严格下三角；
- **衰减比**：`g_t/g_i` 沿**路径**累乘，而不是沿线性序列——树上一个节点的 `g` 是其**祖先链**上的 α 乘积。

### 2.2 求解方式 = 三个方案的工程分野

`(I+G)` 是单位下三角、行列式为 1，必然可逆。怎么解，决定了硬件映射：

| 解法 | 形式 | 谁用 | 硬件特征 |
|---|---|---|---|
| **前代（forward substitution）** | 逐行解，`O(N²d_v)` | TreeWY（"one triangular solve"）、DeltaNet chunk | **N 个串行步**（可 wavefront/分块降低串行深度）；对 `N=4` 的链无所谓，对 `N=40` 的树很痛 |
| **Neumann 级数** | `G^{d+1}=0 ⇒ (I+G)^{-1} = Σ_{m=0}^{d} (−G)^m`，`d` = 最大深度 | **Bole**（Lemma 3 + 式(8)） | **D 次 matmul，无三角求解**；T×T 与 T×d_v 的 GEMM，形状固定 ⇒ 可整图捕获 |
| **链式并行（不做整树）** | 每条 root→leaf 链各自跑一个 chunked kernel，链间并行 | **SpecLA** "chain-decomposed hybrid path" | 依赖链短、setup 成本低，牺牲一部分并行度换取 kernel 简单 |

> ⭐ **Neumann 级数这一条对 Ascend 极其关键**：`d`（MTP depth）通常只有 3–4，于是"解三角系统"退化成 **4 次固定形状的矩阵乘**。Cube 单元天生吃 matmul，而 CANN 里**没有** batched triangular solve / 逆矩阵算子，AscendC 手写 wavefront 前代也很别扭。Bole 这条路把"不友好的算子"换成了"最友好的算子"。

### 2.3 commit：只重算一个状态

接受节点 `a` 之后（TreeWY 式(3) / Bole 式(10)）：

```
S_a = g_a S_0 + Σ_{i≼a} (g_a/g_i) ṽ_i k_i^T
    = P_a S_pre + (D_a K_a)^T U_a          ← Bole 写法：一次 tensor-core matmul
```

- 存储：`O(N·d_v)`（TreeWY 存 `Ṽ`）或 `O(N(d_k+d_v))`（Bole 存 `P,K,U`），而不是 `O(N·d_k·d_v)`
- **`d_k=d_v=128` 时单 head 降 128×**（TreeWY 原文）
- 若整轮全被拒绝：什么都不用做，槽位保持 `S_pre`（Bole 原文明确说明"no additional state snapshot or copy-on-write slot is needed"）

### 2.4 与"输入缓冲"（ReplaySSM）的关系：同一件事的另一种写法

ReplaySSM 不存 `ṽ`，而存**原始输入** `(α, k, v)`，把 state 表示成：

```
S = f(S_ckpt, ring_buffer[(α_j,k_j,v_j)])
```

rollback = **移动环形缓冲区的指针**（`O(1)`，零字节拷贝）。
它的 verify 内核也是同一个式子：

```
OutputOnly(S_ckpt, B, Q) = S_ckpt Q D + V (W ⊙ (K^T Q))
```

注意 `V(W ⊙ (K^T Q))`：先算标量 `k^T q` 再缩放 `v`，**永不 materialize `d_v×d_k` 的状态**（原文："*the left side materializes the d_v × d_k state, while the right side forms the scalar k^T q and scales v*"）。GDN 版本需要 `T×T` 三角求解来定 `ṽ`（ReplaySSM Appendix B.3）。

> **所以 ReplaySSM = 式(2)(3) 的"输入侧因子化"；TreeWY/Bole = "伪值侧因子化 + 树掩码"。** 两者在 GDN 上算的是**同一个三角系统**（TreeWY 原文直接承认："*Inside the verify window it solves the same triangular system we do*"）。

### 2.5 精度轴：DAMP / DASC 是正交的第三条腿

`(I+G)Ṽ=R` 和 `S_a` 的重算都是**逐位**`（bit-level）依赖 FP32 累加。而实际瓶颈常常是 **state 本身的字节数**（`B=256` 时 80 GB）。DAMP（混合精度：关键 key channel 保 FP16、其余 INT8，9.9 bit/value）与 DASC（按 retention horizon 压缩 **prefix-cache 的 state checkpoint**）不解决 n+1 的**份数**问题，但直接把**每份的大小**打下来 1.6–3×，与因子化方案**可以叠加**。

> **叠加效果示例**：35B / B=256 / n+1=4：30 GiB → 因子化后 ≈ 7.5 GiB（只留 1 份 + 小日志）→ 再叠加 9.9 bit 混合精度 → ≈ 3.7 GiB。**从"半张卡"到"可忽略"。**

### 2.6 完整推导（可直接照着实现）

> 这一小节是把 §2.1–2.3 串成一条可以照着写代码的推导链。符号与 TreeWY / Bole 一致。

**Step 0 · 排列**
把 draft 树（链是树的特例）按 **DFS 先序**编号 `1..N`，保证 `i ≺ t ⇒ i < t`。`S_0` 是已提交的 checkpoint。

**Step 1 · 路径累积衰减**
```
g_0 = 1,   g_t = g_{parent(t)} · α_t
```
注意 `g_t` 是**沿树路径**的乘积，不是沿线性序列。

**Step 2 · 伪值定义**
```
ṽ_t = β_t (v_t − α_t · S_{parent(t)}^T k_t)
```

**Step 3 · 路径分解（TreeWY 式(2) 的核心）**
沿根到 `t` 的路径展开递推：
```
S_t = g_t S_0 + Σ_{i≼t} (g_t/g_i) ṽ_i k_i^T              (a)
```
⇒ **任意节点的状态都能用 `S_0` + 祖先的 `ṽ_i` 直接写出，不需要中间状态。**

**Step 4 · 输出读出**
```
o_t^T = q_t^T S_t = g_t q_t^T S_0 + Σ_{i≼t} (g_t/g_i)(q_t^T k_i) ṽ_i^T
⇒ O = D_g Q S_0 + C Ṽ ,
   D_g = diag(g_t),  C[t,i] = (g_t/g_i)(k_i^T q_t)·[i≼t]        (b)
```

**Step 5 · 把 ṽ 的定义代回，得到线性系统**
对 `ṽ_t = β_t v_t − β_t α_t S_{parent}^T k_t` 里的 `S_{parent}` 用 (a) 展开（注意 `α_t g_{parent} = g_t`）：
```
ṽ_t = β_t v_t − β_t g_t (S_0^T k_t) − β_t Σ_{j≺t} (g_t/g_j)(k_t^T k_j) ṽ_j
```
令
```
G[t,j] = (g_t/g_j)(k_t^T k_j)·[j≺t]        ← 严格下三角
R[t]   = β_t (v_t − g_t S_0^T k_t)
```
就得到**唯一需要求解的系统**：
```
(I + diag(β) G) Ṽ = R                     ← TreeWY 式(2)
⇔ (I + G') U = D_β (V − D_g K S_0) ,  G'[t,j] = β_t (g_t/g_j) k_t^T k_j   ← Bole 式(5)，β 折进 G
```

**Step 6 · 求解（Ascend 建议用 Neumann）**
`G` 严格下三角且深度有界 ⇒ `G^{d+1}=0`（`d` = 树最大深度），
```
Ṽ = Σ_{m=0}^{d} (−diag(β)G)^m R
```
**`d` 只有 3–4 ⇒ 3–4 次 `N×N @ N×d_v` 的 matmul**，全部定形状，无三角求解、无分支。

**Step 7 · commit**
采样得到接受节点 `a` 后，只重建这一个：
```
S_a = g_a S_0 + Σ_{i≼a} (g_a/g_i) ṽ_i k_i^T
    = g_a S_0 + (D_a K_a)^T U_a           ← 一次 matmul
```
其中 `K_a/U_a` 是 `a` 的祖先链（含自身）的 `k/u` 堆叠，`D_a = diag(g_a/g_i)_{i∈P(a)}`。
**其余节点全部丢弃，什么都不用回滚。**

**Step 8 · 链的特例**
把 `≺` 换成 `<`（线性序列），Step 5/6 就退化成 **DeltaNet 的标准 chunked UT 变换**；把 chunk 长度设为 `N = 1 + num_spec_tokens`，就是链式投机解码的正确解。**这也是为什么"链"在实现上比"树"简单一个量级——它不需要祖先掩码，只需要因果掩码。**

---

## 3. 七个方案逐个精读

阅读顺序建议：**ReplaySSM → TreeWY → Bole → SpecLA**（这四篇是"n+1 份状态"的正解），再读 **DeltaLog / DAMP / DASC**（这三篇解决"每份太大/搬运太贵"，是叠加项）。

每篇统一按 8 个字段：`论文` / `解决什么问题` / `核心思路` / `怎么处理 n+1` / `开源代码` / `引擎落地` / `代价与局限` / `Ascend 移植要点`。

---

### 3.1 ReplaySSM：Cache SSM Inputs, Not State

**论文**
- arXiv: **2610.13778** — <https://arxiv.org/abs/2610.13778>
- 作者：Ze-Wei Liou, Harris Nover, Amey Naik, Andrii Skliar, Igor Shovkun, Ido Shahaf, **Tri Dao**（Princeton University + NVIDIA）
- 相关材料：[Dao AI Lab blog](https://dao-lab.ai/blog/2026/replayssm/) ／ [NVIDIA 技术博客](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/)

**解决什么问题**
一句话：**SSM/线性注意力的 state 读写是纯带宽开销，而"投机解码无法回滚"逼着系统给每个 draft 位置存一份快照。**

> 上游 PR [#49887](https://github.com/vllm-project/vllm/pull/49887) 把这份算术写得最直白：
> "*a step with T = 1 + num_speculative_tokens tokens writes **T full head_v × head_k states per value head**. Cost grows linearly in T.*"
> 这就是"n+1 份"问题的标准表述。

论文把三个问题归因到同一个根因（"*eagerly summarize each token's inputs into the state at once and write the state to HBM at every step*"）：

1. **C1 状态 I/O 大、算术强度低**：Mamba-2 的 state 更新只有"elementwise 衰减 + 一次 rank-1 外积"。论文实测 Nemotron-3-Super-120B（NVFP4，B=256，单 B300）在 **上下文 100K 以内** SSM state 更新都是最大延迟项——因为 SSM 层数是 attention 层的 5 倍，且每层 FP32 state 是 **4 MiB/序列**，相当于 8K 上下文的 FP8 KV cache。
2. **C2 无廉价回滚**：state 里没有"可按 token 寻址的历史"，vLLM 只能存快照。
3. **C3 量化误差沿递推累积**：低精度 state 既产生输出又喂下一次更新，误差闭环放大。

**核心思路（三个机制）**

**(a) 状态 = checkpoint + 有界输入环形缓冲区**

```
S_t = α_t S_{t-1} + v_t k_t^T
    = γ_h S_ckpt + Σ_{j=1}^{h} γ_{j+1:h} v_j k_j^T          (式 4)
```

于是物理表示改为 `(S_ckpt, B)`，`B` 里存 `(α_j, k_j, v_j)`。**"非 flush 步"只 append、不写 state**；缓冲区满（容量 `L`，投机解码时 `L+T`）才 flush 一次，把缓冲折叠进 checkpoint。state 写次数从"每 token"变成"每 L token"。

**(b) 投机解码的回滚 = 移动指针（§5.1）**

关键工程细节：环形缓冲容量 **`L+T`**，且"每步开始时缓冲区里最多 `L` 条已提交记录"这一不变式，保证 `T` 条 draft 永远塞得进去。验证后：
- draft 记录被 append 到第 `h` 条之后；
- **只有被接受的前 `a` 条**通过 `h ← h+a` 被"提交"；
- 被拒记录**越界、直接被下一轮覆盖**，零拷贝；
- flush 分支只折叠 committed 记录，**因此可以在接受长度未知时就执行**（这一点很重要：不必等采样结果）。

**(c) Output-Only Verification（§5.2）——最漂亮的一步**

```
OutputOnly(S_ckpt, B_{1:h+T}, Q) = S_ckpt·Q·D + V ⊙ (W ⊙ (K^T Q))      (式 6)
```

`D` 是对角衰减，`W` 是带因果掩码的累积衰减矩阵。要点：
- 因为 `(v k^T) q = v (k^T q)`，**右侧只算标量 `k^T q`，永不 materialize `d_v×d_k` 的 state**——FLOPs 更少、且没有中间 state 落地；
- **`K^T Q` 可以在所有共享 K/Q 的 value head 之间只算一次**（GQA 复用），这在 verify 窗口把算术放大的时候收益更大；
- 对 GDN：draft 之间的修正互相依赖，**用一次 `T×T` 三角求解**拿到全部 `T` 个 `ṽ`（Appendix B.3）——与 TreeWY 是同一个系统，只是 ReplaySSM 只做链。

**(d) 双精度 checkpoint（§6）**：HP checkpoint 负责"喂递推"，LP checkpoint 负责"算输出"，LP 的舍入误差**永不回流**到递推。GDN 需要额外缓存 `v`，在 flush 时用 HP 状态重算修正项。

**怎么处理 n+1**
- **份数**：从 `T+1` 份 dense state → **1 份 checkpoint + 1 份 `(L+T)` 条记录的小缓冲**。缓冲是 `O(H(d_k+d_v))` 级，不是 `O(H d_k d_v)`。
- **回滚成本**：`O(1)` 指针移动。
- **验证成本**：output-only 形式，连"临时 state"都不写。
- 论文实测（L=16, T=4, B=256, 单 B300）：投机解码吞吐 ≈ **vLLM 原生投机解码的 1.8×**（两个 120B 级模型）；标准解码 **1.20–1.47×**。

**开源代码**：✅ **有，官方**
- <https://github.com/Johnny-Liou/ReplaySSM>（**vLLM fork**，Apache-2.0，63★，最后更新 2026-10-27）
- README 明确说明：*"This repository is a vLLM fork with ReplaySSM implementations for Mamba2 and Gated DeltaNet (GDN), covering standard and speculative decoding"*；上游 **只合入了 Mamba2 标准解码**，剩余 Triton kernel / 模拟量化 / 低精度 CuTe kernel 只在此仓库。
- 第三方：`zz774ever/fp8-replayssm-gdn-kernel`（0★，实验性质，不建议依赖）

**引擎落地（已逐条核验 PR 状态）**

| 系统 | 范围 | PR | 状态 |
|---|---|---|---|
| TensorRT-LLM | Mamba2 MTP（checkpointing replay） | [#14203](https://github.com/NVIDIA/TensorRT-LLM/pull/14203) | ✅ merged 2026-06-19 |
| TensorRT-LLM | **GDN MTP replay**（NVIDIA 独立贡献） | [#16464](https://github.com/NVIDIA/TensorRT-LLM/pull/16464) | ✅ merged 2026-07-20 |
| vLLM | Mamba2 **标准解码** | [#48018](https://github.com/vllm-project/vllm/pull/48018) | ✅ merged 2026-07-24 |
| vLLM | FlashInfer 后端，Mamba2 标准解码 + MTP | [#52506](https://github.com/vllm-project/vllm/pull/52506) / [#52928](https://github.com/vllm-project/vllm/pull/52928) | ✅ merged 2026-09-01 / 09-30 |
| vLLM | **GDN 标准解码** | [#48792](https://github.com/vllm-project/vllm/pull/48792) | ❌ **closed, NOT merged** |
| vLLM | Mamba2 投机解码 | [#49847](https://github.com/vllm-project/vllm/pull/49847) | ❌ **closed, NOT merged** |
| vLLM | **GDN 投机解码** | [#49887](https://github.com/vllm-project/vllm/pull/49887) | ❌ **closed, NOT merged** |
| vLLM | 原始 draft PR + RFC | [#47576](https://github.com/vllm-project/vllm/pull/47576) / [#47572](https://github.com/vllm-project/vllm/issues/47572) | RFC **open**；PR closed（merge conflict） |
| vLLM | **整个 ReplaySSM-投机分支被批量关闭** | #47576 / #49847 / #49887 / #48792 全部 **2026-10-12 closed，未合并**（未见公开理由，只有 bot 评论） | ⚠️ 上游已改用 **RecoverSSM** 路线 |
| FlashInfer | Mamba2 checkpointing / MTP | [#3975](https://github.com/flashinfer-ai/flashinfer/pull/3975) | ✅ merged |
| FlashInfer | **GDN 投机解码 u/d cache** | [#4081](https://github.com/flashinfer-ai/flashinfer/pull/4081) | ✅ merged 2026-08-11 |
| FlashInfer | prefix-cache materialize | [#4815](https://github.com/flashinfer-ai/flashinfer/pull/4815) | ✅ merged |
| FlashInfer | GDN vcache spec-decode（fused verify+flush） | [#4140](https://github.com/flashinfer-ai/flashinfer/pull/4140) | 🔄 **open** |
| SGLang | 移植 RFC | [sglang#28511](https://github.com/sgl-project/sglang/issues/28511) | closed |

**生产落地**：README 明确写 **GLM-5.3-Flash 与 Kimi-K3 生产环境在用**；Kimi-K3 论文（arXiv:2607.24653）§5.4.2 在 KDA 层"MTP 验证时缓存投影后的 draft 输入"，并把 ReplaySSM 列为 concurrent work。

**代价与局限**
- **只支持链（chain）**。TreeWY 原文点明："*the implementation linked in the RFC is chain-only (not a tree)*"。树分支会需要"每个节点不同状态"，超出了环形缓冲的表达力。
- flush 有偶发尖峰（1/L 的步数要付整份 state 写）；`L` 与 `T` 的权衡与接受率耦合（接受率高 → 缓冲区填得快 → flush 频繁）。
- 环形缓冲的 **cursor bookkeeping 是 host 侧逻辑**，跨 CUDA Graph/多 kernel 时容易出 bug（vLLM 侧就有一堆相关 bugfix，见 §5）。

**Ascend 移植要点（详见 §6）**
- **最容易落 Ascend 的一篇**：它只需要 `checkpoint` + `append` + `flush` **两种固定 shape 的 kernel**，天然可映射到 **ACL Graph 的双图（append / merge）** 模式（DeltaLog §4 已经给出这个范式）。
- 关键算子：`K^T Q`（GQA 复用一次）+ `W` 的因果掩码衰减累乘 + `T×T` 求解（链上 `T` 很小，可用 Neumann 或直接 forward substitution）。
- 注意 `P`/`W` 的累积衰减 `g_t` 需要 **FP32 沿 T 的 cumprod**：昇腾 Vector 单元做 64 长度的 cumprod 没问题，但要避免每步都做跨核同步。

---

### 3.2 TreeWY：Speculative Verification for Gated DeltaNet Hybrids

**论文**
- arXiv: **2608.20961**（v2, 2026-10-22）— <https://arxiv.org/abs/2608.20961>
- 作者：**Sneha Murthy Ghantasala**（Thomson Reuters），单作者，CC BY 4.0
- 一句话：*"对 gated delta rule 做树结构 WY 变换，一次三角求解算完所有 draft 节点输出，commit 时只重建唯一被接受的状态。"*

**解决什么问题**
GDN 混合模型的投机解码里，**快照式回滚无法表达树**：每个 draft 位置一份完整 state，树有 `N` 个节点就要 `N+1` 份，**且分支之间不能共享**（sibling 的状态互不相同）。于是"更宽、接受率更高的树"在内存上直接不可行。TreeWY 的目标就是**把快照彻底删掉**。

**核心思路**
1. 把 delta rule 重写成"带修正值的衰减加法注意力"：
   ```
   S_t = α_t S_{t-1} + ṽ_t k_t^T ,   ṽ_t = β_t (v_t − α_t S_{t-1} k_t)
   ```
2. 把 draft 树按 **DFS 先序**展平，令 `i ≺ t` 表示 `i` 是 `t` 的**严格祖先**，则：
   ```
   (I + diag(β)G) Ṽ = R
   G[t,i] = (g_t/g_i)(k_t^T k_i)·[i ≺ t]      ← 严格下三角
   R[t]   = β_t v_t − β_t g_t (S_0 k_t)
   ```
   一次前代求解 ⇒ **所有节点的输出**（不需要任何中间 state，也不随树变宽而变复杂）。
3. commit：
   ```
   S_a = g_a S_0 + Σ_{i≼a} (g_a/g_i) ṽ_i k_i^T
   ```
   只存 `Ṽ`（`O(N·d_v)`）。`d_k=d_v=128` 时**单 head 小 128×**。

**贡献定位**：把 DeltaNet 的 chunk 内 WY 变换推广到"**树 + 严格祖先**"；标量衰减的祖先掩码来自 STree，**TreeWY 新增的是把 GDN 的非交换 rank-1 修正也塞进这个掩码**（Mamba2 只需要 cumsum，GDN 需要真正的三角求解）。

**怎么处理 n+1**
`N+1` 份 state → **1 份 state + 1 个 `N×d_v` 的伪值矩阵**。链和树**统一处理**，代码里只是一个掩码的形状不同。

**开源代码**：❌ **未找到公开代码仓**
- 论文自述："*We implement TreeWY for vLLM, as a fork of its main branch (**not yet upstreamed as of this writing**)*"
- 单作者、Thomson Reuters，无项目页；GitHub 搜 "TreeWY" 只有同名的 profile 配置仓，**非本工作**。
- 反过来，TreeWY 也在论文里指出 Bole 没有代码："*We have not run a head-to-head, since **Bole's paper does not link to a code implementation***" —— **两篇 concurrent work 互相确认了对方没有公开代码**。

**引擎落地（vLLM）**
- ⭐ **有 RFC**：[vllm#54080](https://github.com/vllm-project/vllm/pull/54080) `[RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models`（**open**）。这是**全网唯一一处给出 n+1 内存硬数字**的地方：
  > Qwen3.5-397B：每层每序列 **90 MiB**；depth-3 chain 共 **360 MiB**；`[2,2,2]` 树约 **1.2 GiB**。
- RFC 同时承认树还不是吞吐赢，且明确指出原因：
  > "*Draft trees are not a throughput win yet … throughput falls at every width. The cause is **CUDA graph capture**.*"
  ⇒ **【关键证据】树的收益被"图捕获"吃掉，这是整个社区目前都没解决的问题，也是 Ascend 上做树的最大风险。**
- 作者私有分支：`sneha5gsm:vllm:treewy-gdn-spec-decode`（commit `b073ed6`, 2026-08-08）。
- 通过两个 `SpeculativeConfig` 选项暴露：
  - `mamba_state_commit="reconstruct"`（替代默认的 `"store_all"`）
  - `draft_tree_widths`（按层给分支因子，把链变成树）
- **链**：一个 fuse 的、**可被 CUDA Graph 捕获**的 Triton kernel（验证 + commit 一体）。
- **树（`w>1`）**：需要非因果祖先掩码，**无法从 CUDA Graph 回放**，vLLM 只能退化成 piecewise capture，而这会把 GDN mixer 一起踢出 graph，**代价远大于掩码本身** ⇒ 论文明确说树"*we report trees as enabled and correct, not as a speedup*"。
- 树还必须**原子调度**：DFS 前缀是另一种拓扑，塞不进 per-step token budget 就直接不投机，而不是截断。
- 上游状态：**无 PR**（作者明说未上游）。

**实测（B200）**
- Qwen3.5-35B-A3B（TP1）+ Qwen3.5-397B-A17B（TP8），depth-3 MTP 链
- 峰值 KV 降低 **2.0–2.45×**（低并发；高并发因主动多接纳请求而摊薄到 ~1.04×）；**接受长度与 baseline 基本一致**（mean 差 <1%，max <3%），per-position 接受率各深度差 <1%
- 内存受限时：吞吐最高 **1.40×**、p99 TTFT 改善最多 **5.62×**；内存不紧张时**反而慢几个百分点**（1–2% per-token）
- 树宽 3→39：接受长度 3.24→3.58，**但还不是吞吐赢**（因为树 kernel 不可捕获）
- 与 ReplaySSM 对比（Appendix C）：两者在内存机制上打平（都删掉了快照），**ReplaySSM 的 per-token 更快**（作者归因为"ReplaySSM 延迟写 state，而我们每次 commit 都 materialize"），这是 TreeWY 自己承认的未验证归因。

**代价与局限**
- 树版本**在 vLLM 上目前是负优化**（graph capture 问题），这是整个"树方案"最现实的坑。
- 需要 prefix caching 关闭（论文所有实验都关）。
- token 流不再与 baseline 逐位一致（fp64/fp32 校验 + bf16 kernel 容差），会用 shared no-spec reference 做 gate。
- 单作者论文，实验规模（2 个模型、1 家 GPU）比 Bole 小。

**Ascend 移植要点**
- ⚠️ **最大的坑和论文里一模一样**：非因果祖先掩码 ⇒ 无法整图捕获。昇腾侧 ACL Graph 对动态掩码/动态 shape 更敏感。**所以 Ascend 上第一版不要做 tree，只做 chain。**

---

### 3.3 Bole：Efficient Tree Speculation for Hybrid-Attention Language Models

**论文**
- arXiv: **2608.01651**（v1, 2026-08-03）— <https://arxiv.org/abs/2608.01651>
- 作者：Li Wang, Yi Su, Xiabao Wu, Chiran You, Yongchao Liu, Zhan Qiu, Juelu Zhang, Jiajun Zheng, Fangxin Liu, Jie Zhang, Chen Tian, **Chengying Huan**（南京大学 + 蚂蚁集团 + 上交 + 北大）
- 稿件状态：PDF 页眉标注 **"HPCA 2027 Submission #814 / Confidential Draft / Do NOT Distribute"**，License CC BY-NC-SA 4.0 —— **引用时请注意这是投稿中的非正式版本**。

**解决什么问题**
和 TreeWY 同源，但**从 kernel + runtime 两端一起做**，并给出三条量化的系统瓶颈（Qwen3.5 系列，A100/GB10）：

| 瓶颈 | 论文实测 |
|---|---|
| **L1 串行递推**：现有引擎必须按节点逐个 apply 递推 | 树从 8 → 64 节点，线性层占 target 前向的比例 **4% → 27%** |
| **L2 状态爆炸**：每个节点一份完整 post-update state | Qwen3.5-122B-A10B、8 个 32 节点请求 → **36 GiB**，另一个更大 batch 点 → **72 GiB**（两个 batch 数值被 LaTeX 剥离，但可见 2× 线性），而 Bole 的因子**两点都 < 1 GiB** |
| **L3 验证容量随硬件变化**：超过阈值后加 token 不再"近乎免费" | MLP 的 GEMM 从 weight-bound 转 compute-bound，转折点 **A100 ≈ 128 行，GB10 ≈ 256 行** |

单请求 100 节点树的瞬态状态：快照 **4.7–14.1 GB** vs Bole 因子 **57–151 MB**（Table I）。

**核心思路（三件套）**

**(1) 闭式树验证（Theorem 1）——与 TreeWY 同一个系统，但用 Neumann 解**
```
O = D_P Q S_pre + C (I + G)^{-1} D_β (V − D_P K S_pre)          (式 2)
G_ij = β_i (P_i/P_j) k_i^T k_j      （仅当 j 是 i 的严格祖先）
```
**Lemma 3（depth-bounded nilpotency）** 是 Bole 的独门武器：
```
G^{d+1} = 0  ⇒  (I+G)^{-1} = Σ_{m=0}^{d} (−G)^m       (式 6)
Z^{(m+1)} = −G Z^{(m)} ,  U += Z^{(m+1)}               (式 8)
```
`d` = 树的最大深度（MTP 通常 3–4）。**于是"解三角系统"变成 `d` 次固定形状的 matmul，不需要任何三角求解或前代**——这是 Bole 与 TreeWY 最本质的实现差异，也是**对 Ascend 最重要的可借鉴点**。

**(2) value-tiled kernel（§IV-B）**
- `G, C ∈ R^{T×T}` **与 value 通道无关**，于是按 `d_v` 切片：`N_v = ⌈d_v/b_v⌉` 个 CTA，每个 CTA 负责一个 value tile，**树本身不切分**（每个 CTA 都覆盖完整 `T` 节点范围）
- 一个 "factor producer" 先算 `KK^T`、`QK^T` 并套掩码/衰减比/β，构造 `G`、`C` **一次**，写进 layer-local workspace，**在所有 value tile 间摊销**
- 只存 `P`、`β` 两个长度 `T` 的向量，**从不 materialize `T×T` 对角矩阵**
- Neumann 传播在 **on-chip** 完成（`Z^{(m)}`、`U^{(r)}` 不落 HBM），最后只有 `O` tile 和 commit 需要的 `U` tile 离开 CTA

**(3) 因子化状态生命周期（§IV-D）**
```
存储 = Θ(H d_k d_v)                     ← 唯一 committed state
     + Θ(T H (d_k + d_v + 1))           ← 树因子 P, K, U
```
- 三级生命周期：canonical state（随请求存活）／factors `P,K,U`（活到本轮接受决定）／`G,C` + Neumann workspace（**只活一层**，跨层复用）
- **被拒分支不占 state-cache slot，也不需要 rollback**
- commit（式 10）：`S_new = P_a S_pre + (D_a K_a)^T U_a`，**一次 tensor-core matmul**；接受路径在所有线性层间**共享**，可 fold 成一个 device launch 写各层 slot；**旧 state 在被覆盖前仍可读 ⇒ 不需要 COW slot**

**(4) 硬件感知的 batch 级验证预算（§V）**
- 一次性 profiler 记录"硬件高效验证容量"（按执行 bucket）
- 运行时按**累积 draft 概率**在 batch 内跨请求排序分配预算，**保证每个请求选中的树仍是前缀连通的**
- 每个 capacity bucket 复用同一套捕获好的 CUDA Graph

**怎么处理 n+1**
`N+1` 份 state → **1 份 state + `N` 组 `(P,k,u)` 因子**，瞬态内存降 **82–99×**，树宽/批大小不再推高 state 占用；线性注意力树验证加速 **3.4–7.7×**；离线解码吞吐最高达自回归的 **4.72×**；在线 agent 负载下 TTFT / TPOT 相对最强树投机基线亦有显著下降（摘要中该两个百分比数字被 LaTeX 剥离，此处不臆测）。

**开源代码**：❌ **未找到公开代码仓**（`gh search repos "Bole-SGLang"` 等均无结果；论文声称集成进 SGLang，但 SGLang 主线搜不到对应实现）
> TreeWY 作者也指出："*Bole's paper does not link to a code implementation*"——两篇 concurrent work 互相确认了对方**没有公开代码**。

**引擎落地（⚠️ 重要更正）**
- **SGLang**：论文称"集成进 SGLang"（含树构造 → target 执行 → 采样 → state commit、device-resident flat-ragged forest、按 bucket 复用 CUDA Graph、约 6.2 kLoC），**但这是不可证实的**：没有 repo、没有分支、没有作者 PR，论文里唯一的 SGLang 链接是基线版本 `release/v0.5.12`。
- **唯一的公开实现是第三方**：[sglang#46702](https://github.com/sgl-project/sglang/pull/46702) `[Feat] linear-attn: Bole spec-verify for GDN (drop per-draft SSM snapshots)`（open，2026-10-27，+918/−7）。**其 PR body 明确写 "Not supported yet … EAGLE tree verify"，即只做了 GDN 的链式验证。**
- vLLM：**无**。
- **结论：不要把 Bole 当成"可直接用的开源实现"。它的价值 = 提供了 Neumann 级数这条最 Ascend 友好的解法思路，而不是代码。**

**代价与局限**
- 论文自己也承认：更宽的树"**还不是吞吐赢**"（acceptance 被 draft depth 卡住，树 kernel 更贵）。
- 需要 profiler 标定 + 按 bucket 的图捕获，工程复杂度高。
- 保守：batch 级预算会**主动砍掉**部分请求的树。
- 状态：HPCA 投稿，**结论可能随评审变化**；CC BY-NC-SA（非商用）。

**Ascend 移植要点**
- ⭐ **Neumann 级数（式 6/8）是整份调研里对 Ascend 最友好的一招**：把"三角求解"变成 3–4 次 `T×T @ T×b_v` 的固定形状 matmul，完全落在 Cube 上，且**形状与树拓扑无关**（只有掩码数值变）⇒ **ACL Graph 可捕获**，直接绕开 TreeWY 的坑。
- value-tiling + factor producer 摊销 `KK^T/QK^T` 的思路可直接照搬（Ascend 的 UB 是 192KB 级，`64×128` FP32 tile = 32KB，正合适）。
- "跨层 fold 成一个 launch 写各层 slot"在 Ascend 上需要 `aclnnBatch` 类接口或多流，成本比 CUDA 高，建议先一层一写。

---

### 3.4 SpecLA：Efficient Speculative Decoding for Linear-Attention Models

**论文**
- arXiv: **2607.16673** — <https://arxiv.org/abs/2607.16673>
- 作者：南京大学（State Key Lab for Novel Software Technology，含 Fuliang Liu、Rong Gu、Chen Tian 等）+ 阿里巴巴（Xue Li、Sheng Zhong）
- 一句话：*"用提交上来的拓扑本身作为 target kernel 的调度信号"*——**唯一一篇把 chain/tree 分流做成一等公民的论文**。

**解决什么问题**
论文先把两种"直接照搬"判死（这一段是极好的 problem statement，建议直接引用）：
1. **用 decode kernel 重放候选**：保留了"逐 token 的 state 往返"，树还要为每个分支准备不同中间状态 ⇒ 投机解码本来想摊销的东西一点没摊销。
2. **用 prefill kernel 跑 draft 链**：投机窗口太短（3–8 token），摊销不了 prefill 的 setup（layout conversion / block summary / UT-solve / 同步 / 中间张量），而且**普通因果 prefill 让"每个候选看到所有更早的候选"，而树要求"只看到自己的祖先路径"**。
3. 第三条挑战：**接受后的 commit 没有廉价回滚**——候选的因子在采样前就产生了，但不能"投机地提交再截断"；全状态快照太贵，token replay 又是重复计算。

**核心思路：三层设计**

**Level 1 — 拓扑感知验证（三路分流，论文的精华）**

| 路径 | 适用 | 做法 | 相对根到叶重放的加速 |
|---|---|---|---|
| **state-resident serial** | 短链 | state tile **常驻 SRAM**，相邻同层更新之间不落 HBM | — |
| **tree-masked factorized** | 分支树 | 把**树依赖掩码塞进 GDN 的 UT 变换**（=TreeWY/Bole 的式子） | 完全并行端点 |
| **chain-decomposed hybrid** | 多链 | 链内跑串行 kernel，**链间并行** | **1.80–7.11×** |

> 这个"不做整树、按 ready chain 并行"的折中是**最被低估的一招**：它避开了非因果掩码的图捕获问题，又比纯 root-to-leaf 重放快一个数量级。

**Level 2 — accepted-factor state management（§5）**
- 核心观察：**验证过程本身已经产生了重建被接受更新所需的紧凑因子**
- 所以：**缓冲因子**（不是全状态，也不是纯 token 日志）→ 采样后**只 gather 被接受路径** → 把 pending 因子**延迟到下一次验证 kernel 内部**应用，**省掉一次独立的 state-update pass**
- 实测：因子缓冲复用让"接受状态恢复"延迟降 **2.74–4.28×**；延迟 state update 让 commit 边界延迟降 **1.15–1.44×**

**Level 3 — target-aligned drafting**
- **置信度剪枝**：低概率路径在验证前就砍掉（因为 target 侧代价与提交的拓扑强相关）
- **EAGLE drafter 用 recurrent target 的 feature 训练**，让剪枝分数反映"这个 target 的动力学"
- 保留的候选打包成 **parent-pointer 拓扑**，直接被 target kernel 消费

**怎么处理 n+1**
同样走"因子"路线，但它**明确区分了"份数"和"调度"**：
- 份数：`N+1` → 1 份 durable state + 因子缓冲
- 额外一层洞察：**"提交多少候选"本身是个调度/代价问题**，不是越多越好（L3）

**开源代码**：❌ **未找到公开代码仓**（`gh search repos "SpecLA"` 全是同名无关项目）
- 论文自述实现为 **PyTorch + Triton** 原型，**H100** + 公开 **GDN-1.3B** target（`m-a-p`），**未集成 vLLM/SGLang**

**实测**
- H100 / GDN-1.3B：混合集 **1.42×**、GSM8K **1.70×**、HumanEval **1.06×**（相对自回归）——**HumanEval 只有 1.06×，论文自己如实报告**
- 状态量化：GDN-1.3B 每层 **2 MiB FP32**，每个 token 的往返 **4 MiB**；L=8 个候选的快照 = **16 MiB/层**，而因子缓冲只有 **128.5 KiB/层（≈128× 小）**
- 微基准：chain-decomposed 比 root-to-leaf replay 快 1.80–7.11×；factor-buffer 2.74–4.28×；delayed update 1.15–1.44×
- ⚠️ **反直觉的负结果（全文最有价值的结论之一）**：朴素的**并行树 kernel 在 16–32 节点时常比串行还慢（0.52–1.02×）**。所以 SpecLA 才需要"三路分流"而不是"一律用树"。

**代价与局限**
- 只有 1.3B 模型、单卡 H100，**没有大模型/服务化证据**（对比 ReplaySSM 的 120B×B300、TreeWY 的 397B/B200、Bole 的 4 模型 2 平台）
- 需要**自己训 EAGLE drafter**（target-aligned），复现门槛高
- 三路分流意味着**三套 kernel 要维护**

**Ascend 移植要点**
- ⭐ **"chain-decomposed hybrid" 是 Ascend 上性价比最高的折中**：链内串行天然对应 Ascend 擅长的定长流水，链间并行用多核/多流；**不需要非因果掩码**。
- "state tile 常驻 SRAM" 在 Ascend 上对应 **UB 常驻**：`128×128` FP32 = 64KB < 192KB UB，**一个 head 的状态能整个放进 UB**，这点比 GPU 的 100KB shared memory 更宽松。但要小心 Vector/Cube 之间的 tile 搬运。
- "延迟 state update"（把 commit 折叠进下一次 verify kernel）在 Ascend 上能省一次 kernel launch + 一次 HBM 往返，值得照抄。

---

### 3.5 DeltaLog：Deferred Materialization of Recurrent States

**论文**
- arXiv: **2608.15533** — <https://arxiv.org/abs/2608.15533>
- 作者：Junqing Lin, Jingwei Sun, Guangzhong Sun（中国科学技术大学）
- 一句话：*"把状态拆成 dense base + 有界 update log，绝大多数步只 append 紧凑因子，周期性 merge 回 base。"*

**解决什么问题：`state-update tax`**
一个 token 贡献的因子只有 `O(BH(d_k+d_v))` 个值，但 eager 实现要写回整个 `[B,H,d_k,d_v]` state ⇒ 每 token 触碰 `O(BH d_k d_v)` 个元素。论文把这个放大效应命名为 **state-update tax**，并指出它来自**物化策略**而非递推本身。

**核心思路**
```
物理状态 = {  dense FP32 base S̄,            // 上次 merge 的结果
              cumulative log-decay ℓ̄,       // head 标量 或 per-key 向量
              K_log, U_log ∈ R^{B×H×C×d},   // 有界日志，C = M-1
              Λ_log,                        // 每条日志的 decay 快照
              L ≤ C }                       // 当前日志长度
```
- 前 `M−1` 步：**append**（只写紧凑因子），dense base 不动
- 第 `M` 步：**merge**（把日志折进 base，写一次 dense state，清空日志）
- **读路径**：每个 decode 步仍要读 dense base（保留长程历史），所以这是**只优化写路径**
- **value-tile 布局**：按 `d_v` 切分，每个 value tile 存"base 切片 + 日志的 value 侧分片"，**key 侧因子跨 tile 共享、每 request-head 只写一次**
- **代数等价**：精确算术下与 eager dense 更新完全相同，**不改权重、不改递推、不改 prefill**

**怎么处理 n+1**
**不直接解决份数**——它是"每份太大/写得太频繁"的解法。但它的**日志结构本身就是回滚机制**（截断指针），因此：
- 与 ReplaySSM/TreeWY **可叠加**（日志 = 因子缓冲）
- 更重要的是它给出了**"双 CUDA Graph"的工程范式**（§4）：append 与 merge 语义不同，**无法用一个 CUDA Graph 表达**，所以 warmup 时**按 uniform decode-batch descriptor 捕获两个图**，用 **phase 编码进 graph-cache key** 来选择。**这正是 Ascend ACL Graph 上做同样事情的标准做法。**

**实测**
- GDN：H200 上比 FLA 快 **1.34–1.60×**（B∈{64..512}），4090 上 **1.69–1.86×**；比 vLLM dense kernel 快 **1.26–1.42×**
- KDA：1.19–1.60× / 1.30–1.55×；RWKV6：1.52–1.69× / 1.41–1.63×
- **写流量降 7.83×**（GDN, B=128, M=8：dense 每 token 写 0.230 GiB → 摊销后 0.0294 GiB）
- 端到端：Qwen3.6-35B-A3B **1.08–1.20×**、Kimi-Linear-48B **1.05–1.12×**
- merge 间隔 `M` 常选 **4 或 8**（更大则 replay 成本吃掉收益）

**开源代码**：❌ **未找到公开代码仓**
- 自述是在 **vLLM-based prototype** 里做的（替换 dense decode 路径），kernel 是独立 **Triton** 实现；未见 repo/PR。

**引擎落地**：无 vLLM/SGLang 上游 PR（截至 2026-10-29 未检索到）。

**代价与局限**
- **只覆盖"一次一个 token"的 decode**，论文明确不处理投机解码的多 token 窗口
- append 步要**多读**一份 live log（读流量升：0.253 → 0.273 GiB），收益全部来自写路径
- 对 KDA / RWKV6（per-key 衰减）收益小：per-key decay 需要 `O(BH d_k)` 元数据，且 split replay 反而变慢
- 需要两套图 + phase 管理

**Ascend 移植要点**
- 这是**最容易被 Ascend 直接受益**的一篇：Ascend 的 HBM 带宽相对算力更紧，"减少写回"的收益通常比 GPU 更大。
- `M`（merge 间隔）在 Ascend 上应**结合 UB 容量重新标定**（日志要能放进 UB 才好）。
- 与投机解码叠加时，`M` 与接受率耦合，建议做成**运行时自适应**。

---

### 3.6 DAMP：Decay-Aware Mixed-Precision Recurrent-State Quantization

**论文**
- arXiv: **2608.27513** — <https://arxiv.org/abs/2608.27513>
- 作者：Tao Zhang, Jianchao Tan, Pingwei Sun, Yanqi Yu, Zunhai Su, Zixu Jiang, Yuchen Xie, **Xunliang Cai**（美团）+ Ziqian Zeng（华南理工）+ 华东师范
- 一句话：*"state 的离群值集中在少数 key channel，且衰减慢的 channel 会长期保留量化误差 —— 所以按 '误差 × 保留时长' 离线挑出高风险 channel，固定预算下给它们 FP16、其余 INT8。"*

**解决什么问题**
GDN/KDA 的 state 通常存 **FP32**，在长上下文/大 batch 下是内存大头（SGLang 下 Qwen3.6-35B、B=256：**估算 80 GB** 仅 recurrent state，含 active state 与 prefix-cache checkpoint）。但**均匀量化会崩**：
- INT8 / FP8：复杂推理精度下降
- **INT4 / NVFP4：几乎归零**（论文原话 "collapse it to near zero"）
- 即使上 **stochastic rounding**：Qwen3.6-35B 的 AIME 2026 从 18.48% → 63.98%，仍远低于 FP32 的 **85.46%**

**核心思路（三条实证观察 → 一个方案）**
1. **离群值集中在特定 key channel 和 value 维度**（Figure 2a–b）
2. 沿 value 维预处理后，**INT8 误差仍集中在少数 key channel**：**12.5% 的 key channel 贡献了 GDN ≈69% / KDA ≈76% 的总平方误差**
3. **衰减强度差异巨大**，而"衰减慢"意味着误差被长期保留；关键发现是——**跨任务、跨 prompt，慢衰减的 GDN head / KDA key channel 基本是同一批**（Figure 2d–f）⇒ **可以离线标定**

⇒ **DAMP**：离线用"量化误差 × 衰减保留率"给每个 key channel 算风险分，在**固定存储预算**下把 top-risk channel 保 **FP16**，其余 **INT8**（per-key-channel 独立 scale，channel 内分组共享 scale）。

**结果**：**9.9 bit/state value** 时平均精度接近 FP32；SGLang 中 state 存储 **−69.1%**、state update kernel 最快 **2.59×**、TPOT 最多 **−19.0%**。测试模型：Qwen3.6-35B、Kimi-Linear-48B、**Kimi-K3**。

**怎么处理 n+1**
**不解决份数，但直接乘在份数上**：`n+1` 份 × 0.31 ≈ **3.2× 等效收益**，且与因子化方案正交（因子日志也可以用同样预算压缩）。

**开源代码**：❌ **未找到公开代码仓**（搜 "DAMP-specreason" 等无结果）

**引擎落地**：**SGLang**（论文称已实现并给出 TPOT 数据）；vLLM / vllm-ascend 未见。

**代价与局限**
- **离线标定**：需要校准数据集 + 每模型一套 risk 排序（可迁移性论文只验证了同家族）
- **不是无损**：9.9 bit 下"接近"FP32，不是等于；对强推理任务仍需逐模型验证
- 混合精度 ⇒ kernel 里有 branch + 两套累加路径，**对向量化不友好**（这也是为什么它在 GPU 上 kernel 收益只有 2.59×）

**Ascend 移植要点**
- ⭐ **昇腾原生支持 FP16/BF16/INT8 混合**，且 `aclnn` 有 per-channel 量化 GEMM；**"少数 channel 保高精度 + 其余 INT8"在 Ascend 上可以映射成"两次 GEMM + mask 累加"**，但会牺牲 Cube 利用率 ⇒ 建议做成**按 head 分组的两个 kernel**（高精度 head 走一组、低精度 head 走一组），而不是同 kernel 内混。
- Ascend 的 HBM 容量（A2 单卡 64GB）通常是**先撞墙**的约束，所以 DAMP 的收益在 Ascend 上可能**比 GPU 更大**（省内存 = 上更多并发）。

---

### 3.7 DASC：Decay-Aware State Compression for Hybrid Linear-Attention Serving

**论文**
- arXiv: **2608.30386** — <https://arxiv.org/abs/2608.30386>
- 作者：ECNU + 美团（Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu 等）
- 一句话：*"不同 head/channel 的 '信息保留时长'（retention horizon）差异巨大 ⇒ 只把长保留期的单元存进 prefix-cache checkpoint，其余零填充或从有界后缀刷新。"*

**解决什么问题**
**注意：DASC 不是投机解码工作**，它解决的是 **prefix-cache 的 state checkpoint 爆炸**：
> 混合模型做前缀复用时，必须在 full-attention KV 之外**额外存 linear-attention 的 state checkpoint**；全量存 ⇒ 内存压力 ⇒ 更多驱逐 ⇒ 重复 prefill。

> **与我们的关系**：投机解码的 `n+1` 份状态 + prefix-cache 的 checkpoint 是**同一块 HBM 上的两个竞争者**。TreeWY 的实验就显示"删掉快照释放的 HBM 转化为更高并发/更低 TTFT"。DASC 提供的是**让 checkpoint 变便宜**的手段，从而间接让"多存几份"变得可行。

**核心思路**
1. 从 GDN/KDA 的**衰减结构**推导每个 head/channel 的 **retention horizon**（从权重即可算，不需数据）
2. 把长保留期的单元**打包成 ragged（不规则）的 state checkpoint**
3. 可选：在 **TP rank 之间均衡**这些 checkpoint 的存储
4. 复用时两种恢复策略：
   - **zero-fill**：省略的单元填 0（便宜，精度略降）
   - **suffix refresh**：用**有界后缀**重算刷新省略单元（多花一点算力换精度）

**结果**：Kimi-Linear / Qwen3-Next / Kimi-K3；**1.32–2.22× checkpoint 压缩**，mean TTFT **−16.1% ~ −42.0%**，mean E2E **−4.9% ~ −10.2%**；保守配置下质量接近全量 checkpoint，激进配置 + suffix refresh 可恢复精度。

**怎么处理 n+1**
不直接处理。但它是**唯一讨论了 TP 均衡**的一篇——对 Ascend 多卡 TP 部署（GDN 的 state 是 per-head 切分，天然可 TP 切）有直接参考价值。

**开源代码**：❌ **未找到公开代码仓**

**引擎落地**：**SGLang**（论文称已实现）；vLLM / vllm-ascend 未见。

**代价与局限**
- 近似，需要按模型挑"保守/激进"配置
- ragged 布局 ⇒ **kernel 变复杂、访存不连续**（对 Ascend 的 DMA 对齐尤其不友好）
- suffix refresh 需要在复用点做额外前向 ⇒ TTFT 不一定总是变好

**Ascend 移植要点**
- ragged checkpoint 在 Ascend 上会**打断 256B/512B 对齐**，DMA 效率下降。建议改成 **"按 head-group 的整体保留/丢弃"**（粒度粗一点但整齐），而不是 channel 级 ragged。
- **TP 均衡**这条可以直接用：Ascend 上 state 存储本来就按 head 切，均衡只是换一个 head→rank 的映射。

---

### 3.8 七个方案速查表

| 方案 | 核心机制 | 解决 | 减少 **份数** | 减少 **每份大小** | 减少 **写流量** | 树支持 | 公开代码 | 引擎落地 |
|---|---|---|---|---|---|---|---|---|
| **ReplaySSM** | checkpoint + 输入环缓 + output-only | 状态 I/O / 回滚 / 精度 | ✅ **T+1 → 1** | ❌ | ✅ ~2× | ❌ 链 only | ✅ **官方 vLLM fork** | TRT-LLM ✅ / vLLM(Mamba2) ✅ / vLLM(GDN) ❌ / FlashInfer ✅ / SGLang RFC |
| **TreeWY** | 树 WY 变换，一次三角求解 | 树无法共享快照 | ✅ **N+1 → 1** | ✅ 128×/head | ✅ | ✅（但慢） | ❌ | vLLM **私有 fork + RFC #54080** |
| **Bole** | 树闭式 + **Neumann** + value-tile | 树验证慢 + 瞬态内存 | ✅ **N+1 → 1** | ✅ | ✅ | ✅ 理论上最佳 | ❌（仅第三方 PR [sglang#46702](https://github.com/sgl-project/sglang/pull/46702)，chain-only） | SGLang（作者侧不可证实） |
| **SpecLA** | 三路分流 + 因子缓冲 + 延迟提交 | 拓扑不匹配 + 回滚 | ✅ **N+1 → 1** | ✅ | ✅ | ✅（chain-decomposed 折中） | ❌ | PyTorch/Triton 原型 |
| **DeltaLog** | base + 有界 update log | state-update tax | ❌ | ⚠️ 间接 | ✅ **7.83×** | ❌ | ❌ | vLLM 原型（双图） |
| **DAMP** | 关键 channel FP16 + 其余 INT8 | state 太大 | ❌ | ✅ **−69.1%** | ✅ | ➖ 无关 | ❌ | SGLang |
| **DASC** | retention horizon + ragged checkpoint | prefix-cache checkpoint | ❌ | ✅ **1.32–2.22×** | ➖ | ➖ 无关 | ❌ | SGLang |

---

## 4. 选型：为什么"因子日志 + 链"是 Ascend 的正解

### 4.1 决策树

```
你的 draft 是链（MTP / topk=1）还是树（EAGLE topk>1）？
│
├─ 链（MTP，绝大多数国产/开源 hybrid 模型的默认）
│   └─ ✅ checkpoint + 因子日志 + 指针回滚
│        （= ReplaySSM / TreeWY 的 mamba_state_commit="reconstruct"）
│        · 因果掩码 → ACL Graph 可整图捕获
│        · 已有生产验证（GLM-5.3-Flash / Kimi-K3）
│        · SGLang 有可直接参考的 GDN Triton kernel
│
└─ 树（要拿更高 acceptance）
    ├─ 想省事 → chain-decomposed（SpecLA 第三路）：链内串行、链间并行
    │            · 不需要非因果掩码 → 可捕获
    │            · 论文实测比 root-to-leaf 重放快 1.80–7.11×
    │
    └─ 要极限 → 树掩码 + Neumann 级数（Bole 式 6/8）
                 · 3–4 次固定形状 matmul，无三角求解 → Ascend Cube 友好
                 · 掩码只是"数值"，形状固定 → 理论上可捕获（需实测）
                 · 风险：Bole 未开源，需自研
```

### 4.2 排除法：为什么不是别的

| 候选 | 为什么不选（对 Ascend 而言） |
|---|---|
| **纯快照 `store_all`**（当前 vLLM/SGLang 默认） | 内存直接不可承受（35B/B=256/n+1=4 ≈ 30 GiB）；且每 draft 位置一次全量 state 写，**恰好在最需要省的路径上加倍 HBM 流量** |
| **decode-kernel replay**（重放递推） | 保留了逐 token 的 state 往返 = 投机解码想摊销的东西一点没摊销；树还要为每个分支准备不同状态 |
| **prefill-kernel 复用**（把 draft 当短 prefill） | 窗口太短（3–8 token），摊销不了 UT 变换 / layout conversion / 同步的 setup；且普通因果 prefill 的依赖模型与树不匹配 |
| **纯 tree-masked（TreeWY 式）** | 论文自己在 vLLM 上验证过：非因果祖先掩码 → 掉出整图捕获 → **比不做树还慢** |
| **DeltaLog（单 token）** | 只做 chain 的单 token decode，**没有覆盖投机窗口**（但其"双图"范式可借用） |
| **DAMP / DASC** | 不减份数，是**乘数**（叠加项），单用不解决问题 |

### 4.3 关键洞察：**保持"因果"是 Ascend 上的第一约束**

这是整份调研里最容易被忽略、但工程上最致命的一条：

- TreeWY 在 **B200 + vLLM** 上的实测：tree（`w>1`）**放弃 CUDA Graph 整图捕获**后，"*the cost far larger than the mask itself*"，最终 tree 只做到"enabled and correct, **not a speedup**"。
- Ascend 侧同一约束更强：`aclgraph`（`torch.npu.graph`）对**动态 shape、动态掩码、控制流**都更敏感；一旦掉图，decode 阶段的 host 侧下发开销会吃掉全部收益。
- vllm-ascend 自己的 GDN checkpoint/replay PR（#18765）也明确写：**"Primitive graph support does not enable an engine graph path."** —— 即算子能跑图，但引擎侧还没有图路径。

> ⇒ **第一版必须做"因果掩码的链"**。树留到第二期，且必须先把掩码做成"静态形状 + 数值变化"的形式（Bole 的 Neumann 形式正好满足）。

---

## 5. 引擎落地现状（逐条核验，2026-10-29）

### 5.0 时间线：这条技术路线只用了 5 个月就从论文走到生产

```
2026-06       ReplaySSM blog 发布；SGLang RFC #28511 开
2026-06-19    TensorRT-LLM #14203（Mamba2 MTP replay）     ✅ merged
2026-07-03    vLLM RFC #47572 + 初版 PR #47576              (PR 因 conflict 关闭)
2026-07-06    vllm-ascend #11236  MAX_MTP 8→16              ✅ merged  ← 最早的 Ascend 动作
2026-07-20    TensorRT-LLM #16464（GDN MTP replay, NVIDIA）  ✅ merged
2026-07-24    vLLM #48018（Mamba2 标准解码）                 ✅ merged
2026-08-11    FlashInfer #4081（GDN u/d cache spec decode）  ✅ merged
2026-09-01    vLLM #52506（FlashInfer 后端）                 ✅ merged
2026-09-30    vLLM #52928（FlashInfer MTP）                  ✅ merged
2026-10-14    vllm-ascend #18765（GDN checkpoint/replay）    🔄 open（已完成测试）
2026-10-24    vLLM #63045（KDA CuTe verify/commit）          ✅ merged
2026-10-26    vllm-ascend #19769（SketchSSM NPU）            🔄 open
2026-10-27    vllm-ascend RFC #19987（GDN RecoverSSM on MRV2）🔄 open  ← 现在
                                                                （今天 2026-10-29）
```
> **解读**：CUDA 侧已经收敛并进入生产；**Ascend 侧 GDN 正好停在"PR 已完成、RFC 刚开"的窗口**。
> 这是介入的最佳时机，但也意味着**窗口期只有几周**——上游随时可能合入。

### 5.1 vLLM 上游（vllm-project/vllm）

**已合入 main 的能力**（我用 `raw.githubusercontent.com` + `gh api` 直接核对了 `vllm/config/cache.py`、`vllm/config/vllm.py`、`vllm/v1/attention/backends/mamba_attn.py`）：

```python
# vllm/config/cache.py
replayssm_buffer_len: int = Field(default=16, gt=0)
"""ReplaySSM logical history length B for Mamba2."""
use_replayssm: bool = False
"""Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip
   the per-step state write-back."""
use_kda_recoverssm: bool = field(default=False, init=False)
"""Whether Kimi-K3 KDA uses RecoverSSM speculative decode."""
```

**命名很重要**：上游把 ReplaySSM 这条线叫 **RecoverSSM**（`--use-replayssm` 是开关名）。

**main 里已经有的通用框架**（`gh search code` 命中 27 个文件）：
```
vllm/v1/worker/gpu/model_states/recoverssm.py        ← RecoverSSMState：协调 attention 与 postprocess 的元数据
vllm/v1/attention/backends/recoverssm_metadata.py    ← RecoverSSMMetadata
vllm/v1/worker/gpu/model_states/mamba_hybrid.py      ← 集成点
vllm/models/kimi_k3/nvidia/recoverssm/cutedsl/verify.py   ← CuTeDSL verify kernel
vllm/models/kimi_k3/nvidia/recoverssm/cutedsl/commit.py   ← CuTeDSL commit kernel
vllm/models/kimi_k3/nvidia/ops/recoverssm.py
```
> 一个很有意思的细节：`RecoverSSMState.record_step(..., for_capture: bool)` —— 图捕获时直接 `self._step = None` 跳过记录。**说明上游已经把"图捕获与 recover 记录的互斥"写进了设计**，Ascend 侧应当照抄这个接口形态。

模型级接线状态：
- **KDA** ✅ 已进 main（自动对 `KimiLinearForCausalLM` / `KimiK3ForConditionalGeneration` 推导，`num_speculative_tokens>0` 时）；
- **GDN** 🔄 **通用框架在了，Qwen 的接线还在 PR 里**（#61309 / #58863）。

**约束（`validate_mamba_cached_kernel`，可直接抄到 Ascend 的 config 校验里）**：

| 场景 | 约束 |
|---|---|
| KDA RecoverSSM | `--mamba-backend triton`；`pipeline_parallel_size=1`；不支持 `--enable-mamba-cache-stochastic-rounding`；align 模式 + prefix caching 时需 `VLLM_USE_V2_MODEL_RUNNER=1` |
| Mamba2 ReplaySSM + 投机 | `mamba_cache_mode` 必须 `none`（即**禁 prefix caching**）；`replayssm_buffer_len >= 1+num_speculative_tokens`；FlashInfer 后端要求 `buffer_len <= 16` |
| 通用 | `--use-replayssm` 需模型架构声明 `supports_replayssm` |

**Mamba 后端里的环形缓冲机制**（`mamba_attn.py`）：
```python
write_pos_cpu = torch.remainder(decode_steps_cpu, self.replayssm_buffer_len)
is_flush_cpu  = write_pos_cpu == self.replayssm_buffer_len - 1
```
即 **per-row ring cursor + flush flag**，`B_cache` 形状 `(ngroups, replayssm_buffer_len, dstate)`。**这就是"n+1 份状态"被压成"1 份 + L 条记录"的落地形态。**

**GDN RecoverSSM 的进行中 PR（全部 open，未合入）**：

| PR | 标题 | 状态 |
|---|---|---|
| [#61309](https://github.com/vllm-project/vllm/pull/61309) | `[Spec Decode][Qwen3.5] RecoverSSM for Qwen3.5 / Qwen3-Next GDN` | 🔄 open |
| [#58863](https://github.com/vllm-project/vllm/pull/58863) | `[Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv` | 🔄 open |
| [#62625](https://github.com/vllm-project/vllm/pull/62625) | `[Spec Decode][Qwen GDN] Keep RecoverSSM replay records out of the Mamba page` | 🔄 open |
| [#61017](https://github.com/vllm-project/vllm/pull/61017) | `RecoverSSM: commit all GDN KV cache groups with one set of launches` | 🔄 open |
| [#59366](https://github.com/vllm-project/vllm/pull/59366) | `Run GDN RecoverSSM verify in the fused CUDA MTP kernel` | 🔄 open |
| [#59983](https://github.com/vllm-project/vllm/pull/59983) | `RecoverSSM: support pipeline parallelism` | 🔄 open |
| [#59982](https://github.com/vllm-project/vllm/pull/59982) | `[XPU][Spec Decode] RecoverSSM: route Qwen GDN through Triton/FLA` | 🔄 open |

**已合入的 KDA 侧（可作模板）**：

| PR | 内容 | 状态 |
|---|---|---|
| [#63045](https://github.com/vllm-project/vllm/pull/63045) | `[Kimi K3] KDA RecoverSSM: CuTe verify and commit kernels, records outside the KV cache` | ✅ merged 2026-10-24 |
| [#61358](https://github.com/vllm-project/vllm/pull/61358) | `Enable RecoverSSM for GLM5Next KDA` | ✅ closed(merged) |
| [#61566](https://github.com/vllm-project/vllm/pull/61566) | `[Spec Decode][GLM-5.3-Flash] Support KDA RecoverSSM (--use-replayssm)` | 🔄 open |

**没有 RecoverSSM 时，vLLM 靠什么撑 GDN 投机解码？** → `mamba_cache_mode=align`
- [#33024](https://github.com/vllm-project/vllm/pull/33024) / [#33705](https://github.com/vllm-project/vllm/pull/33705)：在 align 模式下启用投机解码
- 机制：**在每个 block 边界 materialize 一份 state，并保留一个 block 的"后撤"**
- 已知遗留问题：[#53479](https://github.com/vllm-project/vllm/pull/53479)（open）"*materialize a state at every boundary and drop the speculative one-block back-off*"

> ⚠️ 也就是说：**vLLM main 上 GDN + 投机解码 = align 模式的 block 级快照**，本质仍是"多份状态"，只是粒度从 token 变粗到 block。这正是 RFC #19987 想在 Ascend 上替换掉的东西。

### 5.2 vLLM-Ascend（vllm-project/vllm-ascend）—— 已经是"半落地"状态

**已合入 main 的（KDA 线）**：
```
vllm_ascend/models/glm5next/recoverssm_config.py
vllm_ascend/patch/platform/patch_replayssm_config.py
vllm_ascend/patch/worker/patch_bind_kv_cache.py
vllm_ascend/patch/worker/patch_v2/patch_model_runner.py
vllm_ascend/worker/npu_input_batch.py
tests/ut/models/test_glm5next_recoverssm_config.py
tests/e2e/pull_request/four_card/test_glm5_3_flash.py
```
⇒ **Ascend 上已经有 GLM-5.3-Flash（KDA）的 RecoverSSM 端到端路径**，这是最好的"抄作业"起点。

**GDN 线的关键 issue/PR（全部 open）**：

| # | 标题 | 日期 | 为什么重要 |
|---|---|---|---|
| **[19987](https://github.com/vllm-project/vllm-ascend/issues/19987)** | **[RFC] Support GDN RecoverSSM for Speculative Decoding on Ascend** | **2026-10-27** | ⭐ **和你的目标完全一致**，且已经写好了 execution flow、验证清单、性能指标 |
| **[18765](https://github.com/vllm-project/vllm-ascend/pull/18765)** | `[Attention][Feature] Add opt-in GDN checkpoint/replay for eager MTP on Ascend` | 2026-10-14 | ⭐ **已经写完并测过的 GDN checkpoint/replay**，带 `enable_gdn_recover` 开关 |
| [17763](https://github.com/vllm-project/vllm-ascend/pull/17763) | `[GDN] Support adaptive verification for variable-length GDN` | 2026-09-29 | 变长验证 + DFlash 自适应预算 |
| [15207](https://github.com/vllm-project/vllm-ascend/pull/15207) | `[Ops] Add variable-length GDN operators` | open | 固定 2-D state row + ragged query |
| [19769](https://github.com/vllm-project/vllm-ascend/pull/19769) | `[SketchSSM] Ascend NPU backend for Mamba-2 SketchSSM` | 2026-10-26 | **另一条技术路线**（低秩 sketch，不是因子日志） |
| [18797](https://github.com/vllm-project/vllm-ascend/issues/18797) | `[RFC]: Tree Speculative Decoding` | 2026-10-15 | ⭐ **Ascend 官方的树方案 RFC**，GDN 的思路是"**每条链各自更新自己的状态，只保留被接受的链**"（per-chain state + recompute-on-accept）——注意这**不是** Bole 的闭式解，而是更朴素的"多链并行 + 接受后重算" |
| [15149](https://github.com/vllm-project/vllm-ascend/issues/15149) | RFC：变长验证打破定长 `[B,K+1]` 表 | open | 见 §6.1 第 8b 条 |
| [19603](https://github.com/vllm-project/vllm-ascend/pull/19603) / [19604](https://github.com/vllm-project/vllm-ascend/pull/19604) | 线性注意力 verify 扩展 / speculator hooks（非链式验证） | open | ⭐ **树验证的插件挂载点**（镜像 vLLM #62962/#62963） |
| [17393](https://github.com/vllm-project/vllm-ascend/pull/17393) | Add FLA RecurrentGDN Backend | open | 实测 recurrent GDN decode **538µs → 148µs** |
| [17805](https://github.com/vllm-project/vllm-ascend/pull/17805) | H-Spec drafter | open | H-Spec（2609.24197）已移植到 Ascend（910B3 验证） |
| [19898](https://github.com/vllm-project/vllm-ascend/pull/19898) | `[Ops] GDN decode fuse gating + recurrent update` | open | 算子级融合 |
| [17604](https://github.com/vllm-project/vllm-ascend/pull/17604) | `[Perf][GDN] fused GDN prepare Triton kernel for Ascend` | open | prepare 阶段融合 |
| [19133](https://github.com/vllm-project/vllm-ascend/pull/19133) | `[Feature] Add experimental GLM KDA RecoverSSM for Model Runner V2` | open | KDA 的 MRV2 版本 |
| [18239](https://github.com/vllm-project/vllm-ascend/pull/18239) | `[Bug][MRV2] Qwen3.8-27B + MTP: GDN metadata KeyError ... during FULL_DECODE_ONLY capture` | open | **MTP + 图捕获的现成坑** |

**#18765 的关键硬数据（可直接引用）**：
> "*For 48 value heads, K=V=128 and MTP3, **SSM plus records are approximately 3.188 MiB versus 12 MiB for four snapshots**.*"

⇒ **3.8× 的 state 内存下降**，且是在 **910B2 / CANN 9.1.0 / Triton Ascend 3.2.2 / FLA NPU A2** 上实测的。这直接证明了"因子日志"在 Ascend 上可行。

**#18765 的限制（= 我们下一步要突破的地方）**：
- 需要 **A2/A3、单卡（TP/PP/DP/PCP/DCP=1）**
- **eager Model Runner V1**（V2 见 RFC #19987）
- dense Qwen3.5 家族 + eager native MTP，**1–5 个 draft token**
- BF16 model/conv + **FP32 SSM**
- **prefix caching 关闭**、`mamba_cache_mode=none`
- 拒绝 KV transfer 与 sleep mode
- **算子支持图，但引擎没有图路径**

### 5.3 SGLang —— GDN 侧其实**领先** vLLM

SGLang main 里已经有 **ReplaySSM 的 GDN 投机解码 kernel**（75 个文件命中 `replayssm`）：

```
python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py   ← 核心
python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_fold.py
python/sglang/kernels/ops/attention/linear/kda_replayssm_spec_decode.py
python/sglang/kernels/ops/attention/linear/fused_recurrent_linear_replayssm.py
python/sglang/kernels/ops/attention/helion/kda_replayssm.py
benchmark/kernels/attention/bench_gdn_replayssm_decode.py
```

**这份文件是整个调研里最有价值的工程参考**，其 docstring 直接写清了设计：

> *"Ported from the vLLM ReplaySSM reference implementation (`github.com/Johnny-Liou/ReplaySSM`, commit `3c85112`, `vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py`) … the recurrent intermediate-state snapshot (full `[V, K]` state written per draft token to `intermediate_ssm`) is replaced by a per-slot **circular cache** of the last `L` committed steps' `(d, k, g)` records plus a frozen checkpoint `h0`. The verify output for the whole draft window is reconstructed **output-only** (never materialising the state) via the chunked delta-rule `(I+A)^{-1}` UT-transform, and the full state is flushed back only every `L` committed tokens. Rejected drafts roll back by a pointer move on the cursors — no state write-back."*
>
> *"**Linear-chain only**: the intra-window interaction uses a strictly-lower causal mask, so this kernel is valid only for linear draft chains (`speculative_eagle_topk <= 1`). **Tree verify (topk > 1), KDA, and NPU/CPU fall back to the recurrent verify kernel.**"*

**Tensor 布局（直接可抄）**：
```python
h0        : [num_slots, HV, V, K]      # 只读 checkpoint
d_cache   : [num_slots, HV, L, V]      # 分块 delta（仅用于输出重建）
k_cache   : [num_slots, H,  L, K]      # L2 归一化 key
g_cache   : [num_slots, HV, L]         # fp32 衰减
rawv_cache, rawk_cache, beta_cache     # exact replay 用
```
**kernel 拆分**（4 个 Triton kernel）：
1. `gdn_replayssm_spec_circular_kernel` —— 验证 + output-only 重建
2. `gdn_replayssm_exact_fold_kernel` —— 精确折叠（把日志折进 checkpoint）
3. `gdn_replayssm_compact_commit_kernel` —— 紧凑 commit（update 向量与归一化 key 拆高低位）
4. `_advance_gdn_spec_cursors_kernel` —— 游标推进（**纯元数据，几乎零成本**）

**合入路径（已核验）**：

| 部分 | PR | 状态 |
|---|---|---|
| RFC | [sglang#28511](https://github.com/sgl-project/sglang/issues/28511) | closed 2026-07-21 |
| Part A：标准 decode | [sglang#28451](https://github.com/sgl-project/sglang/pull/28451) | ✅ merged 2026-06-26（+1771/−14） |
| Part B：**投机验证** | [sglang#28695](https://github.com/sgl-project/sglang/pull/28695) | ✅ merged 2026-07-20（+1577/−48） |

**对外开关**（`python/sglang/srt/arg_groups/fields/exec_.py:575-596`）：
```
--enable-linear-replayssm            # 标准 decode
--enable-linear-replayssm-spec       # 投机验证
--linear-replayssm-cache-len         # 默认 16；要求 2 的幂，且 L >= 2*num_draft_tokens
```

**默认路径有多贵**（`srt/mem_cache/memory_pool.py`）：
```python
intermediate_ssm  # 形状 [num_layers, size+1, speculative_num_draft_tokens, HV, V, K]
# 源码注释原文："the dominant spec scratch, ~46x the conv state"；"~9GB @ K3 dspark γ=7"
```
⇒ 开 `--enable-linear-replayssm-spec` 后该 buffer **直接设为 `None`**，改为 per-slot ring（`replayssm_d/rawv/rawk/beta`）。**"46× conv state" 这个数字比任何论文都更能说明 n+1 的代价。**

**Bole 在 SGLang 的"集成"无法证实**：仓库/分支/作者 PR 都找不到；论文里唯一的 SGLang 链接是它的基线版本 `sgl-project/sglang/tree/release/v0.5.12`，所谓 "≈6.2 kLoC" 是私有代码。唯一公开实现是**第三方 PR** [sglang#46702](https://github.com/sgl-project/sglang/pull/46702)（open，2026-10-27），且其 body 明确写 "**Not supported yet … EAGLE tree verify**" —— **即也是 chain-only**。

**DAMP / DASC** 只把 SGLang 当 eval harness 引用，**无任何 SGLang PR**。

### 5.4 TensorRT-LLM / FlashInfer

| 系统 | 内容 | PR |
|---|---|---|
| TensorRT-LLM | Mamba2 MTP checkpointing replay | [#14203](https://github.com/NVIDIA/TensorRT-LLM/pull/14203) ✅ |
| TensorRT-LLM | **GDN MTP replay**（NVIDIA 独立贡献） | [#16464](https://github.com/NVIDIA/TensorRT-LLM/pull/16464) ✅ |
| FlashInfer | Mamba2 checkpointing SSU（两 kernel 拆分 + ring buffer) | [#3975](https://github.com/flashinfer-ai/flashinfer/pull/3975) ✅ |
| FlashInfer | **GDN u/d cache spec-decode kernel** | [#4081](https://github.com/flashinfer-ai/flashinfer/pull/4081) ✅ |
| FlashInfer | prefix 场景 materialize | [#4815](https://github.com/flashinfer-ai/flashinfer/pull/4815) ✅ |
| FlashInfer | GDN vcache spec-decode（fused verify+flush） | [#4140](https://github.com/flashinfer-ai/flashinfer/pull/4140) 🔄 |

> **结论**：CUDA 生态（TensorRT-LLM + FlashInfer + vLLM KDA + SGLang GDN）**已经收敛到 RecoverSSM/ReplaySSM 这一条路线**。Ascend 的 KDA 版已在 main，**GDN 版正是当下的空白**。你要做的不是"发明方案"，而是**把已验证的方案在 Ascend 上补齐并做好 kernel**——这反而是更容易出成果的位置。

### 5.5 vLLM/SGLang 里与"n+1 状态"直接相关的 issue 清单

| 系统 | # | 标题 | 说明 |
|---|---|---|---|
| vLLM | [53479](https://github.com/vllm-project/vllm/pull/53479) | Mamba align: materialize a state at every boundary and drop the speculative one-block back-off | align 模式的快照粒度问题 |
| vLLM | [52942](https://github.com/vllm-project/vllm/pull/52942) | Fix stale mamba/GDN states with ngram spec decode on hybrid | 状态陈旧 = 回滚没做对 |
| vLLM | [56524](https://github.com/vllm-project/vllm/pull/56524) | Zero mamba state blocks on (re)allocation to stop stale-state poisoning | 同上 |
| vLLM | [64303](https://github.com/vllm-project/vllm/pull/64303) | Keep align-mode Mamba state in fixed block-table slots on MRV2 | 布局稳定性 |
| vLLM | [59933](https://github.com/vllm-project/vllm/pull/59933) | RecoverSSM align mode commits the final SSM state to an unwritten block at exact block boundaries (floor vs ceil-1) | **commit 边界的 off-by-one 坑**（Ascend 一定会遇到） |
| vLLM | [59962](https://github.com/vllm-project/vllm/pull/59962) | Correct aligned state indices at exact block boundaries | 同上 |
| vLLM | [54495](https://github.com/vllm-project/vllm/pull/54495) / [57828](https://github.com/vllm-project/vllm/pull/57828) | KDA: `--use-replayssm` 崩溃 / 无投机解码时应拒绝 | 配置校验 |
| vLLM | [56082](https://github.com/vllm-project/vllm/pull/56082) | Reduce RecoverSSM state commit overhead | commit 开销仍有优化空间 |
| vllm-ascend | [18239](https://github.com/vllm-project/vllm-ascend/pull/18239) | Qwen3.8-27B + MTP: GDN metadata KeyError during FULL_DECODE_ONLY capture | MTP + 图捕获 |
| vllm-ascend | [15831](https://github.com/vllm-project/vllm-ascend/issues/15831) | Qwen3.5-35B + graph mode accuracy issue on A5 | 图模式下精度问题 |
| vllm-ascend | [18483](https://github.com/vllm-project/vllm-ascend/pull/18483) | Document and test configurable SSM cache dtype on Ascend | dtype 选择 |
| vllm-ascend | [19608](https://github.com/vllm-project/vllm-ascend/issues/19608) | **AICore MTE invalid GM access 崩溃：GDN spec-decode + MTP + `FULL_DECODE_ONLY`** | ⭐ 崩在 `gdn_attn_builder._remove_spec_graph_padding_queries`，placeholder 填充的 spec batch（`{req: [-1,-1,-1]}`）触发 —— **这就是"图 + 投机 + 填充行"三者的交汇处，期 1 一定会撞上** |
| vllm-ascend | [19845](https://github.com/vllm-project/vllm-ascend/issues/19845) | `[RFC] Extend adaptive verification support across ascend attention backends` | 自适应验证要铺到各 attention backend，GDN 是其中之一 |
| vllm-ascend | [9508](https://github.com/vllm-project/vllm-ascend/issues/9508) | `[RFC] Support Hybrid Prefix Cache in Ascend Store`（align 模式） | 原文："*The number of KV heads of linear-attention is different from that of full attention. Currently, the logic that the same KV head in Ascend Store reuses the state cannot meet the requirements of linear-attention.*" |

---

## 6. Ascend 实现方案（本文最重要的一节）

### 6.1 昇腾与 GPU 的 8 个关键差异（决定了方案不能照搬）

| # | 差异 | 对方案的影响 |
|---|---|---|
| **1** | **`MAX_MTP` 是编译期 tiling 常量**，不是运行时分配 | vllm-ascend [#11236](https://github.com/vllm-project/vllm-ascend/pull/11236)（**已合入**）把 `recurrent_gated_delta_rule` 的 `MAX_MTP` 从 **8 提到 16**，因为 "*The old cap of 8 blocked MTP-3/5/7 speculative-decoding block sizes*"。**⇒ Ascend 上"n+1"这条轴是 UB 预算问题，不是 allocator 问题。** |
| **2** | **UB 是硬约束且有 host/kernel 两套账** | 同 PR 的评审发现：arch35 kernel 从 `tmpBuff` 多切了 `gamaInUb`，但 host 侧 `CalcWorkingUbBytes` 没算 → `MAX_MTP` 翻倍后 **UB 溢出 / 静默数据损坏**。任何加宽 verify 窗口的改动**必须同步改 tiling 计算**。 |
| **3** | **三角求解"有"，但它是串行 fp32 的延迟地板** | ⚠️ 更正一个常见误解：**CANN 没有原生 batched `solve_tril`，但 FLA 已经把它带到了 Ascend** —— `fla/ops/utils/triton_ascend/solve_tril.py`（已实测存在），`vllm_ascend/ops/triton/fla/` 有完整 Triton-Ascend 版，`fla_npu`（天津大学）还有 AscendC 版。**问题是它的实现是"16 步串行 fp32 代入"，本身就是延迟下限**，且 in-tree 版本要 `extract_slice`/`insert_slice` 来躲 UB OOM。⇒ 所以选型不是"能不能求逆"，而是"**要不要用 Neumann（D≤4 次 matmul）换掉这个串行 16 步**"。 |
| **4** | **Triton-Ascend 只支持部分语义** | 实例：vllm-ascend [#19769](https://github.com/vllm-project/vllm-ascend/pull/19769) 实测 "*`SKETCH_DOT_PRECISION="tf32x3"` … **triton-ascend accepts only `('ieee','hf32')`***" ⇒ 上游 kernel 不能直接编译，必须改写。 |
| **5** | **ACL Graph 对动态 shape / 动态掩码更敏感** | TreeWY 在 CUDA Graph 上已经踩过"非因果掩码 ⇒ 掉图 ⇒ 变慢"；Ascend 侧 vllm-ascend 直接说 "*Primitive graph support does not enable an engine graph path*"（#18765）。**树必须先做成静态形状。** |
| **6** | **误差/精度路径不同：SSM state 惯用 FP32，Cube 吃 BF16** | #18765 的配置是 "BF16 model/conv + **FP32 SSM**"。FP32 matmul 在 Cube 上要拆成多次，**建议：state 存 FP32、tile 内 cast 到 BF16 做 matmul、结果回 FP32 累加**，并用 DAMP 式混合精度补精度。 |
| **7** | **HBM 容量是首要瓶颈（A2 单卡 64GB）** | 35B 模型 bf16 权重 ~70GB 已经要 TP≥2；state 再吃几十 GB 会直接压并发。**这让 §3 的"减少份数"收益比在 GPU 上更显著**。 |
| **8a** | **state gather/scatter 是一等开销，且快路径只在新硬件** | 融合的 `KDAStateCopyPlan` 只在 **A5/950** 上可用，**A3 回落到 masked byte-copy**；且**重复的 scatter 目标会静默竞争**（不同请求映射到同一 slot 时不报错）。⇒ 因子日志的 commit 必须保证 slot 映射唯一，并加断言 |
| **8b** | **变长验证会打破定长 `[B,K+1]` 表** | vllm-ascend [#15149](https://github.com/vllm-project/vllm-ascend/issues/15149)（RFC，结构化输出/D-Cut 会按请求裁剪 draft 长度）原文："*Once different requests retain different draft-prefix lengths, the native fixed-length metadata can **update the wrong state position or state slot***" ⇒ **这是"自适应投机长度"在 Ascend 上的真实拦路虎**，比 GPU 更严重（因为 slot 是编译期表） |
| **8** | **移动/边缘 NPU 的额外约束** | Tandem（arXiv:2610.26660，手机 NPU）实测指出："*Retaining rollback states for speculative decoding can also **move prompt processing off the NPU's matrix unit**, adding overhead*" ⇒ **回滚状态若放到 Vector 上做，会抢走 Matrix 单元的流水**。这提醒我们：**commit 阶段的重建必须走 Cube，不能退化成 Vector 上的逐元素拷贝。** |

**还有一条必须知道的"前车之鉴"**：SGLang 的 NPU 混合 GDN 路径目前用的是 **`conv_state_rollback`（把 conv_state 右移 N−k 位）** 这个 hack，而 [sglang#25587](https://github.com/sgl-project/sglang/issues/25587) 明确指出它**在数学上是错的**：
> "*The current code calls conv_state_rollback to right-shift conv_states by (N−k) positions … This is **mathematically incorrect** … SiLU is **non-linear and non-invertible** … this intermediate state is **not preserved anywhere**.*"

⇒ **教训：任何"回滚"都不能靠"逆运算"，只能靠"保存因子/重算"。** 这与 §1.1 的第 2 条性质完全一致。（该 issue 已 closed，但修复是否落地**未核实**——如果你在 Ascend 上沿用 NPU 路径，务必先验这一点。）

### 6.2 三条可选落地路径

| 路径 | 做法 | 工作量 | 风险 | 建议 |
|---|---|---|---|---|
| **A. 接续 vllm-ascend 已有资产** | 在 #18765（MRV1 GDN checkpoint/replay）+ RFC #19987（MRV2）基础上补齐：图路径、变长验证、多卡 | ★★ | 低（已验证 3.188 MiB vs 12 MiB） | ✅ **主线** |
| **B. 移植 SGLang 的 GDN Triton kernel** | 把 `gdn_replayssm_spec_decode.py` 的 4 个 kernel 改写成 Triton-Ascend / AscendC | ★★★ | 中（Triton-Ascend 语义差异） | ✅ **并行做**，作为 A 的 kernel 参考 |
| **C. 自研树（Neumann）** | 在 A/B 之上加 tree mask + `Σ(−G)^m` | ★★★★ | 高（需自证正确性） | ⏳ **第二期**，且必须先解决静态形状 |
| **C′. 走插件接口（比 C 更正确）** | 不做 fork，而是对接**上游已开的插件机制**：vLLM [#62962](https://github.com/vllm-project/vllm/pull/62962) 的 `LinearAttnVerifyExtension`（GDN/Mamba-2 把 spec 行以**零拷贝类型化视图**交给树外 verifier）+ 方法注册表；[#62963](https://github.com/vllm-project/vllm/pull/62963) 的 MRV2 speculator hooks（自定义 rejection sampler、accepted-path hooks）。Ascend 侧已镜像：[#19603](https://github.com/vllm-project/vllm-ascend/pull/19603) / [#19604](https://github.com/vllm-project/vllm-ascend/pull/19604)（`vllm_ascend/worker/v2/spec_decode/hooks.py`），并挂在 RFC [#19406](https://github.com/vllm-project/vllm-ascend/issues/19406)（EAGLE-4 移植）下 | ★★★ | 中 | ✅ **期 3 的首选形态：写插件，不改 `ops/gdn.py`** |

> **强烈建议**：不要从零开始。**vllm-ascend 已经有 KDA 的 RecoverSSM 端到端路径在 main 上**（`vllm_ascend/models/glm5next/recoverssm_config.py` 等 6+ 文件），GDN 的 PR #18765 也已完成并测过。你的增量价值在：**图路径 + 树 + 精度 + 多卡**，而不是重写因子日志。

### 6.3 推荐路线图（三期）

#### 期 1（4–6 周）：把链做对、做快、做进图

**目标**：GDN + native MTP（T=1..5）+ 单卡 A2/A3，`--enable-gdn-recover`，state 内存 **≥3.8×** 下降，**ACL Graph 可捕获**。

1. **对齐 #18765 的语义**（如果 PR 未合，直接复用它的设计）：
   - `h0` checkpoint `[slots, HV, V, K]` FP32，验证期**只读**
   - 环缓 `d_cache [slots, HV, L, V]` FP32、`k_cache [slots, H, L, K]` bf16、`g_cache [slots, HV, L]` FP32、`beta_cache` FP32
   - cursor per row：`write_pos = steps % L`，`flush = (write_pos == L-1)`
2. **kernel 拆分（4 个，与 SGLang 一一对应）**：
   | kernel | 作用 | Ascend 实现要点 |
   |---|---|---|
   | `gdn_spec_verify` | output-only 验证整个窗口，不 materialize state | 主体是 `Q̂·S0ᵀ`（Cube，`T×d_k @ d_k×d_v`）+ `(I+A)^{-1}` 求解 + `Attn·U` |
   | `gdn_exact_fold` | 把日志精确折进 checkpoint（flush） | 每 L 步一次，注意 UB 里放得下 `L×d_v` tile |
   | `gdn_compact_commit` | commit 接受路径（update 向量 + 归一化 key 拆高低位） | 走 Cube 的 `(D_a K_a)^T U_a`，**不要退化成 Vector 逐元素** |
   | `advance_cursors` | 纯元数据推进 | 放 host 或 device 小 kernel，**必须能被 graph 捕获（定长）** |
3. **图捕获**：采用 DeltaLog §4 的 **"双图"范式**（append 图 / flush 图），把 **phase 编码进 graph-cache key**。这是 Ascend 上唯一可行且已被 GPU 验证的做法。
4. **UB 预算**：`MAX_MTP` 与 `L` 必须**同时**进 tiling 计算。给出一个示例预算（需实测修正）：
   ```
   d_k = d_v = 128,  FP32 → 单 head state = 64 KiB
   UB ≈ 192 KiB（A2 910B，公开资料，需实测）
   ⇒ 192 / 64 ≈ 3：最多驻留 3 个 head 的 state，且这是"裸"预算
   ⇒ 扣掉 q/k/v/o tile、beta/g 向量、Neumann 的 Z/U 工作区后，实际只有 1–2 个
   ⇒ 必须按 head 分核 + 按 value tile 切分（Bole 的 value-tiling）
   ```
5. **正确性门禁**：照抄 #18765 的做法——**独立 FP32 递推 oracle**，逐窗口/逐前缀对比；再做 `check_logprobs_close` 式的端到端比对（注意：BF16 baseline 本身在重排请求时会变 greedy 选择，这是**数值兼容**而非**逐位一致**）。

#### 期 2（4–6 周）：变长 + 自适应预算 + 多卡

1. **变长验证**：接 vllm-ascend [#15207](https://github.com/vllm-project/vllm-ascend/pull/15207)（变长 GDN 算子）+ [#17763](https://github.com/vllm-project/vllm-ascend/pull/17763)（adaptive verification, DFlash）
   - 关键：`fixed 2-D state row + ragged query_start_loc`，**state 仍定长**，只有 query 变长 ⇒ 图仍可捕获
2. **自适应验证预算**：直接复用 vllm-ascend [#15098](https://github.com/vllm-project/vllm-ascend/pull/15098)（已合入）的 confidence estimator + cost profiling + budget assignment，**不要自己再造一套**（#17763 明确说 "*does not add another controller, cost table, environment variable*"）
3. **多卡**：GDN state 按 head 天然可 TP 切；参考 DASC 的 **TP 均衡**思路。注意 vLLM 上游 RecoverSSM 目前 **PP=1**（#59983 在解），Ascend 先做 TP-only。

#### 期 3（6–8 周）：树 + 精度

1. **【优先】走插件接口，不做 fork**：vLLM [#62962](https://github.com/vllm-project/vllm/pull/62962)（`LinearAttnVerifyExtension` + 方法注册表）/ [#62963](https://github.com/vllm-project/vllm/pull/62963)（MRV2 speculator hooks）→ Ascend 镜像 [#19603](https://github.com/vllm-project/vllm-ascend/pull/19603) / [#19604](https://github.com/vllm-project/vllm-ascend/pull/19604)。**这决定了你的代码形态：写插件，而不是改 `vllm_ascend/ops/gdn.py`。**
   - 同时读 Ascend 官方树 RFC [#18797](https://github.com/vllm-project/vllm-ascend/issues/18797)：它的思路是"**per-chain state + 接受后重算**"，比 Bole 的闭式解朴素但实现快——**建议先落地它，再升级到 Neumann**
2. **树 + Neumann**：`U = Σ_{m=0}^{d} (−G)^m R`，`d` = MTP depth（3–4）
   - **形状固定**（`T_max × T_max` 掩码），只有**数值**随拓扑变 ⇒ **有希望保持图捕获**（这一点需要自己实测验证，是本文最大的"待证实假设"）
   - 与 TreeWY 的 forward substitution 相比，Neumann 在 Ascend 上**几乎肯定更快**（Cube 满负荷、无串行）
2. **精度**：DAMP 式"关键 key channel FP16 + 其余 INT8"
   - Ascend 映射建议：**按 head 分组两个 kernel**（高精度 head 一组、低精度 head 一组），而不是同 kernel 内 branch（Cube 讨厌 branch）
   - 先做 **FP32 → BF16 checkpoint + FP32 累加**（最保守），再评估 INT8
3. **prefix caching**：vllm-ascend [#9508](https://github.com/vllm-project/vllm-ascend/issues/9508)（Hybrid Prefix Cache in Ascend Store）已指出 "*linear-attn KV-head count breaks Ascend Store state reuse*"，**这是期 3 的真实拦路虎**，需先解决。

### 6.4 关键 kernel 的 Ascend 映射（伪代码级）

**A. output-only 验证（链）**

```python
# 输入: h0 [HV,V,K] fp32 (只读), d_cache [HV,L,V], k_cache [H,L,K], g_cache [HV,L] fp32,
#       beta_cache [HV,L], q/k/v [T,...], A_log, dt_bias
# 目标: o [T,HV,V]，且不写回 h0

# 1) 衰减与门控（Vector，FP32）
g   = cumsum(log_decay(T))                    # 沿 T 的累积衰减，长度 ≤16
beta= sigmoid(b)                              # 写入门
k_n = l2norm(k)                               # 归一化 key（GDN 要求）

# 2) 构造 UT 系统（极小矩阵，T×T ≤ 16×16）
A   = tril_mask(beta_i * (g_i/g_j) * (k_i · k_j), i>j)   # 严格下三角
M   = inverse(I + A)                          # ← 关键：见下

# 3) 输出重建（Cube 主导）
u   = M @ (beta * (v - (k_n @ h0)))           # [T,V]
o   = (g * q) @ h0.T  -  attn_part + u_part   # 主体 FLOPs
```

> **`M = inverse(I+A)` 的三种 Ascend 实现**（按推荐序）：
> 1. **Neumann**：`M ≈ Σ_{m=0}^{T-1} (−A)^m`，`T≤16` 时是 16 次 `16×16` matmul —— **小到可以忽略**，且形状固定、无分支、可图捕获。**推荐。**
> 2. **分块前代**：`T=16` 拆 4 个 4×4 块，3 轮 wavefront。AscendC 手写较繁琐。
> 3. **直接求逆**：`T≤16` 时数值上可用 Gauss-Jordan，但**没有原语**，要手写 Vector 循环，**不建议**。
>
> ⚠️ 注意 `T` 是 verify 窗口（≈1+MTP），不是树的节点数。**树的 `N` 可能是 16–64**，Neumann 的 `d`（深度）仍是 3–4，但 `G` 是 `N×N` ⇒ matmul 变成 `64×64 @ 64×128`，仍然很小。

**B. commit（走 Cube，勿走 Vector）**

```python
# 接受路径 a，祖先集 P(a)
S_new = P_a * S_pre + (D_a @ K_a).T @ U_a        # 一次 matmul
# Ascend: aclnnMatmul + 前置的 D_a 行缩放（Vector，O(N)）
```

**C. flush（每 L 步）**

```python
S_ckpt = fold(S_ckpt, d_cache[0:L], k_cache[0:L], g_cache[0:L])   # 一次 chunked 折叠
clear(d_cache, k_cache, g_cache)                                   # 只清 cursor，不清内存
```

### 6.5 数据布局（建议）

```
LogicalSlot s →
  ssm_state[s]      : [HV, V, K] fp32        # checkpoint（持久，等于现在的 state 槽）
  spec_d[s]         : [HV, L, V] fp32        # 因子日志：delta
  spec_k[s]         : [H,  L, K] bf16        # 归一化 key（跨 V head 共享）
  spec_g[s]         : [HV, L]    fp32        # 衰减
  spec_rawk[s]      : [H,  L, K] bf16        # exact replay 用（可选）
  spec_rawv[s]      : [HV, L, V] bf16
  spec_beta[s]      : [HV, L]    fp32
  cursor[s]         : int32                  # 必须是 device 侧（#18765: "device-side sampled-token context commits"）
```

**容量对比**（`HV=48, K=V=128, MTP3 ⇒ T=4, L=16`，对齐 #18765 的口径）：

| 表示 | 每 request 每层 | 说明 |
|---|---|---|
| `store_all`（4 份快照） | **12 MiB** | 现状 |
| checkpoint + records | **≈3.19 MiB** | #18765 实测数字（含 conv/history） |
| 理论下界（仅 h0） | 3.0 MiB（48×128×128×4B） | 因子日志只占 ~6% |

> ⇒ **因子日志的额外开销只有 ~6%**，其余 94% 是那份"必须存在"的 checkpoint。**所以真正的下一步优化空间在 checkpoint 本身（DAMP/DASC 方向），而不是日志。**

这一条推论很重要：**"n+1 份状态"问题在 Ascend 上被解决后，瓶颈立刻转移到"那一份 checkpoint 还是太大"**。这决定了期 3 应该投 DAMP 而不是继续优化日志。

### 6.6 正确性验证矩阵

| 维度 | 用例 | 判据 |
|---|---|---|
| 数值 | 独立 FP32 递推 oracle，逐前缀/逐窗口 | 相对误差 < 1e-5（FP32 路径） |
| 部分接受 | 接受 0/1/…/T 个 | 与 oracle 一致 |
| 首 token 即拒 | `a=0` | checkpoint **不变**（不能误写） |
| 连续多轮 | ≥3 轮 verify/commit | cursor 不漂移 |
| 多请求 | 不同 accepted count、乱序、slot 复用 | 无串扰 |
| 空行/填充行 | inactive request、graph padding | 不写脏数据 |
| 边界 | `steps % L == L-1`（flush 点）、block 边界 | **参考 vLLM #59933/#59962 的 floor-vs-ceil-1 off-by-one** |
| 图 | eager vs FULL_DECODE_ONLY | 输出一致 |
| 精度 | FP32 / BF16 / 混合 | KL 与下游精度 |

### 6.7 风险清单（按 概率×影响 排序）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | **UB 溢出 / 静默损坏**（加宽 T 或 L 时 host tiling 没同步） | 高 | 致命 | 照 #11236 的教训：**任何 `MAX_MTP`/`L` 改动必须同时改 `CalcWorkingUbBytes`/`CalcFixedUbBytes`**，并加 UB 断言 |
| R2 | **图捕获失败**（动态掩码/动态 shape） | 高 | 高（掉图后 host 开销吃掉收益） | 期 1 只做因果链；期 3 的树用"定形状 + 数值变"，并**先写 microbench 验证可捕获性再投入** |
| R3 | **commit 边界 off-by-one** | 中高 | 高（输出正确但性能/正确性偶发） | 直接抄 vLLM #59933/#59962 的修法；单测覆盖 `steps % L == 0/-1` |
| R4 | **Triton-Ascend 语义缺失**（`tf32x3`、`solve_tril`、动态循环） | 高 | 中 | 优先 AscendC / aclnn；Triton 只做原型 |
| R5 | **FP32 gemm 效率低** | 中 | 中 | state 存 FP32、matmul 用 BF16 + FP32 累加；配 DAMP 补精度 |
| R6 | **prefix caching 不兼容** | 高 | 中（限制部署形态） | 期 1/2 学 #18765 直接禁掉；期 3 解 #9508 |
| R7 | **多卡 TP 下 state 切分与日志一致性** | 中 | 中 | 先 TP-only + 每 rank 独立日志；参考 DASC 的 TP 均衡 |
| R8 | **接受率下降导致负收益** | 低 | 中 | 用 no-spec reference gate；对比 acceptance length（TreeWY 实测差 <1%） |
| R9 | **上游接口变动**（vLLM GDN RecoverSSM 仍在 open） | 中 | 中 | 把 Ascend 实现做成 patch/plugin（学 #19769 的 `patch/worker/patch_*.py` 模式） |
| R10 | **已有 PR 抢跑**（#18765 / #19987 被合入） | 中 | 低（其实是好事） | **先联系作者/在 RFC 下认领子任务**，避免重复劳动 |
| R11 | **AICore MTE 非法 GM 访存崩溃**（spec + MTP + 图 + 填充行） | 高 | 高（直接崩） | 已有人踩过：[#19608](https://github.com/vllm-project/vllm-ascend/issues/19608)，崩点 `gdn_attn_builder._remove_spec_graph_padding_queries`。**先复现再动手**，并把"填充行不写 state"作为硬性单测 |
| R12 | **`solve_tril` 类算子的精度陷阱** | 中 | 中 | 固定尺寸 16×16/64×64 的求逆在 BF16 下条件数差；**必须在 FP32 里做 `(I+A)^{-1}`，或直接用 Neumann（无需求逆）** |

---

## 7. 其他最新/值得关注的方案

除了指定的 7 个，我按 arXiv API 全量扫了一遍 2025-09 ~ 2026-10 的相关工作。以下按"与 n+1 状态问题的相关度"排序。

### 7.1 强相关（直接处理 recurrent state × 投机解码）

| 论文 | arXiv | 核心思想 | 与我们的关系 |
|---|---|---|---|
| **LumoTree**（Path-Parallel Speculative Verification for Hybrid LMs） | [2609.23900](https://arxiv.org/abs/2609.23900)（2026-09-20） | ⭐ **唯一在服务系统里真正跑起来的 GDN 树验证**：FA2 tree bias + **branch-local GDN scan/replay** + device 侧 commit + **"只发布被接受链的 state"**（被拒分支变成非持久，而不是删除）。+27% token-weighted decode 吞吐 | **和 Bole/TreeWY 是同一目标的第三条实现**，且"accepted-chain-only state publication"这个提法比"reconstruct on commit"更精确。**注意：其等价性只做到 oracle-p-score closure，不是分布证明** |
| **KVBuffer**（IO-aware Serving for Linear Attention） | [2605.19049](https://arxiv.org/abs/2605.19049) | ⭐ 最强的"掀桌子"方案：**直接从一个缓冲的 K/V 窗口并行验证 draft，完全不 materialize 任何临时 state** ⇒ n+1 变 **0**。−45% LA decode 延迟，K=4 时提升服务容量 | 如果它成立，**"n+1 份状态"这个问题从根上不存在**。但它只在短窗口成立，长上下文仍需 checkpoint。**必须读，且要评估与因子日志的关系** |
| **Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers** | [2605.21325](https://arxiv.org/abs/2605.21325) | 系统比较 `inv(I − tril(diag(β)KKᵀ))` 的**直接法 vs 迭代法**、matrix-product-rich 变体、低精度数值稳定性，**并在 NPU 上与 SGLang 对标** | ⭐⭐ **整份调研里最"Ascend 相关"的一篇 kernel 论文**：它直接回答了 §6 的核心问题——**Neumann（迭代）vs 直接求逆，到底哪个在低精度/加速器上更稳更快**。期 3 之前必读 |
| **STEPQuant**（When and Where Errors Matter in Delta-Rule Recurrent State Quantization） | [2609.38169](https://arxiv.org/abs/2609.38169) | 空间×时间的 state PTQ；**6-bit 接近 FP32**，>5× 压缩，服务内存 −68.7% | 与 DAMP 同类但更激进（6-bit vs 9.9-bit）。**在 64GB HBM 的 Ascend 上，这比任何 kernel 优化都值钱** |
| **Exact Record Omission in Delta Attention** | [2609.06872](https://arxiv.org/abs/2609.06872) | **证明"delta 相减"这种捷径不成立**，replay 才是正确解法，并给出 replay certificate | ⭐ **理论护栏**：它解释了为什么不能"存 delta 再减回去"（这正是 §1.1 里"更新几乎不可逆"的严格版本）。**做方案时用它来排除错误设计** |
| **STree** | [2505.14969](https://arxiv.org/abs/2505.14969) | 对 **Mamba-2** 的标量衰减做"转移矩阵树组合"，把门的乘积变成 cumsum | **TreeWY 的直接前身**。局限：只对 Mamba-2 成立，**GDN 的非交换矩阵转移不适用**（TreeWY 原文明确指出） |
| **Snakes and Ladders** | NeurIPS ENLSP-W 2024 | 早期 SSM 投机解码；两种变体 | ReplaySSM 的 Appendix A 实测：**比 vLLM 快照慢 1.4–2.2×**；"*Their recovery kernels are serial recurrences that hold the …*" |
| **The Mamba in the Llama** | [2408.15237](https://arxiv.org/abs/2408.15237) | 把恢复融合进验证 kernel，但**需要为每种接受结果捕获一个 CUDA Graph** | ReplaySSM 明确批评这种做法不可扩展 |
| **ReaDy** | [2610.20503](https://arxiv.org/abs/2610.20503)（2026-10-14） | ⭐ **把 HLA 的 recurrent state 当作 drafter 的"动态上下文"**，每个 draft 位置用 learned query 检索 | **正交方向**：别人用 state 做 **draft**，我们用它做 **verify**。可组合——尤其 Ascend 上 state 已在手边 |
| **Component-Aware Self-Speculative** | [2605.01106](https://arxiv.org/abs/2605.01106) | 把 **SSM/线性注意力子图**当作零成本内部 drafter（Falcon-H1 并行型、Qwen3.5 串行型） | 与 MTP 竞争/互补：不增加 draft 模型，但需要 target 架构支持；对 Ascend 上的 Qwen3.5 系列有直接价值 |
| **Tandem** | [2610.26660](https://arxiv.org/abs/2610.26660)（2026-10-19） | **手机 NPU** 上服务 HLA GUI agent：checkpoint 策略 + 回滚状态开销 | ⭐ **唯一的 NPU 侧论文**，且明确警告"回滚状态会把 prompt 处理挤出矩阵单元" ⇒ 对 Ascend 的设计有直接警示价值 |
| **Windowed-MTP** | [2607.21535](https://arxiv.org/abs/2607.21535) | 百万上下文下 **MTP draft head 的 full-attention over KV** 成为主导成本；用窗口化去掉 draft-KV 税 | 说明"MTP 本身也要省"，与我们的 state 优化是两条正交的省钱路 |
| **H-Spec** | [2609.24197](https://arxiv.org/abs/2609.24197) | 无 drafter 侧 KV cache 的并行投机解码 | drafter 侧省内存，与 target 侧 state 正交 |
| **Mamba Drafters** | [2506.01206](https://arxiv.org/abs/2506.01206) | 用 Mamba 作 drafter（线性结构，避免二次复杂度） | drafter 侧；与 MTP 互补 |

### 7.2 SketchSSM：一条**不同的**技术路线（重要！）

| | |
|---|---|
| 论文/PR | vLLM [#61124](https://github.com/vllm-project/vllm/pull/61124)（open）；Ascend 后端 vllm-ascend [#19769](https://github.com/vllm-project/vllm-ascend/pull/19769)（open） |
| 思想 | **不存完整 state，存它的低秩 sketch**（`mean rank ≈ 5.8`，预算 8；46.5% 的 head 是 rank-1）+ 一个短窗口 |
| 精度 | GSM8K：dense **82.18% ± 0.40** vs SketchSSM **82.16% ± 0.87**（差 −0.03pp） |
| 性能 | 910B3 / Nemotron-3-Nano-4B / B=16：**2.54×** decode；B=128：**5.10×**（注：dense 侧 kernel 未调优） |
| 与我们的关系 | **它不是"减份数"，而是"把每份压成低秩"** ⇒ 与因子日志**可叠加**，且**已经在 Ascend 上有 NPU kernel**（896+553 行 Triton-Ascend） |

> ⭐ **这是最值得跟进的一条"旁路"**：如果 state 本身能压到 mean rank 5.8，那 "n+1 份" 的成本也会同步下降 5–20×。**建议把 SketchSSM 列为期 3 的备选，而不是忽略。**

### 7.3 已被证伪/不建议的方向

| 方向 | 证据 |
|---|---|
| **纯 decode-kernel 重放** | SpecLA 实测被 chain-decomposed 打败 **1.80–7.11×**；ReplaySSM 实测比快照快不了几个点 |
| **纯 prefill-kernel 复用短窗口** | SpecLA：窗口太短，setup 摊不掉；且因果依赖模型与树不匹配 |
| **并行树 kernel（无 chain 分解）** | ⚠️ **SpecLA 的反直觉负结果：并行树 kernel 在 16–32 节点时常比串行还慢（0.52–1.02×）**。TreeWY 也印证（树"not yet a throughput win"）。**这是对"树一定更好"的最有力的反驳。** |
| **每接受结果一个 CUDA Graph**（Mamba in the Llama） | 组合爆炸，ReplaySSM 明确批评 |

---

## 8. 结论与建议

### 8.1 一句话结论

> **在 Ascend 上实现 GDN/MTP 的 n+1 状态问题，正解是"1 份 FP32 checkpoint + 一段紧凑因子日志 + commit 时重建"，第一版严格保持因果（链），用 ACL Graph 双图（append/flush）落地；树留到第二期并用 Neumann 级数替代三角求解；最后用混合精度（DAMP 式）压 checkpoint 本身。**

### 8.2 为什么这个方案"合适"

1. **有生产验证**：GLM-5.3-Flash、Kimi-K3 已在用；TensorRT-LLM / FlashInfer / vLLM(KDA) / SGLang(GDN) 四家引擎都已合入。
2. **Ascend 已有半成品**：KDA RecoverSSM 在 vllm-ascend main；GDN 的 #18765 已实测（**3.188 MiB vs 12 MiB**）；RFC #19987 已经把路线写清楚。
3. **对 Ascend 硬件友好**：全程 matmul + elementwise，**不需要三角求解、不需要 scatter/gather、不需要动态 shape**。
4. **可增量验证**：链 → 变长 → 树 → 精度，每一期都能独立出数据。
5. **风险可知**：所有已知坑（UB 记账、off-by-one、图捕获、prefix caching）都有现成的上游 issue 可参考。

### 8.3 立即可做的 5 件事

1. **去 vllm-ascend RFC [#19987](https://github.com/vllm-project/vllm-ascend/issues/19987) 下认领子任务**（2026-10-27 刚开，正缺人手，和你的目标完全重合）。同时确认 [#18765](https://github.com/vllm-project/vllm-ascend/pull/18765) 的状态（含 `vllm_ascend/ops/gdn_recover.py`、`ops/triton/mamba/gdn_recover{,_batch}.py`）能否直接复用。
2. **通读 SGLang 的 `gdn_replayssm_spec_decode.py`**（4 个 kernel，Apache-2.0，从 vLLM fork `3c85112` 移植），是目前最完整的 GDN 因子日志实现，可直接对照移植。
3. **确认你不需要从零写三角求逆**：FLA 已有 `fla/ops/utils/triton_ascend/solve_tril.py`，`vllm_ascend/ops/triton/fla/` 有完整 Triton-Ascend 版，`fla_npu` 有 AscendC 版（`solve_tri`、`recompute_w_u_fwd`、`prepare_wy_repr_*`、`chunk_local_cumsum`、`chunk_fwd_h/o`、`recurrent_gated_delta_rule`）。**你要改的是 mask 构造与 scan 顺序，不是写逆矩阵。**
4. **写 microbench 验证 "Neumann + 定形状树掩码" 能否被 ACL Graph 捕获**——期 3 的 Go/No-Go 判据，成本低、决定性高。
5. **读 Fast & Stable Triangular Inversion（[2605.21325](https://arxiv.org/abs/2605.21325)）**：它直接在 **NPU** 上比较了直接求逆 vs 迭代法，能帮你省掉一轮试错。
6. **复现 #18765 的 3.188 MiB vs 12 MiB 口径** + **建立 FP32 递推 oracle**（不要依赖端到端对比，BF16 下 greedy 会漂）。

---

## 9. 附录

### 9.1 论文一览表

| 方案 | arXiv | 日期 | 机构 | 树 | 公开代码 | 引擎 |
|---|---|---|---|---|---|---|
| ReplaySSM | [2610.13778](https://arxiv.org/abs/2610.13778) | 2026-10-09 | Princeton + NVIDIA | ❌ | ✅ [Johnny-Liou/ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM) | SGLang ✅（#28451/#28695）/ TRT-LLM ✅ / FlashInfer ✅ / vLLM：Mamba2+KDA ✅、**GDN 走 RecoverSSM 仍 open** |
| TreeWY | [2608.20961](https://arxiv.org/abs/2608.20961) | 2026-08-21 (v2 10-22) | Thomson Reuters | ✅ | ❌ | vLLM 私有 fork（分支 `b073ed6`）+ RFC [#54080](https://github.com/vllm-project/vllm/pull/54080) |
| Bole | [2608.01651](https://arxiv.org/abs/2608.01651) | 2026-08-03 | NJU + Ant Group | ✅ | ❌ | 仅第三方 PR [sglang#46702](https://github.com/sgl-project/sglang/pull/46702)（chain-only） |
| SpecLA | [2607.16673](https://arxiv.org/abs/2607.16673) | 2026-07-18 | NJU + Alibaba | ✅ | ❌ | PyTorch/Triton 原型，未进引擎 |
| DeltaLog | [2608.15533](https://arxiv.org/abs/2608.15533) | 2026-08 | USTC | ❌ | ❌ | vLLM 原型 |
| DAMP | [2608.27513](https://arxiv.org/abs/2608.27513) | 2026-08 | 美团 + ECNU + SCUT | ➖ | ❌ | SGLang |
| DASC | [2608.30386](https://arxiv.org/abs/2608.30386) | 2026-08-31 (v2 10-09) | ECNU + 美团 | ➖ | ❌ | SGLang |

### 9.2 相关基础工作

| 工作 | arXiv | 作用 |
|---|---|---|
| Gated Delta Networks (GDN) | ICLR 2025 (Yang, Kautz, Hatamizadeh) | 本文主角 |
| Parallelizing Linear Transformers with the Delta Rule（UT/WY 变换） | [2406.06484](https://arxiv.org/abs/2406.06484) | **所有方案的数学基础** |
| STree | [2505.14969](https://arxiv.org/abs/2505.14969) | 树版本前身（仅 Mamba-2） |
| Medusa / EAGLE-2 | [2401.10774](https://arxiv.org/abs/2401.10774) / — | 树 draft 的来源 |
| Mamba-2 (SSD) | — | 状态空间对偶，ReplaySSM 的另一目标 |

### 9.3 关键 GitHub 链接（按重要性）

**Ascend 侧（最相关）**
- RFC: <https://github.com/vllm-project/vllm-ascend/issues/19987>
- GDN checkpoint/replay PR: <https://github.com/vllm-project/vllm-ascend/pull/18765>
- MAX_MTP 8→16（UB 记账教训）: <https://github.com/vllm-project/vllm-ascend/pull/11236>
- 变长 GDN 算子: <https://github.com/vllm-project/vllm-ascend/pull/15207>
- Adaptive verification: <https://github.com/vllm-project/vllm-ascend/pull/17763>
- SketchSSM NPU: <https://github.com/vllm-project/vllm-ascend/pull/19769>
- Hybrid prefix cache: <https://github.com/vllm-project/vllm-ascend/issues/9508>

**参考实现**
- SGLang GDN ReplaySSM spec decode: <https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py>
- vLLM ReplaySSM 参考 fork: <https://github.com/Johnny-Liou/ReplaySSM>
- vLLM upstream KDA RecoverSSM（已合）: <https://github.com/vllm-project/vllm/pull/63045>
- vLLM GDN RecoverSSM（open）: <https://github.com/vllm-project/vllm/pull/61309>

### 9.4 证据等级说明

| 等级 | 含义 | 本文中哪些属于此级 |
|---|---|---|
| **A 已核验** | 我直接读取了 arXiv HTML 全文 / GitHub API 元数据 / 原始源码 | 所有 arXiv id、标题、作者、机构；所有 PR 号 + merged/open 状态；SGLang/vLLM/Ascend 的代码路径；#18765 与 #11236 的引文 |
| **B 论文自述** | 论文中声称、我未独立复现 | 所有 benchmark 数字（1.8×、3.4–7.7×、69.1%、1.32–2.22× 等） |
| **C 本报告估算** | 我按公开配置算的 | §1.3 的 state 内存表、§6.3 的 UB 预算示例 |
| **D 未验证** | 明确标注 | §6.3 期 3 的"Neumann + 定形状掩码可被 ACL Graph 捕获"= **待证实假设**；Ascend UB 容量具体数值；各 PR 的实际性能 |

> ⚠️ 需要特别说明：Bole 的 arXiv 版本标着 **"HPCA 2027 Submission #814 — Confidential Draft"**，其数字可能随评审变化；TreeWY 是**单作者 preprint**；SpecLA / DeltaLog / DAMP / DASC 均为 preprint。**引用这些数字做决策前建议先小规模复现。**



### 9.5 术语表

| 术语 | 含义 |
|---|---|
| **GDN** | Gated DeltaNet：带门控的 delta-rule 线性注意力，Qwen3.5/3.6/3.8 系列的线性层 |
| **KDA** | Kimi Delta Attention：Kimi Linear / Kimi-K3 的线性注意力，**逐 key channel 衰减**（`diag(exp(g_t))`），比 GDN 的标量衰减更复杂 |
| **SSM / 线性注意力** | 用固定大小 recurrent state 替代增长 KV cache 的一类层 |
| **recurrent state** | 每层每 head 的 `d_k × d_v` 矩阵（GDN/KDA）或 `d_state × d_inner`（Mamba-2） |
| **recurrent state 快照 / `store_all`** | 每个 draft 位置存一份完整 state，用于回滚；本文要消除的对象 |
| **WY / UT 变换** | 把 delta rule 的 token 级依赖重写成"块内严格下三角线性系统"的变换（DeltaNet 提出） |
| **伪值 `ṽ` / 因子 `(d,k,g,β)`** | 取代完整 state 的紧凑表示，用于重建 |
| **commit / 提交** | 采样后把"被接受路径"的状态写回持久槽位 |
| **rollback / 回滚** | 丢弃被拒绝分支的状态（本文的核心难点） |
| **Neumann 级数解** | `(I+G)^{-1} = Σ_{m=0}^{d} (−G)^m`，因 `G` 是深度有界幂零矩阵；**把三角求解换成 d 次 matmul** |
| **ancestor mask / 祖先掩码** | 树验证中"每个节点只能看到自己的祖先"的掩码；**非因果** ⇒ 破坏图捕获 |
| **DFS 先序** | 让祖先下标小于后代下标的排列方式，保证 `G` 严格下三角 |
| **MRV1 / MRV2** | vLLM ModelRunner V1 / V2；Ascend 侧大量 GDN 工作正在向 MRV2 迁移 |
| **DFlash** | 一种 draft 方案；vllm-ascend 的自适应验证（adaptive verification）以它为入口 |
| **UB** | Unified Buffer，昇腾 AI Core 上的片上缓存（约 192 KiB/AI Core，具体随型号） |
| **Cube / Vector** | 昇腾 AI Core 的两类计算单元：Cube 做 matmul（16×16×16 fractal），Vector 做逐元素/归约 |
| **ACL Graph** | 昇腾的图捕获/回放机制（`torch.npu.graph`），要求静态 shape |
| **`MAX_MTP`** | vllm-ascend `recurrent_gated_delta_rule` 算子里的**编译期** tiling 常量（当前 16），限制一次能处理的最大投机 token 数 |
| **RecoverSSM** | vLLM 上游对 ReplaySSM 这条线的命名（开关 `--use-replayssm`） |
| **`mamba_cache_mode=align`** | 无 RecoverSSM 时 vLLM 撑 GDN 投机解码的方式：block 对齐 + 快照 + 一个 block 的后撤 |
| **SketchSSM** | 另一条路线：不存完整 state，存其低秩 sketch（mean rank ≈5.8）+ 短窗口 |

### 9.6 这份调研的"未做/待办"

- ❌ 未在任何真实 Ascend 硬件上跑过任何数字（本文所有 Ascend 结论都是**基于上游 PR 的二手证据 + 架构推理**）。
- ❌ 未验证"Neumann + 定形状树掩码"能否被 ACL Graph 捕获（§6.7 R2）——这是**最该先做的 microbench**。
- ❌ 未拿到 UB 容量的官方数值（按 192 KiB 估算）。
- ❌ Bole 的 TTFT/TPOT 百分比、TreeWY 的 MiB 级具体数值被 arXiv HTML 的 LaTeX 剥离，未从 PDF 补全。
- ❌ 没有对 seven 篇论文做任何独立复现；所有 benchmark 数字都是**论文自述**。
