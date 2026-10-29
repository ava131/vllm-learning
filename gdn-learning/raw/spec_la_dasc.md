# SpecLA & DASC — Deep Read Notes for the "n+1 Recurrent State" Problem (GDN / MTP Speculative Decoding)

**Compiled:** 2026-10-29. **Method:** arXiv HTML full text via `curl`; equations recovered from MathML `alttext` (the plain-text conversion drops inline math, so many numbers below were re-extracted from HTML math nodes). GitHub state queried live. Every item is tagged **[paper]** (authors' claim, quoted where useful) or **[verified]** (checked myself in HTML/API). Unconfirmed items are marked **未能验证 / unverified**.

Both papers attack the same resource from opposite ends. SpecLA manages the **transient** recurrent state inside one speculative step (the `T = 1 + K` state versions the "n+1" problem is named after). DASC manages the **persistent** recurrent-state checkpoint that must live in HBM for prefix reuse. An Ascend port needs both halves: the transient half is bounded by UB, the persistent half by HBM, and they compete for the same tensor layouts.

---

# Paper 1 — SpecLA

## 中文摘要

SpecLA 研究**线性注意力（GDN）目标的投机解码运行时**。核心矛盾：Transformer 靠"截断 KV 后缀"丢弃被拒草稿，而循环状态没有后缀边界——每个候选 token 都原地改写同一份稠密状态，验证 `T=1+K` 个 token 天然需要 `T` 份状态。论文给出三层设计：(1) **拓扑感知验证内核**（层主序状态驻留串行核 / 树掩码并行核 / 链分解混合核）；(2) **accepted-factor 状态管理**——不存稠密快照也不重放 token，而缓存验证本就产生的紧凑因子 `(k,v,α,β)`，接受后做廉价 delta 构造；(3) **延迟状态更新**，把提交融合进下一次验证核。H100 + GDN-1.3B 端到端最高 **1.70×**。**局限**：单模型、单层微基准、无混合注意力、无 MTP、无 paged/连续批处理；草稿器需重训。**无官方代码，vLLM/SGLang/Ascend 均无落地**。

## (a) Metadata

| Field | Value |
|---|---|
| Title | SpecLA: Efficient Speculative Decoding for Linear-Attention Models |
| arXiv | **2607.16673** (v1, `[cs.CL]`, 18 Jul 2026) — **[verified]** via `citation_title`/`citation_date` |
| URL | https://arxiv.org/abs/2607.16673 · https://arxiv.org/html/2607.16673 |
| Authors | Zhibin Wang, Xuying Han, Zhaohua Yang, Fuliang Liu, Xue Li, Rong Gu, Sheng Zhong, Chen Tian |
| Affiliations | State Key Lab of Novel Software Technology, Nanjing University; Xue Li at Alibaba Group |
| Venue | Preprint only — no venue line in HTML **[verified]** |
| Project page | None. No author repo link; only citation links (`NVlabs/GatedDeltaNet`, `fla-org/flash-linear-attention`) plus arXiv's own LaTeXML links **[verified]** |

## (b) Problem and relation to "n+1 recurrent state"

Linear-attention decode is **still memory-bound**, just on a different object:

> "the recurrent state is 2 MiB, and prior work characterizes batch-1 GDN decode as having arithmetic intensity below 1 FLOP/B." **[paper]**

Arithmetic the paper prints (GDN-1.3B, FP32 state) **[paper]**:

- Persistent state **2 MiB/layer**; per token a **4 MiB read–write round trip**.
- H100 SM SRAM = **228 KiB** → one layer's state cannot fit one SM; V-dimension tiling forced.
- **Snapshot (naive):** `L` candidates → `L ×` state = **16 MiB/layer at L = 8**.
- **Factors:** "≈16L KiB per layer, about **128×** smaller than full-state buffering's 2L MiB"; concretely **128.5 KiB/layer at L = 8**.

That 16 MiB/layer × 24 layers **is** the n+1 blow-up — and it is exactly what production vLLM does today (see (f)). The paper names three coupled challenges, the second being the n+1 problem itself:

> "Dense-state acceptance: posterior selection must advance only the accepted recurrent-state updates **without full-state snapshots, token replay, or an extra state-access pass** at every iteration boundary." **[paper]**

## (c) Core technical idea

**Recurrence as used:**

```
S_t = A_t S_{t-1} + β_t k_t v_tᵀ ,   A_t = (I − β_t k_t k_tᵀ) Λ_t ,   o_t = S_tᵀ q_t
Λ_t = α_t I (GLA)  or  diag(exp(g_t)) (channel-wise)
```

Two facts carry the design: the update is a **rank-1 outer product** plus a linear map (so the state tiles along either `d_k` or `d_v`), and each token's state delta splits into **factor generation** (expensive: projections + FFN) and **delta construction** (cheap: rank-1 + elementwise). Only the first is costly to redo.

**(i) State-resident serial kernel.** Reorders execution from token-major to **layer-major**:

> "This schedule removes the layer gap between neighboring token updates, provides an opportunity to keep the recurrent state resident on SRAM across candidate tokens, and therefore avoids the round-trip cost of writing the state back to HBM after each token." **[paper]**

Tiling matters: K-dimension tiling would need a **cross-block reduction per token**, so they tile along **V** and keep full `d_k` in-block.

**(ii) Tree-masked parallel kernel.** Tree dependency inserted into the Delta-rule **UT (WY) transform**:

```
A_tree = M_tree ⊙ (K_b K_aᵀ) ,   T_tree = solve_tril(I − A_tree)
O_GDN  = Q̂ S₀ᵀ − Attn(W S₀ᵀ) + Attn U          (M_tree ∈ {0,1}^{L×L} = ancestor mask)
```

All candidate logits come from the same committed `S₀`, updates propagating only along ancestor paths. Exact parallel endpoint, but "closer to short prefill than to recurrent decode" — the `K,V,β,α` construction, `T` solve and intermediate movement can't amortize over 16–32 tokens.

**(iii) Chain-decomposed hybrid kernel** (the one that wins). **Heavy-light decomposition** into node-disjoint chains covering all candidates; serial kernel within each chain; chains whose **boundary states** are ready run in parallel. Serial across layers, parallel within a layer.

**Accepted-factor state management.** Three options:

| Approach | Speculative storage | Post-selection cost |
|---|---|---|
| Full-state snapshot | `L ×` state (16 MiB/layer @ L=8) | cheap copy |
| Token replay | tokens only | **re-runs FactorGenerate** |
| **Factor buffering** | **128.5 KiB/layer @ L=8** | cheap delta construction |

```
f_t = FactorGenerate(x_t)                    # k_t, v_t, α_t, β_t
S_A = S₀ + Σ_{t=1..A} DeltaConstruct(f_t)
```

The record is layer-type-specific (GLA: `k,v,gate`, no `β`; DeltaNet: `k,v,β`, no decay; Mamba: selective SSM params). Measured **[paper]**: 4-token accepted prefix → replay **29.73 ms** vs factor reuse **8.62 ms** (3.45×); over commit length 1–8, replay 19.70→35.96 ms vs 5.15→13.10 ms (**2.74–4.28×**). Exactness: "recurrent-state max difference is at most [bound] and the convolution-state difference is **zero**."

**Delayed state update.** A standalone commit kernel makes three state passes per iteration; instead the pending accepted-factor buffer is applied **at the top of the next verification kernel**, same tile: 0.102→0.071 ms (len 1), 0.153→0.133 ms (len 8), **1.15–1.44×**.

**Draft side.** Confidence pruning `q(v) = Σ_i log p_draft(t_i | prefix, t_<i)`, keep `q(v) ≥ q* − τ_tree`; monotonicity along root-to-leaf makes the window **ancestor-closed**, so one threshold prunes width and depth jointly. Plus a **target-aligned EAGLE drafter** trained on the target's own recurrent features, deliberately kept Transformer-style.

**Cost vs naive baseline** (Table 1, normalized to root-to-leaf replay) **[paper]**:

| top-k | depth | nodes | R2L (ms) | Serial | Parallel | Hybrid |
|---|---|---|---|---|---|---|
| 2 | 12 | 16 | 22.5 | 1.31 | 0.58 | **2.20** |
| 2 | 24 | 32 | 55.0 | 1.36 | 1.02 | **2.55** |
| 4 | 4 | 16 | 37.8 | 3.73 | 1.70 | **5.14** |
| 4 | 4 | 32 | 91.2 | 4.74 | 3.47 | **7.11** |
| 4 | 12 | 16 | 19.0 | 1.19 | 0.52 | **1.80** |
| 4 | 24 | 32 | 41.0 | 1.30 | 0.95 | **2.19** |

The **parallel kernel is usually slower than serial** (0.52–1.02×) — a load-bearing negative result against the intuition "tree verification ⇒ parallel kernel."

## (d) What it solves and does not solve

**Solves** the transient state problem exactly: output hashes are checked against AR decoding for all routes; factor buffering is exact in FP32; no weight change, no target retraining, no calibration.

**Does not solve:**

- **Drafter training required** — "an EAGLE-style drafter trained on recurrent target features."
- **Single model, single layer** — "The target is a single GDN layer configured to match GDN-1.3B"; end-to-end uses a *pure* GDN-1.3B. **No hybrid (GDN + full-attention) target**, which is what Qwen3-Next/Kimi-Linear actually are.
- **No MTP** — assumes an external EAGLE-style drafter.
- **No serving system** — no paged state, no continuous batching, no request-level state cache; batch size appears only in a microbenchmark.
- **Modest gains:** 1.42× mixed, 1.70× GSM8K, **1.06× HumanEval**; candidly, "HumanEval remains the hardest workload: the tree route accepts only 2.34 tokens on average."
- **Gated by proposal quality:** below the mid-acceptance transition region, speedups "remain below 1 for every reported draft length."
- **Does not reduce persistent state size** — that is DASC's job.
- `τ_tree`, node budget, top-k are workload-tuned.

## (e) Open-source code

**No official implementation. 未能验证 any repo.** Ran `gh search repos "specla"` (only unrelated `SpecLab`, `SpeclatentGS`, `Speclang`), `gh search code`, global `gh search issues "SpecLA"` (unrelated only). Reference repos the paper *does* link (dependencies, not SpecLA): https://github.com/NVlabs/GatedDeltaNet (**691 stars**, push 2026-03-13) and https://github.com/fla-org/flash-linear-attention (**6,000 stars**, push 2026-10-28) **[verified]**. Target checkpoint https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure exists **[verified]**.

## (f) Implementation status in real engines

**No PR or issue anywhere names SpecLA** (vLLM, SGLang, vLLM-Ascend, FlashInfer, TileLang all empty) **[verified]**. But the *concept* is shipping independently in vLLM as **ReplaySSM / RecoverSSM**, which states the n+1 arithmetic outright:

> **vllm #49887**, "[Kernel] ReplaySSM: cache SSM inputs for faster Gated DeltaNet speculative decode" — **CLOSED, not merged** — https://github.com/vllm-project/vllm/pull/49887
> "vLLM's native GDN spec path keeps a separate recurrent state snapshot for every draft token, so a step with `T = 1 + num_speculative_tokens` tokens writes `T` full `head_v x head_k` states per value head. Cost grows linearly in `T`. ReplaySSM replaces the snapshots with one checkpoint plus a small circular buffer of the per-step inputs. For GDN the buffer holds `(d, k, g)`, where `d = beta * (v - S k)` … Caching `d` rather than `v` removes the serial state dependency during replay. … Rollback is a cursor move. … The full state is materialized only on a periodic flush." **[verified, quoted]**

This is *SpecLA §5.1's insight* reached independently, shipped as a kernel cursor rather than a factor buffer. Related, all **[verified]**:

| Item | Title | State |
|---|---|---|
| vllm #48018 | ReplaySSM: … Mamba2 standard decode | **MERGED** 2026-07-24 |
| vllm #62004 | ReplaySSM: … GDN standard decode | open |
| vllm #61309 | [Spec Decode][Qwen3.5] RecoverSSM for Qwen3.5 / Qwen3-Next GDN | open |
| vllm #61566 | [Spec Decode][GLM-5.3-Flash] Support KDA RecoverSSM | open |
| vllm #56466 | [WIP][ReplaySSM] prefix caching for GDN speculative decode | open |
| vllm #55688 / #52962 | ReplaySSM STP/MTP/prefix lifecycle; quantized checkpoints | open |

(#61309 claims "+8.6% output throughput at K=5" on Qwen3.5-4B + native MTP, L40S, c=32 — **PR claim, not independently verified**.)

**Open bugs proving the problem is unsolved in production [all verified]:**

- **vllm #39273** (open) — ngram spec corrupts GDN SSM state on Qwen3.5 hybrid: *"the SSM state was already evolved for all N tokens during the forward pass. There is no mechanism to rollback/revert the GDN state to the position after only the accepted tokens. … this relies on the block-aligned state checkpoints being correct — which they aren't."* https://github.com/vllm-project/vllm/issues/39273
- **vllm #63593** (open) — MTP with forced-reject diverges from target-only on Qwen3.8-27B (GDN hybrid) at K≥5, n≥8, L≥192 — directly contradicts #39273's assumption that the MTP path is clean.
- **vllm #55504** (MERGED) — "Bound GDN speculative decode query widths": the bounded speculative state layout needed a hard width cap.
- **vllm #50146** (closed) — hybrid draft models; Wall 1 is *"Target-side `short_conv` state corruption … no room for rollback. Widens the target's `short_conv` state by `num_spec`."*
- **SGLang #43514** (open) — DFlash + Qwen3.8-27B: accept rate silently collapses to 0.00.
- **FlashInfer #6595** (open) — CAKE linear-attention/SSM tracker; KDA kernels tracked, **nothing on speculative rollback state**.

## (g) Ascend/NPU relevance

**Zero mention of Ascend/NPU; all experiments on H100** **[verified]**. What breaks:

1. **On-chip residency premise.** The serial kernel holds a `d_k × d_v_tile` slice resident across tokens. The Ascend analogue is AI Core UB via explicit `T.copy`/`T.Pipelined` (Triton-Ascend) or AscendC `DataCopy` + `PIPE_V` barriers. There is no free "one block owns a persistent tile across a token loop" — you hand-write the double-buffered pipeline. (UB ≪ 2 MiB/layer is safe; the exact KB is **未能验证** here.)
2. **`solve_tril`** on a 16–32 lower-triangular system has no Ascend primitive; needs a hand-written fixed-size inversion on Cube (fp16/bf16), awkward against the FP32 state requirement.
3. **The MTP axis is already a compile-time constant.** vllm-ascend **#11236** (MERGED, "Increase MAX_MTP from 8 to 16 for `recurrent_gated_delta_rule`") shows the Ascend GDN recurrent op tiles the MTP axis with hard `MAX_MTP`, and review had to fix UB accounting: *"the arch35 kernel carved an extra `gamaInUb` slice from `tmpBuff` that the host-side UB sizing never reserved … With `MAX_MTP` doubled to 16, this unaccounted `betaNumAlign`-float requirement also doubles … → risk of UB overflow / silent corruption."* **[verified, quoted]** Any kernel widening the per-step token axis must extend the host-side UB tiling model in lockstep.
4. **The Ascend GDN spec path is already fragile.** vllm-ascend **#19608** (open): AICore MTE invalid-GM-address crash in the GDN spec-decode path with `qwen3_5_mtp` + `FULL_DECODE_ONLY`, surfacing at `gdn_attn_builder._remove_spec_graph_padding_queries` with placeholder-padded spec batches (`{req: [-1,-1,-1]}`). Parent-pointer trees add more padding/aliasing hazards.
5. **Landing zone exists but is unbuilt.** vllm-ascend **#19845** (open) lists *"Linear attention: KDA and GDN require correct state handling under adaptive verification"* with `[ ] KDA adaptation — PR(s): TBD`, `[ ] GDN adaptation — PR(s): TBD`.
6. **The drafter** is an ordinary Transformer (portable); its *training* (feature collection through GDN state holders) needs GPU infrastructure.

---

# Paper 2 — DASC

## 中文摘要

DASC 研究**混合线性注意力模型的 prefix cache**：循环状态被就地覆盖，prefix 复用必须在可复用边界存"状态检查点"，完整检查点很快耗光 HBM，导致驱逐与重复 prefill。作者发现 GDN 的**头**与 KDA 的**通道**保留视界差异极大，且该视界可**仅从权重**离线推导（把输入相关门控替换为代表值 `a_s`）。据此构造静态掩码，只把长视界单元打进**不规则检查点**，可选地在 TP rank 间均衡放置；复用时对省略单元**零填充**（dasc-nr）或从**有界后缀重放**（dasc-wr）。SGLang 实现，三模型代表性工作点 **1.32–2.22×** 压缩、TTFT 降 **16.1–42.0%**、E2E 降 **4.9–10.2%**。**注意**：压缩的是**持久检查点**，不解决投机解码**瞬时** n+1 状态；dasc-wr 是近似恢复。

## (a) Metadata

| Field | Value |
|---|---|
| Title | **DASC**: Decay-Aware State Compression for Hybrid Linear-Attention Serving (HTML body renders lowercase `dasc`; `citation_title` is `DASC: …`) **[verified]** |
| arXiv | **2608.30386** — v1 `[cs.LG]` 31 Aug 2026; **v2 09 Oct 2026** **[verified]** |
| URL | https://arxiv.org/abs/2608.30386 · https://arxiv.org/html/2608.30386v2 |
| Authors | Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu |
| Affiliations | East China Normal University (Yu, "work done during internships at Meituan"; Liu, corresponding); Meituan; South China University of Technology (Zhang) |
| Venue / code | Preprint; **no project page, no code link** — only the ShareGPT dataset **[verified]** |

## (b) Problem and relation to "n+1 recurrent state"

> "Because each state checkpoint contains large state matrices for all linear-attention layers, frequent state checkpointing quickly exhausts HBM, while sparse state checkpointing increases replay or repeated prefill." **[paper]**

The relation is **indirect but real**. n+1 is a *state-versioning* problem: each speculative step needs either `T` state copies or one checkpoint plus replayable inputs. DASC shrinks **the checkpoint tier** — the same HBM pool that must hold speculative snapshots. Its unit is `(layer, head)` or `(layer, head, channel)`, i.e. **spatial not temporal**, so it does **not** reduce `T`.

Scale **[paper]**: Qwen3-Next-80B-A3B = **36 GDN + 12 full-attention layers** (head-wise decay); Kimi-Linear-48B-A3B = **20 KDA + 7 full-attention** (channel-wise); Kimi-K3 = 93 blocks, 69 KDA layers, **6,624 layer–head pairs**. Checkpoints use SGLang defaults: **FP32 temporal + BF16 convolution** state, `B_temp` dominating.

## (c) Core technical idea

```
S_t = S_{t-1} Diag(α_t) (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ ,   α_t ∈ (0,1)^{d_k}
```

Three operations — attenuate, remove content along `k_t`, write the new association. **GDN decays per head, KDA per channel**, and that granularity sets the compression ceiling.

**Step 1 — weight-only static log-decay.** Replace the input-dependent gate input `a_{t,u}` with one fixed model-level representative `a_s`:

```
g_s^std(u) = −exp(A_log,u) · softplus(a_s + d_u)          # Qwen3-Next GDN, Kimi-Linear KDA
g_s^K3(u)  = ℓ · σ( exp(A_log,u)(a_s + d_u) ),  ℓ = −5    # Kimi-K3 bounded safe gate
α_s = exp(g_s),   H_s = ln(ε)/g_s ,  ε = 1e−3             # retention horizon (tokens)
```

`a_s = −0.3` (Qwen3-Next, Kimi-Linear — deliberately slower decay/longer horizons than `a_s = 0`, biasing toward retention); `a_s = 0.8` (Kimi-K3). **This is the entire calibration: one scalar per model**, not per layer, not per request.

Heterogeneity **[paper]**: median `α = 0.471` (Kimi-Linear) vs `0.969` (Qwen3-Next) vs `0.942` (Kimi-K3). At a 16-token cutoff **389/640 (60.8%)** Kimi-Linear layer–head pairs are *mixed*; at 64 tokens **6,512/6,624 (98.3%)** Kimi-K3 pairs are mixed. Layerwise mean retained fractions 0.179–0.558 and 0.267–0.741 — the imbalance that motivates ragged storage.

**Step 2 — selection and ragged packing.** Global (retained) iff `H_s(u) > W_max`, else local. Mask built once offline, request-independent.

```
C_temp^rag = Σ_l n_l / Σ_l r_l
C_ckpt^rag = (B_temp^dense + B_conv) / (B_temp^rag + B_conv)
C_temp^pad = (Σ_l n_l)/(L·R_max) ,  gain_ragged = R_max / r̄ ,  r̄ = L⁻¹ Σ_l r_l
```

**Step 3 — load policies.** `dasc-nr` (default) zero-fills omitted units: zero extra compute, pure approximation. `dasc-wr` replays a bounded suffix from a **zero-initialized scratch** state:

```
W_u = clamp(next_pow2(H_s(u)), 4, W_max)
W_replay = min(n_p, max_{u ∈ L_local} W_u)
```

The subtlety is stated plainly: *"During scratch replay, all recurrent units, including global units, start from zero. Missing pre-window history therefore affects intermediate activations across layers, not only the local-state initial condition. … dasc-wr is approximate recovery, not an exact reproduction of dense prefix computation."* **[paper]**

**Step 4 — TP-balanced placement.** Capacity is `max_r U_r`-bound. Two independent knobs: **head-bundle permutation** of all coupled tensors at load, and **storage balancing** to slot width `C = ⌈Σ_r U_r / T⌉` via variable-split all-to-all at `Store` / inverse at `Load`. Both are **exact reindexings**; recurrence stays rank-local.

**Step 5 — SGLang integration.** Two-tier cache: native Mamba pool (running states + dense checkpoints share a slot allocator) + a separate fixed compressed pool; radix metadata carries tier/slot/lock; one global LRU across both; KV untouched.

## (d) What it solves and does not solve

**Solves** persistent-checkpoint HBM pressure with **no retraining and no weight change**. Capacity with TP balancing **[paper]**: Kimi-Linear **2.64/7.06/14.72/28.24×** at `W_max = 16/64/128/512`; Qwen3-Next **1.10/1.32/1.60/2.49×**; Kimi-K3 **1.61/2.01/3.48×** at `64/128/512`.

| Model | Config | TTFT Δ | E2E Δ | Quality Δ |
|---|---|---|---|---|
| Kimi-Linear | `W_max=16`, TP-bal Off | **−42.0%** | **−10.2%** | 0.56 pts MMLU-Pro |
| Qwen3-Next | `W_max=64`, TP-bal On | **−25.6%** | **−4.9%** | ≤0.92 pts (6 tasks) |
| Kimi-K3 | `W_max=64`, TP-bal Off | **−16.1%** | **−6.0%** | ≤0.42 pts (5 tasks) |

Composes with quantization: `W_max=16` + INT8 checkpoint + dasc-nr → **8.11×** capacity vs dense mixed-precision, all five Kimi-Linear tasks within ~1 pt of dense **[paper]**.

**Does not solve:**

- **Approximation, not exactness.** `dasc-nr` zero-fills; `dasc-wr` is "approximate recovery." No exactness guarantee anywhere.
- **Selection ≠ importance.** Unusually honest: *"These weight-derived, request-independent profiles characterize storage heterogeneity, **not the importance of omitted units**"*; *"Rank agreement does not establish agreement at a fixed horizon cutoff."* Spearman `H_s` vs empirical horizon = **0.801 / 0.856 / 0.895**; vs state magnitude only **0.632 / 0.533 / 0.443**.
- **Task-dependent risk:** *"recurrent states can be redundant for simple recall yet remain important for multi-key, aggregation, and reasoning tasks."* Their zeroing probe shows ≤1 pt change on RULER NIAH-S1 — but that is retrieval, carried by full-attention KV.
- **Not about speculative decoding** — no transient `T = 1+K` state, no rollback, no kernels.
- **Scope, quoted:** *"Our evaluation focuses on hybrid architectures; applicability to purely linear-attention or state-space models remains to be studied. We have not evaluated serving at tensor-parallel degrees larger than eight."*
- **Engine surgery required**: ragged layout + dual pools + LRU + all-to-all. Not a drop-in.

## (e) Open-source code

**No official implementation. 未能验证 any repo.** Ran `gh search repos "dasc"` (only `daScript`, `dascore`, `DSAC-v2`, `DAScratchPad`, a HKU course), `gh search repos "retention horizon"` / `"state checkpoint compression"` (empty), `gh search issues "decay-aware state"` (only aggregator repos like `xllm-io/daily-papers`). The §4.4 SGLang integration is **not public** as far as I can find.

## (f) Implementation status in real engines

**No PR or issue in SGLang, vLLM, or vLLM-Ascend references DASC** **[verified]**. Related SGLang activity is the *infrastructure* DASC would plug into, and it is all capacity/admission repair rather than compression **[verified]**:

| Item | Title | State |
|---|---|---|
| sglang #43204 | `Can not alloc mamba cache` kills engine when every cached mamba state is locked | open |
| sglang #38846 | Expose effective `max_running_requests` after the mamba/linear-attention state-cache cap | open |
| sglang #37619 | mamba: prevent scheduler crashes when checkpoint slots run out | open |
| sglang #42443 / #37613 | hicache host Mamba capacity; overflow ring for host pool write-backup | open |
| sglang #40907 | Restore non-DCP Mamba checkpoint donation (HiCache) | **merged** |

That gap — nobody compresses the checkpoint — is exactly DASC's contribution, and it is **unlanded**.

## (g) Ascend/NPU relevance

**No Ascend/NPU mention**; hardware is "eight Hopper-architecture GPUs with TP8" **[verified]**. What breaks:

1. **The ragged state tensor** `[slots, Σ_l r_l, d_v, d_k]` with per-layer offsets `o_l`. Ascend GDN state buffers are dense per-layer tensors consumed by `recurrent_gated_delta_rule`; a ragged view needs a new AscendC kernel or a per-layer gather into scratch (negating much of the bandwidth win).
2. **Variable-split all-to-all.** HCCL `AlltoAllV` exists, but DASC assumes it can permute head bundles at load time and keep recurrence rank-local. vllm-ascend **#9508** (open RFC, hybrid prefix cache / align mode) states the blocker directly: *"The number of KV heads of linear-attention is different from that of full attention. Currently, the logic that the same KV head in Ascend Store reuses the state cannot meet the requirements of linear-attention."* **[verified, quoted]** DASC's per-`(head, channel)` units are strictly finer.
3. **`dasc-nr` is the NPU-friendly half** — a memset of omitted units, trivially portable, and it is the default.
4. **`dasc-wr` is the hostile half** — a second, zero-initialized prefill over ≤`W_max` tokens, extra UB pressure, and interaction with `MAX_MTP` tiling if it enters the speculative path.
5. **Landing zone exists**: vllm-ascend #9508 already asks for `kv_cache_group_id` in remote keys, null-block skipping for aligned state groups, and cross-group prefix-hit computation. DASC's compressed pool would be a third tier on it.
6. **Offline mask computation** (one scalar `a_s` per model) runs on CPU — ports for free.

---

# Synthesis for an Ascend GDN/MTP implementation

| Family | Transient cost | Exact? | Where it exists |
|---|---|---|---|
| **Snapshot per candidate** | `T × head_v × head_k` per value head; 16 MiB/layer at T=8 | exact | vLLM native GDN spec path (per #49887) |
| **Replay from one checkpoint** | store `(d,k,g)` or tokens; recompute delta | exact (input replay) / approx (bounded suffix) | vLLM ReplaySSM/RecoverSSM (#48018 merged); SpecLA's replay baseline (29.73 vs 8.62 ms) |
| **Factor buffering** | `≈16L KiB/layer` (~128× less) | exact in FP32 | SpecLA §5.1 (no public code) |

**SpecLA's real contribution is not the factor buffer** — vLLM reached that independently. It is (i) the **layer-major, V-tiled, state-resident serial kernel**, (ii) the **chain-decomposed hybrid schedule**, and (iii) the **negative result that the fully parallel tree-masked UT kernel loses** at 16–32 nodes. Copy (i)+(ii); skip (iii).

**DASC is orthogonal** and matters more for *serving* than speculation: it decides which `(layer, head, channel)` units deserve HBM residency at all. On Ascend, where HBM is scarcer relative to compute than on H100, the 42% TTFT reduction may be worth more than SpecLA's 1.06–1.70× decode speedup.

**Feasibility (my assessment, not the papers'):**
- **SpecLA-style factor buffering + delayed update on Ascend: feasible.** It is a tiling/bookkeeping change on the existing `recurrent_gated_delta_rule` AscendC op, and vllm-ascend already has the `MAX_MTP` tiling hook (#11236) and an open GDN/KDA adaptive-verification task (#19845). Hardest: extending the host-side UB tiling model when the token axis widens (#11236's review shows this is already a live overflow risk), and the `solve_tril` UT transform.
- **SpecLA's tree-masked parallel kernel on Ascend: likely not worth it** — it already loses on H100 at realistic tree sizes, and Ascend adds kernel stages and layout conversions.
- **DASC on Ascend: partially feasible.** `dasc-nr` + static mask + two-tier cache mirrors work already in flight (#9508). Ragged packing and TP-balanced all-to-all are the blockers. Recommended first step: implement the static `H_s` selection (CPU-only, no kernel change) and measure zero-fill quality before committing to ragged layout.

**Verification gaps.** No independent measurement of any SpecLA/DASC number — all figures are authors' or PR authors' claims as quoted. Ascend UB size per AI Core **未能验证** (no hardware spec fetched). Whether vllm-ascend's `gdn_attn_builder` path can carry a parent-pointer tree topology **未能验证** (not stated in #19608 or #19845).
