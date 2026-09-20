# Marconi 论文 + vLLM / vllm-ascend 支持情况调研 — 对话原文记录

> 记录日期：2026-09-20
> 说明：以下为本次对话的完整原文，未做删改。

---

## 用户

https://arxiv.org/pdf/2411.19379
读一下这个文章，他要解决的问题是什么？怎么解决的？有什么效果？

---

## 助手

已读完这篇论文（PDF 全文 + arXiv HTML/ar5iv 版本都取了，PDF 我下载后用 pypdf 抽取了正文，所以 5.4 微基准和结论都覆盖到了）。

**论文**：*Marconi: Prefix Caching for the Era of Hybrid LLMs*，Princeton + AWS，MLSys 2025（[arXiv:2411.19379](https://arxiv.org/abs/2411.19379)，[代码](https://github.com/ruipeterpan/marconi)）。

---

## 1. 要解决的问题

Hybrid LLM（Attention 层 + SSM/线性循环层，如 Mamba、Jamba、Zamba，常见配比 1 个 Attention 配 6–10 个 SSM 层）是长上下文服务的主流方向：SSM 计算 O(L)、状态内存 O(1)，比 Transformer 的 O(L²)/O(L) 省得多。但它把**前缀缓存（prefix caching）**这个跨请求复用的大杀器废掉了：

- **SSM 状态是 in-place 更新的**，一个序列结束时的状态**无法回滚**成它的某个前缀 → 前缀复用变成"全有或全无"，只能精确匹配。
- 要支持任意前缀的复用，就必须**细粒度 checkpoint**（每 x 个 token 存一份 SSM 状态）；而 SSM 状态大小固定、却是**单 token KV 的 10–100 倍**。
- 结果：每条序列产生一大堆**又大又几乎不会被命中**的缓存条目 → 缓存被撑爆、频繁 thrashing。

论文给出的量化证据很直观：

- block size 32 时，KV block 的复用率是 **25.0%**，而 SSM 状态只有 **0.4%**（差 65.3 倍）；即使用 block size 128，也只有 3.3%。
- 7B 模型、单条 10K token 序列，缓存条目要 **17.4 GB**，是同规模 Transformer 的 **3.3 倍**。

一句话：Hybrid 模型单请求更省，但让跨请求的复用优化失效了；现有系统（vLLM、SGLang）都只针对 Transformer 的 KV 设计，**没有系统支持 Hybrid 的前缀缓存**。

## 2. 怎么解决的（Marconi 的两个核心策略）

**统一用一棵 radix tree 同时管理 KV 和 SSM state**——因为前缀复用被"复用机会最少的层"卡住，两套状态必须绑定在同一个节点上，不能分开缓存。

**(1) 审慎准入（Cache Admission）**：只收"高复用概率"的 SSM 状态。作者分析真实 trace 后把前缀复用分成两类：

- *Purely input*（system prompt、few-shot、长文档 QA 等）：会被很多请求共享。做法是 prefill 前先做一次 **speculative insertion**，看插入后是否会产生新的中间节点（分支点），会才在 prefill 时 checkpoint。
- *Input + output*（多轮对话、agent 轨迹）：对话是往后追加而不是分叉，所以**只 checkpoint 最后一个解码 token 的状态**。

因此每条序列最多准入约 2 个 SSM 状态（KV 仍然全量缓存，和现有系统一致）。取状态的实现：支持 chunked state passing 的模型（Mamba2 等）直接物化倒数第二个 chunk 的状态，精度损失在一个 chunk 内；不支持的（Mamba1、Jamba）用 **two-pass prefill** 拿精确前缀状态。

**(2) FLOP-aware 驱逐（Cache Eviction）**：关键观察是 KV 的**大小与它能省下的计算量成正比**，而 SSM 状态大小固定、与序列长度无关——所以传统按 size 的 cost-aware 算法（如 GDSF）在 Hybrid 上失效。于是提出新指标：

> flop_efficiency = 所有层被省下的总 FLOPs / 全部 state 的内存占用
> 节点效用 S(n) = recency(n) + α · flop_efficiency(n)

α=0 就退化成 LRU；α 由启动期 bootstrap + 并行 grid search 在几秒内自动调优。实现上还有两个细节：驱逐候选包含"只有 ≤1 个孩子的中间节点"（不只是叶子）；命中时只更新被命中节点的 timestamp（祖先节点的 KV 会被子节点吸收，传统做法会高估其热度）。

## 3. 效果

评估设置：7B Hybrid（{4,24,28} Attention/SSM/MLP），TTFT 部分用 Jamba-1.5-Mini（12B active/52B total），8×A100-40GB，FP16；workload 为 LMSys、ShareGPT、SWE-Agent/SWE-Bench。

| 对比 | 结果 |
|---|---|
| Token 命中率 vs vLLM+（细粒度 checkpoint） | 平均 **4.5× / 7.3× / 34.4×**（LMSys / ShareGPT / SWEBench） |
| P95 TTFT vs 无前缀缓存 | 降低最多 **36.9% / 73.2% / 46.8%**（281.4 / 106.3 / **617.0 ms**） |
| P95 TTFT vs vLLM+ | 再多降 **36.1% / 71.1% / 46.8%**（275.4 / 103.3 / 617.0 ms） |
| P95 TTFT vs SGLang+（LRU 驱逐） | 再多降 **17.2% / 12.8% / 24.7%** |
| FLOP-aware 驱逐 vs LRU | token 命中率提升 **19.0%–219.7%**；SWEBench 上整体命中率 16.4% → **32.7%**，FLOP saved **+90.3%** |

**代价是有意做的权衡**：Marconi 牺牲短序列的命中率（<7K token 最多 -3.0%）换取长序列命中率（>7K 最多 +25.5%）；P5 TTFT 比 SGLang+ 差 6.3%，但绝对值只差 **2.1 ms**，而 P50/P95 TTFT 改善 13.4%/22.0%。这个取舍在 Hybrid 上划算，因为短序列的 prefill 本来就快。

**微基准的适用边界**（也解释了为什么它"赶上好时代"）：

- 缓存竞争中等时收益最大（60→140 GB 下提升 10%–68.3%），高竞争时容量本身就不够，低竞争时驱逐策略无所谓。
- SSM 层比例越高越赚：Attention:SSM 从 1:2 到 1:8，相对 vLLM+ 的提升从 13.5% 升到 2.6×；**纯 Transformer 上三者性能相同**（说明它是 Hybrid 专属优化，不损害原有场景）。
- SSM state 维度从 16 增到 128（Mamba1→Mamba2 趋势），相对 vLLM+ 的提升从 5.7× 升到 35.4×。

## 4. 局限 / 值得注意的点

- 它不是新的数据结构，而是**缓存准入与驱逐策略的重设计**，建立在已有 radix tree/block 管理之上；因此和 vLLM 的 KV block 管理是互补的——更准确地说，baseline 里的 "vLLM+" 就是把 vLLM 的 block 缓存机制搬到 Hybrid 上，而 Marconi 换掉了它的 admission/eviction 逻辑。
- *Purely input* 前缀要到**第 3 次出现**才开始享受复用（第 2 次用于识别），且每序列只留约 2 个 SSM 状态，覆盖率有损。
- 取精确前缀状态需要 two-pass prefill（有额外开销）或依赖 chunked state passing（有一格 chunk 的误差）。
- 评估指标是 token 命中率/TTFT/FLOP，不做下游精度对比（前缀复用是精确复用，不改输出）——这是合理的，但也意味着没有验证缓存管理对模型质量的影响。
- 方法可推广到 RWKV、RetNet、GLA、xLSTM、DeltaNet 等其他循环/线性层架构（状态更新规则不同，但"大状态 + in-place 更新"的性质相同）。

如果你想，我可以在当前仓库里整理一份带图表要点和公式的中文笔记（比如放到 `vllm-learning/` 下作为一节），或者顺着它和 vLLM APC/SGLang RadixAttention 的差异再深挖一层。

---

## 用户

你帮我在社区找一下vllm/vllm-ascend对这个论文的支持情况，特别是最新版的。先给我个结论：成熟/不成熟。有没有pr能够作为证据。我们还有没有开发的空间。然后讲一下支持到什么情况了

---

## 助手

已查完。数据截至 **2026-09-20**（今天），证据都是 GitHub API/PR 原文，不是二手转述。

## 一、结论（先给判断）

| 仓库 | 结论 | 一句话理由 |
|---|---|---|
| **vllm（上游）** | 🟡 **部分成熟：论文的"准入"那一半已落地且进主线；"驱逐"那一半没有实现** | Marconi-style admission 已合入（#37898，2026-06-10），后续还在加固；FLOP-aware eviction 在仓库里搜不到任何实现 |
| **vllm-ascend** | 🔴 **不成熟** | 只落地了基础 hybrid/Mamba align 模式前缀缓存（#7103，2026-03），**Marconi 准入策略在 Ascend 上实际不生效**，修复 PR #14468 至今 open；PD/MTP/MRV2 组合问题一堆 |

一句话：**上游"抄了一半"（准入），Ascend 连这一半都还没跑通。**

---

## 二、关键 PR 证据

### 上游 vLLM

| PR | 状态 | 内容 |
|---|---|---|
| [#30877](https://github.com/vllm-project/vllm/pull/30877) | ✅ merged 2026-01-23 | `--mamba-cache-mode align` 的块对齐 Mamba 前缀缓存（地基，非 Marconi） |
| [**#37898**](https://github.com/vllm-project/vllm/pull/37898) | ✅ **merged 2026-06-10** | **[Hybrid] Marconi-style admission policy for hybrid cache** —— 论文的直接落地：检测共享前缀（FA KV 命中但 SSM 落后 → 判定为 shared prefix junction），在块对齐位置物化并缓存 Mamba 状态。作者实测开 APC 后比 main 快 40%（Qwen3.5-0.8B）/ 66%（Qwen3.5-35B-A3B） |
| [#47782](https://github.com/vllm-project/vllm/pull/47782) | ✅ merged 2026-07-13 | njhill：把 Marconi 的 shared-prefix 缓存扩展到 retention-interval 稀疏缓存（Mamba + SWA），使 `retention interval = 0` 也能吃到隐式 system prompt 缓存 |
| [#52216](https://github.com/vllm-project/vllm/pull/52216) | ✅ merged 2026-08-17 | `prefix_cache_retention_interval` 升级为正式参数，默认 0 |
| [#52789](https://github.com/vllm-project/vllm/pull/52789) | ✅ merged 2026-08-22 | Mamba prefix caching 的内部 prefill checkpoint，TTFT 改善 9%~25% |
| [#26201](https://github.com/vllm-project/vllm/issues/26201) | 🔵 open（tracking issue） | 明确把 "Marconi-style caching of shared system prompt in align mode (#37898)" 勾成 **已完成**；但 "relax mamba block size 约束(#33194)"、"fuse SSM state 写入 kernel"、"TP>1 测试"、"**把 align 模式设为 hybrid 默认**" 仍未勾 |
| [#50551](https://github.com/vllm-project/vllm/pull/50551) | 🟠 **open** | Marconi 的"**缓存最后一个 decode token 状态**"（多轮对话复用）那一半，只在 sparse retention 下补齐，尚未合 |
| [#55697](https://github.com/vllm-project/vllm/issues/55697) + [#55873](https://github.com/vllm-project/vllm/pull/55873)/[#55875](https://github.com/vllm-project/vllm/pull/55875)/[#55876](https://github.com/vllm-project/vllm/pull/55876) | 🟠 open（2026-09） | 另一条路线：应用显式声明 checkpoint 边界（`<|mamba_checkpoint|>`）+ 调度器 Producer/Consumer 协调，声称 L40S 上吞吐 +110%、kernel 7.58x |

### vllm-ascend

| PR/Issue | 状态 | 内容 |
|---|---|---|
| [#7103](https://github.com/vllm-project/vllm-ascend/pull/7103) | ✅ merged 2026-03-15 | Ascend 上第一个 hybrid 前缀缓存（Qwen3.5/Next + `--mamba-cache-mode align`），跟随上游 #30877。**PR 自己列了三个坑**：① align+PD 分离不支持；② 混合 KV 导致 block_size 被抬到 2048（TP2 跑 Qwen3.5-35B-A3B），**短于 2048 的前缀永不缓存**；③ 自己 patch 了 triton kernel 绕开 Ascend 上的 bug |
| [**#14468**](https://github.com/vllm-project/vllm-ascend/pull/14468) | 🟠 **open（2026-08-18，base 是 releases/v0.25.1rc，只有 bot 评论）** | **[Bugfix] Enable Marconi-style admission policy for hybrid prefix caching** —— **最关键证据**：`AscendHybridKVCacheCoordinator.find_longest_cache_hit()` 没有设置 `num_uncached_common_prefix_tokens`，导致调度器的判据恒为 0，Marconi 准入永不触发。作者 910C/A3 实测：修复前多组 hit rate **0.0%**，修复后 34.3%~69.2%（8192/5120/4：TTFT 982.9→737.9ms；8192/7168/4：979.2→406.4ms） |
| [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517) | 🔵 open RFC | Ascend 侧 hybrid 前缀缓存提升的 tracking。7-28 的评论里有人问"ascend 是否已有 Marconi admission"，回答是"上游合了，应该能直接用"——但 #14468 证明**并不能直接用** |
| [#12176](https://github.com/vllm-project/vllm-ascend/pull/12176) | ❌ draft 已关闭未合（2026-09-16） | hybrid partial prefix cache hits，"要等 0.25.0 之后的版本"，最后因冲突关闭 |
| [#16451](https://github.com/vllm-project/vllm-ascend/issues/16451) | 🔵 open RFC 2026-09-13 | GDN+FA 的 **bit-consistent 部分命中**，定位了 3 个 cold/warm 数值漂移根因（FA 命中元数据越界、scheduler split、fused kernel 漂移），还有 2 个未修 |
| [#11711](https://github.com/vllm-project/vllm-ascend/issues/11711) | 🔵 open | Qwen3.5-35B 在 910B 上**开了前缀缓存反而比 Qwen3-30B 慢近 2 倍**（命中率都很高）——align 模式性能问题 |
| [#16639](https://github.com/vllm-project/vllm-ascend/issues/16639) / [#15745](https://github.com/vllm-project/vllm-ascend/pull/15745) | 🔵 open | 指标 `vllm:prefix_cache_hits_total` 恒为 0（缓存明明在工作）；BalanceScheduler 不报指标 |
| 已合入的周边 | ✅ | #9533（AscendStore 的 Hybrid & Mamba align 前缀缓存）、#10009（KV consumer 部分组缓存）、#9514/#7372（310P）、#7796/#7814（layerwise connector 的 Mamba prefill 前缀缓存）、#15763/#15674（AscendStore 释放修复）、#16562（MRV2 block-table 容量）、#15406（310P MRv2） |

**版本现状**：上游最新正式版 **v0.29.0（2026-09-09）**，Marconi 准入从 v0.25.x 起就在里面；Ascend 最新正式版 **v0.23.0（2026-08-16）**、最新 RC **v0.26.0rc1（2026-09-03）**。也就是说 Ascend 的正式版比上游落后约 6 个版本，而且承载 Marconi 准入的 v0.25.1rc 分支上，修复 PR 还是 open 状态。

---

## 三、到底支持到什么程度

**已经支持的（上游 vLLM ≥ v0.25.1，最新 v0.29.0）**
- `--enable-prefix-caching --mamba-cache-mode align`：Mamba/GDN/ShortConv/LinearAttention 的块对齐前缀缓存，覆盖 **Marconi 的 shared-prefix 准入**（#37898）和 retention-interval 稀疏保留（#47782/#52216）。
- 支持 decode 侧 checkpoint（`mamba_cache_mode=all` 的块边界状态物化）、MRv2、spec decode/MTP 的部分组合，且有专门的内部 prefill checkpoint 优化（#52789）。
- 关键差异：**vLLM 的实现和论文不一样**——论文用 radix tree 做投机插入识别共享前缀，vLLM 用"FA KV 命中但 SSM 缓存落后"来反推共享前缀（PR 里明确写了 "differs from Marconi (that uses radix tree)"），并只缓存到块对齐位置以避免改 kernel。所以这是 **Marconi-style，不是 Marconi**。

**没有支持的**
- **FLOP-aware eviction（论文的第二大贡献）完全缺席**：我按 `marconi` / `flop efficiency` / `eviction hybrid` 在 vllm 仓库检索，没有任何实现该效用函数（`S(n)=recency+α·flop_efficiency`）的 PR。最接近的只有 [#55403](https://github.com/vllm-project/vllm/pull/55403)（block priorities，让稀疏缓存能捡漏）和 [#37003](https://github.com/vllm-project/vllm/issues/37003)（context-aware retention API）。也就是说：**论文里"牺牲短序列换长序列"的那套收益目前 vLLM 拿不到。**
- Marconi 的 "last decoded state"（多轮对话）只在 v0.25.x 的 sparse retention 路径下有；通用化仍在 #50551（open）。
- align 模式默认关闭；TP>1、block size 约束放宽等仍在 tracking issue 未完成项里。

**Ascend 侧（最新 v0.23.0 / v0.26.0rc1）**
- 有的：基础 hybrid/Mamba align 前缀缓存、AscendStore 持久化、部分组缓存、310P/MRv2 适配。
- 没有/不确定的：**Marconi 准入策略在 Ascend 上不生效**（#14468 的代码分析 + 实测 0% 命中）。v0.23.0 的 release notes 只写了 "Added hybrid/Mamba attention prefix caching" 和 "Mamba prefix-cache corruption 修复"，没有提 Marconi 准入。v0.26.0rc1 是否已通过 main2main 修好，**我无法从公开 PR 确认**——需要在你机器上实际验证（见下）。

---

## 四、还有没有开发空间：有，而且不小

按"性价比"排序，每条都有 open 的 PR/issue 作依据：

1. **把 Marconi 准入真正在 Ascend 上跑通**（最直接）
   - #14468 是一个小改成（coordinator 里补 `num_uncached_common_prefix_tokens`），作者实测命中率 0%→34~69%。但它 8-18 提的，到现在只有 bot 评论、base 还是 `releases/v0.25.1rc`。
   - 可做：rebase 到 main / 0.26.0rc1、补单测+E2E、推动 review；或先本地验证主线是否已修。
2. **实现 FLOP-aware eviction（论文空缺的一半）**
   - 上游零实现，Ascend 也没有。可以把 `flop_efficiency` 接到 vLLM 的 block 优先级 / Ascend 的 KV pool 与 AscendStore 分级缓存上（长序列优先保留 SSM checkpoint），这正好补上 `#55403`、`#37003` 想做但没做完的事。
3. **partial hit 细粒度化 + 数值一致性**
   - Ascend RFC #16451 列了 3 个 cold/warm 漂移根因，1 个已修、2 个待修；#16449/#16528 在修 PD+MTP 下的细粒度命中；`--prefix-match-unit` 相关文档/粒度问题也在 open。
4. **block_size 膨胀 → 短前缀永不命中**
   - #7103 注释里自己说 TP2 下 block_size=2048；对应上游 #53749、Ascend #12589/#16270、以及 Ascend #15601（`hash_block_size` 断言过严反而挡住缓存）、#16044（two-level hybrid cache + Jenga）。
5. **PD 分离 / MTP / spec decode / MRV2 的组合正确性**
   - 上游 9-17 还有新开的 bug（#57267 非确定性、#57266 `--mamba-block-size` 组合崩、#53912 MTP 输出损坏、#43587 多模态增量请求）；Ascend #16449/#16528/#16258/#15326/#14693/#8829 全 open。
6. **观测性**
   - Ascend #16639（hits_total 恒 0）、#15745（BalanceScheduler 指标）；上游 #52527（报告因缺 checkpoint 丢掉的 shared-prefix token 数）。这块改动小、对定位"到底有没有生效"极有用，适合入门。
7. **存储/远端保留**
   - 上游 #46896（GDN all-mode + Mooncake 保留）、Ascend #12781（Mamba 部分命中的 Mooncake 处理）、#13300、#16044、#16044 的 AscendStore 路线。

---

## 五、给你一个 5 分钟自测（确认你手上的版本到底支不支持）

```bash
# 1) 看你的 vllm-ascend 里这个字段有没有被赋值（没有 = Marconi 准入不生效）
python - <<'PY'
import inspect, vllm_ascend
from vllm_ascend.core import kv_cache_coordinator as m   # 路径按你的版本调整
src = inspect.getsource(m)
print("num_uncached_common_prefix_tokens" in src)
PY
grep -rn "num_uncached_common_prefix_tokens" $(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))") | head

# 2) 跑一次，直接看命中率（Qwen3.5-35B-A3B, tp2）
vllm serve <model> --enable-prefix-caching --mamba-cache-mode align \
  --additional-config '{"recompute_scheduler_enable":true}'
# 同一个长 system prompt 发 3 次以上（Marconi 的 shared prefix 从第 3 次开始命中）
# 观察 vllm:prefix_cache_hits_total / 日志里的 prefix cache hit rate
```

如果第 1 步 grep 不到、且第 2 步共享前缀在第三次请求仍不命中，就说明你的 Ascend 版本属于 #14468 描述的状态——**这就是可以立刻切入的开发点**。

需要的话，我可以接着做两件事之一：① 把 #14468 的代码路径和当前 main 对照，判断主线到底修没修，输出一份可提交的补丁/rebase 方案；② 出一份"在 vLLM/Ascend 上实现 FLOP-aware eviction"的设计草案（挂到 block 优先级或 AscendStore 分级缓存）。你想先走哪条？
