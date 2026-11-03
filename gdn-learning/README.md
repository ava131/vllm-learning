# gdn-learning

探究 **GDN / MTP 为什么需要 n+1 份 recurrent state**，以及如何用尽可能小的代价消掉它，
目标是**在昇腾（Ascend NPU）上落地**。

## 文件

| 文件 | 行数 | 内容 |
|---|---|---|
| **`七个算法汇总.md`** | 682 | ⭐ **先看这份**：7 个算法，先讲读写流水账与链/树，再逐项写「论文地址 / 实现思路 / 解决什么问题 / 开源仓库 / 实现情况 / 相关 PR 和 issue / 在 Ascend 上怎么做」 |
| **`GDN-RecoverSSM-实现方案.md`** | 667 | ⭐ **要开工看这份**：ReplaySSM/RecoverSSM 命名关系、ReplaySSM 原理与完整公式、issue #19987 要求与名词表、KDA RecoverSSM 在 vllm-ascend 的现状（版本/文件/算子）、GDN 工作量拆解（可复用 / 需重写 / 人天估算 / 风险）、以 Qwen3.8-27B 为算例的收益与代价量化（带宽 1.6–4.2×、释放 106 GiB、重算开销 0.14%） |
| `GDN-MTP-N+1-state-survey.md` | 1386 | ⭐ **主文档**：七个方案精读 + 统一数学推导（含可直接照着实现的 8 步）+ 引擎落地逐 PR 核验 + Ascend 三期方案与风险 |
| `raw/github_issues_vllm_ascend.md` | 577 | vLLM / vllm-ascend 的 PR/issue 逐条核验（含 `MambaSpec` 的 n+1 代码出处） |
| `raw/sglang_and_repos.md` | 589 | SGLang 线性注意力架构 + 七个方案开源实现核查（含 SGLang ReplaySSM 合入 PR） |
| `raw/alternatives_and_ascend.md` | 734 | 其他最新方案综述（15+ 篇）+ Ascend 硬件/算子约束 |
| `raw/spec_la_dasc.md` | 300 | SpecLA / DASC 的逐字段原文调研（(a)–(g)、引文、未验证项标注） |

## 一句话结论

> **不要存 N+1 份 dense state。存"1 份 checkpoint + 一段紧凑因子日志"，commit 时用一次
> 矩阵乘法重建唯一被接受的状态。** 这个变换对 GDN 是**代数精确**的（WY/UT 变换从 chunk
> 扩展到 verify 窗口 / 树），已进 GLM-5.3-Flash、Kimi-K3 生产环境，并已在
> SGLang / TensorRT-LLM / FlashInfer 合入，vLLM 侧叫 **RecoverSSM**（开关 `--use-replayssm`）。

**n+1 在代码里就是这一行**（vLLM main）：
```python
# vllm/v1/kv_cache_interface.py — MambaSpec
max_memory_usage_bytes = page_size_bytes * (1 + num_speculative_blocks)   # 1 + K
# 打开 RecoverSSM 时 num_speculative_blocks = 0 → × 1
```

## 五个立即行动项

1. 认领 **vllm-ascend RFC [#19987](https://github.com/vllm-project/vllm-ascend/issues/19987)**
   （"Support GDN RecoverSSM for Speculative Decoding on Ascend"，2026-10-27 开）的子任务；
   同时确认 [#18765](https://github.com/vllm-project/vllm-ascend/pull/18765)（GDN checkpoint/replay，
   实测 **3.188 MiB vs 12 MiB**）能否直接复用。
2. 读 **SGLang 的 `gdn_replayssm_spec_decode.py`**（Apache-2.0，4 个 kernel，从 vLLM fork `3c85112` 移植）——
   目前最完整的 GDN 因子日志实现。
3. **不必从零写三角求逆**：FLA 已有 `fla/ops/utils/triton_ascend/solve_tril.py`；`vllm_ascend/ops/triton/fla/`
   有完整 Triton-Ascend 版。要改的是 mask 构造与 scan 顺序。
4. 写 microbench 验证 **"Neumann 级数 + 定形状树掩码"能否被 ACL Graph 捕获** —— 树方案的 Go/No-Go。
5. 读 **Fast & Stable Triangular Inversion（[2605.21325](https://arxiv.org/abs/2605.21325)）**——唯一在 **NPU** 上
   比较"直接求逆 vs 迭代法"的论文，能省掉一轮试错。

## 阅读路径

- 只要结论 → 主文档 §0、§8.2
- 要写方案 → §2（含 §2.6 可直接照着实现的 8 步推导）+ §6（Ascend 三期路线 + 12 条风险）
- 要复现 → §5.3（SGLang 参考实现）+ §6.4（kernel 到 Ascend 的映射）

> ⚠️ 免责：所有 benchmark 数字均为**论文/PR 作者自述**，未独立复现；Ascend 结论基于上游 PR
> 二手证据 + 架构推理，**未在真机验证**。证据等级见主文档 §9.4。
