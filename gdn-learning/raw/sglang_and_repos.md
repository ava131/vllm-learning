# SGLang GDN/linear-attention archaeology + implementation status of 7 recurrent-state papers

Research date: **2026-10-29**. All file paths / PR numbers / issue numbers below were read live from
GitHub (`gh api` / `raw.githubusercontent.com`) at that date. SGLang `main` HEAD was
`pushed_at 2026-10-29T07:04:02Z`.

Repo baselines used throughout:

| repo | stars | license | last push |
|---|---|---|---|
| `sgl-project/sglang` | 38,334 | Apache-2.0 | 2026-10-29 |
| `vllm-project/vllm` | 95,820 | Apache-2.0 | 2026-10-29 |
| `fla-org/flash-linear-attention` | 6,000 | MIT | 2026-10-28 |
| `flashinfer-ai/flashinfer` | 6,812 | Apache-2.0 | 2026-10-29 |
| `tile-ai/tilelang` | 8,842 | NOASSERTION (see §4.3) | 2026-10-29 |
| `NVIDIA/TensorRT-LLM` | — | Apache-2.0 | (PR #14203 merged) |

**Methodological caveat (important):** GitHub *code search* is unreliable for these engines.
`gh api search/code -f q='RadixAttention+repo:sgl-project/sglang'` returns `total_count: 0` —
the main `sglang` repo is not in the code-search index (forks of it *are*). So every code claim
below comes from fetching `raw.githubusercontent.com` blobs or the git tree API, not from code search.
`gh search issues` / `gh search prs` / `gh search repos` do work normally.

---

## 1. SGLang GDN / linear-attention architecture

### 1.1 Layering

SGLang models hybrid attention as **two backends wrapped in one**, selected per layer index:

```
HybridLinearAttnBackend            python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:1197
  ├── full_attn_backend            (FlashAttention / FlashInfer / trtllm MLA / DSA ...)
  ├── linear_attn_backend          (MambaAttnBackendBase subclass)
  │     ├── Mamba2AttnBackend                     hybrid_linear_attn_backend.py:1080
  │     ├── GDNAttnBackend                        linear/gdn_backend.py:540
  │     ├── KDAAttnBackend                        linear/kda_backend.py:472
  │     ├── ShortConvHybridAttnBackend            hybrid_linear_attn_backend.py:1730
  │     └── AscendMambaAttnBackendBase            hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py
  └── full_attn_layers: list[int]  → dispatch by layer id
```

`MambaAttnBackendBase` (`hybrid_linear_attn_backend.py:62`) is the shared base: it resolves the
recurrent-state part of the memory pool once (`router.part_view(STATE_PART)`), owns CUDA-graph
capture/replay of the state metadata, and — the key method for our question —
`update_mamba_state_after_mtp_verify()` (`hybrid_linear_attn_backend.py:1527`), which is the
post-acceptance commit hook.

Full linear-attention file list:

```
python/sglang/srt/layers/attention/linear/
  __init__.py
  gdn_backend.py            (1777 lines)  GDNAttnBackend + GDNKernelDispatcher
  kda_backend.py            (1913 lines)  KDAAttnBackend + KDAKernelDispatcher
  lightning_backend.py
  short_conv_backend.py
  inkling_sconv_backend.py
  linear_metadata.py        ForwardMetadata-lite for the linear part
  utils.py                  pp_spec_stable_rows_enabled() etc.
  gdn_prefill_graph.py, kda_prefill_graph.py, kda_flashinfer_prefill_graph.py
  kernels/
    kernel_backend.py       LinearAttnKernelBase (ABC)  ← the "LinearAttnKernelApi" analogue
    gdn_triton.py  gdn_cutedsl.py  gdn_flashinfer.py
    kda_triton.py  kda_cutedsl.py  kda_flashinfer.py  kda_flashinfer_prefill.py
    kda_flashkda.py  kda_nvidia.py  kda_ptx.py  kda_helion.py
```

### 1.2 The kernel abstraction ("`LinearAttnKernelApi`")

There is **no symbol named `LinearAttnKernelApi`** in SGLang. The equivalent is
`LinearAttnKernelBase` in `python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py`,
an ABC with `decode()`, `extend()`, `target_verify()` and capability flags. Verbatim:

```python
class LinearAttnKernelBase(ABC):
    uses_state_checkpoints: bool = False
    supports_fused_chain_verify: bool = False
    # Opt in only when target-verify kernels honor non-unit token strides.
    supports_strided_target_verify_qkv: bool = False
    # True when extend() honors the fp32 track snapshot (track_state /
    # track_chunk_idx), natively or by routing tracked batches to a kernel
    # that does.
    supports_track_state_snapshot: bool = False
```

`decode()` takes `ssm_states, cache_indices, query_start_loc`; `extend()` takes
`ssm_states, cache_indices, query_start_loc`; `target_verify()` additionally takes
`intermediate_states_buffer, intermediate_state_indices, cache_steps, retrieve_parent_token`.
That `intermediate_states_buffer` + `cache_steps` pair **is the n+1 mechanism**.

Mode-level dispatch is in `GDNKernelDispatcher` (`gdn_backend.py:276`), constructed with
`(decode_backend, prefill_backend, verify_backend)`. `self.tree_verify_kernel = TritonGDNKernel()`
is pinned: **tree verify is always Triton**, and the code comment at `gdn_backend.py:532` says
"Tree drafts use Triton even when linear MTP verification uses FlashInfer."

`--linear-attn-backend` / `--linear-attn-decode-backend` / `--linear-attn-prefill-backend` /
`--linear-attn-verify-backend`, choices
(`python/sglang/srt/arg_groups/choices.py:194`):

```python
LINEAR_ATTN_KERNEL_BACKEND_CHOICES = [
    "triton", "cutedsl", "flashinfer", "flashkda",
    "nvidia_kda", "ptx_kda", "helion", "intel_xpu",
]
```

(`--linear-attn-verify-backend` additionally allows `"nv_cutedsl"`; `helion` is KDA-only — GDN raises
"The Helion linear-attention backend supports KDA only, not GDN.")

### 1.3 Which kernels, concretely

| path | implementation | source |
|---|---|---|
| GDN decode/prefill | **Triton, vendored from FLA** | `sglang/kernels/ops/attention/fla/chunk.py::chunk_gated_delta_rule`, `fla/fused_recurrent.py::fused_recurrent_gated_delta_rule_packed_decode` |
| GDN verify (chain + tree) | Triton `fused_sigmoid_gating_delta_rule_update` | `sglang/kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py` (765 lines) |
| GDN decode/prefill/verify | **FlashInfer** (CUDA ≥ SM90) | `flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`, `gated_delta_rule_mtp`, `gdn_kernels.gdn_decode_bf16_state.gated_delta_rule_mtp`, `flashinfer.gdn_prefill.chunk_gated_delta_rule` — loaded in `linear/kernels/gdn_flashinfer.py:136-143` |
| GDN decode/prefill | **CuTe DSL** (`gdn_cutedsl.py`) and Blackwell-specific `gdn_blackwell/{kernel_h,kernel_kkt_inv_uw,kernel_o}.py` | generated CUDA-DSL kernels |
| KDA | Triton, CuTe DSL, FlashInfer, `flashkda`, `nvidia_kda` (nv_cutedsl), `ptx_kda`, Helion, AMD `gfx950` | `linear/kernels/kda_*` |
| causal conv1d | vendored CUDA (`kernels/aot/csrc/mamba/causal_conv1d.cu`) + Triton (`kernels/ops/mamba/causal_conv1d_triton.py`) | no dependency on the `causal-conv1d` PyPI package |
| Mamba-2 SSD | vendored Triton (`kernels/ops/mamba/triton_ops/ssd_*.py`) | from `state-spaces/mamba` |

SGLang does **not** depend on the `fla` PyPI package. `python/pyproject.toml` pins
`flashinfer_python[cu13]==0.7.2` and `tilelang==0.1.12`; FLA code is vendored under
`python/sglang/kernels/ops/attention/fla/` (66 files: `chunk.py`, `chunk_delta_h.py`, `chunk_fwd.py`,
`chunk_intra.py`, `chunk_intra_token_parallel.py`, `chunk_o.py`, `cumsum.py`, `fused_norm_gate.py`,
`fused_recurrent.py`, `index.py`, `kda.py`, `l2norm.py`, `layernorm_gated.py`, `op.py`, `utils.py`,
`wy_fast.py`). Note `chunk_intra_token_parallel.py` — a token-parallel intra-chunk kernel, which is
the SGLang-side enabler for parallel verify.

### 1.4 State pool / cache layout

* `HybridReqToTokenPool` (`srt/mem_cache/memory_pool.py`) holds the slot table; the recurrent state
  lives in a *separate* pool part reached via `token_to_kv_pool.layer_router.part_view(STATE_PART)`
  (`srt/mem_cache/kv/router.py`, `STATE_PART`). `mamba_cache_indices` (formerly
  `req_pool_indices`-derived) is the per-request slot index tensor used by every kernel as
  `cache_indices`.
* `MambaPool` allocates `conv_state` (list, one per conv width) and `temporal` (the SSM/GDN state,
  `[num_layers, size+1, HV, V, K]`). `size + 1` slots — the "+1" is the pad/dummy slot
  (`PAD_SLOT_ID`).
* `--mamba-ssm-dtype` controls the state dtype (fp32 default on CUDA, bf16 option); see
  `expected_state_dtype = torch.float32` / `torch.bfloat16` in the FlashInfer prefill guard
  (`gdn_backend.py:~236`).
* Radix/prefix caching of recurrent state uses "mamba track" buffers:
  `track_mamba_states_if_needed`, `track_mamba_states_all_layers`,
  `fused_commit_track_indices` in `python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py`,
  gated by `--mamba-radix-cache-strategy {no_buffer,extra_buffer,auto}` and
  `--mamba-track-interval`.
* Model registration: `python/sglang/srt/configs/linear_attn_model_registry.py` —
  `register_linear_attn_model(LinearAttnModelSpec(config_class, backend_class_name, arch_names,
  uses_mamba_radix_cache, support_mamba_cache, support_mamba_cache_extra_buffer,
  hybrid_backend_class_name, config_predicate))`.

---

## 2. Speculative decoding on hybrid/linear-attention models

### 2.1 Verdict up front

**Yes — EAGLE / MTP / DFLASH / DSPARK speculative decoding works with GDN and KDA in SGLang, and
tree verify (topk > 1) is supported.** But it is supported through **three different mechanisms
with very different memory behaviour**, and the default one *is* the naive n+1 snapshot scheme.

### 2.2 Tier 0 — the default: per-draft-position full-state snapshots (the n+1 problem, verbatim in code)

`python/sglang/srt/mem_cache/memory_pool.py` (~line 880):

```python
if speculative_num_draft_tokens is not None:
    # Cache intermediate SSM states per draft token during target verify
    # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, V, K]
    ...
    intermediate_ssm_state_cache = torch.zeros(
        size=(num_mamba_layers, spec_state_size + 1, speculative_num_draft_tokens,
              temporal_state_shape[0], temporal_state_shape[1]), ...)
```

Comment in the same block, verbatim:

> "ReplaySSM spec-verify owns rollback via the ring + cursors (the verify kernel never writes
> per-draft snapshots; the commit never reads them), so this buffer -- the dominant spec scratch,
> **~46x the conv state** -- is dead weight there and is skipped."

So SGLang's own estimate is that this buffer is ~46× the conv state, and a separate comment puts it
at **~9 GB** for "K3 dspark γ=7" (Kimi-K3, DSPARK, 7 draft tokens).

How the snapshots are written (`fused_sigmoid_gating_recurrent.py`): the kernel
`fused_sigmoid_gating_delta_rule_update` runs with `disable_state_update=True` and writes each
step's `[HV, V, K]` state into `intermediate_states_buffer + cache_idx * cache_steps * HV * K * V`
(line 313/497). It **also accepts `retrieve_parent_token`** (line 164, preloaded at 282-292), so a
tree is walked by parent pointer over those snapped states — a serial per-node recurrence, not a
batched tree kernel. That is exactly the design TreeWY / Bole / SpecLA attack, and it is why
SGLang's tree path is memory-hungry.

Commit: `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`
(`hybrid_linear_attn_backend.py:1527`) → `scatter_mamba_states_after_mtp_verify(...)`
(`kernels/ops/mamba/mamba_state_scatter_triton.py:781`), which gathers
`src[:, src_indices, steps]` for the accepted step per request. Conv state additionally needs an
accept-rollback: `fused_conv_window_scatter_with_mask` (line 487) / `fused_conv_window_scatter_multi`
(line 709).

### 2.3 Tier 1 — ReplaySSM (landed, opt-in, chain-only)

**This is the big finding: ReplaySSM is fully implemented and merged in SGLang, shipped as two CLI
flags.** RFC `sgl-project/sglang#28511` (opened 2026-06-17, **closed 2026-07-21**), authored by
`yuan-luo`.

`python/sglang/srt/arg_groups/fields/exec_.py:575-596`, verbatim:

```python
    enable_linear_replayssm: A[
        bool,
        "Enable the ReplaySSM buffered output-only linear-attn decode kernel. "
        "Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x "
        "at batch >= 64). KDA uses its selected Triton or Helion implementation, "
        "but its per-K gate ring is larger and ReplaySSM is typically slower "
        "than packed KDA decode; benchmark before enabling it. Requires the "
        "Triton linear-attn decode backend, or Helion for KDA, and "
        "--mamba-radix-cache-strategy no_buffer (the default).",
    ] = False
    linear_replayssm_cache_len: A[int, "Ring-buffer length L for ReplaySSM linear-attn decode. ..."] = 16
    enable_linear_replayssm_spec: A[
        bool,
        "Enable the ReplaySSM spec-verify: fold-every-commit -- a per-slot raw-input window replaces "
        "the recurrent verify's per-draft full-state snapshots. GDN or KDA hybrid linear-attn models, "
        "linear-chain (--speculative-eagle-topk in {None, 1}) only.",
    ] = False
```

Files (all on `main` now):

| file | role |
|---|---|
| `python/sglang/kernels/ops/attention/linear/fused_recurrent_linear_replayssm.py` (628 L) | ReplaySSM buffered output-only **decode** kernel (Part A) |
| `python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py` (1371 L) | ReplaySSM **spec-verify** kernel, GDN (Part B) |
| `python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_fold.py` (270 L) | fold-every-commit replay kernel |
| `python/sglang/kernels/ops/attention/linear/kda_replayssm_spec_decode.py` | KDA variant (`commit_kda_replayssm_after_verify`) |
| `python/sglang/kernels/ops/attention/linear/fused_kda_conv_recurrent_verify.py` (631 L) | fused `causal_conv1d_update(chain, SAVE_INTERMEDIATE)` + recurrence, with `cache_ring=True` |
| `benchmark/bench_linear_attention/bench_gdn_decode.py` etc. | benches |

The GDN verify kernel's module docstring is the single most useful paragraph in the whole repo for
this topic — verbatim excerpt:

> "Ported from the vLLM ReplaySSM reference implementation (github.com/Johnny-Liou/ReplaySSM,
> commit `3c85112`, `vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py`) and the Dao
> AI Lab ReplaySSM blog (dao-lab.ai/blog/2026/replayssm). Part B of SGLang RFC #28511. […]
> The verify output for the whole draft window is reconstructed *output-only* (never materialising
> the state) via the chunked delta-rule `(I + A)^{-1}` UT-transform, and the full state is flushed
> back only every `L` committed tokens. Rejected drafts roll back by a pointer move on the cursors --
> **no state write-back**. […] **Linear-chain only**: the intra-window interaction uses a strictly
> lower causal mask, so this kernel is valid only for linear draft chains
> (`speculative_eagle_topk <= 1`, i.e. MTP / frozen-KV-MTP). **Tree verify (topk > 1), KDA, and
> NPU/CPU fall back to the recurrent verify kernel.**"

Ring buffer shapes (`memory_pool.py:539-549`):

```
replayssm_d: [num_layers, num_slots, HV, L, V]
replayssm_k: [num_layers, num_slots, H,  L, K]
replayssm_g: [num_layers, num_slots, HV, L]   (fp32)
replayssm_rawv / replayssm_rawk / replayssm_beta   (exact-replay / compact-materialization parts)
```

Invariants enforced at pool construction (`memory_pool.py:~830`): `L` must be a power of two, and
`L >= 2 * speculative_num_draft_tokens` ("spec-verify ring too small: {L} < 2 * {n} (early-flush
margin)"). Under `--enable-linear-replayssm-spec`, `intermediate_ssm_state_cache = None` — the n+1
buffer is *not allocated at all*; the code even asserts loudly if the recurrent fallback were ever
reached:

```python
assert intermediate_state_cache is not None, (
    "recurrent target_verify fallback requires intermediate_ssm, "
    "which is not allocated under --enable-linear-replayssm-spec")
```

Production maturity evidence (registered tests in `main`):
`test/registered/attention/unittests/gdn/test_gdn_replayssm_spec_fold.py`,
`test_gdn_cutedsl_ring_verify.py`, `test_linear_replayssm_decode.py`;
`test/registered/kernels/ops/attention/test_kda_replayssm_{fold,fold_batched,ring_fused,ring_ragged}.py`,
`test_kda_mtp_cutedsl_replayssm_ring.py`, `test_fused_kda_conv_recurrent_verify.py`,
`test_fused_verify_triton_gdn.py`; plus `test/manual/test_kda_target_verify.py`,
`test/manual/test_kda_spec_integration.py`.

Commit path: `commit_mamba_states_after_verify` in `python/sglang/srt/speculative/spec_utils.py:1164`
branches on `mamba_pool.replayssm_spec_fold` / `replayssm_cache_base` and advances block-keyed
cursors instead of scattering a snapshot. Comment verbatim:
> "the CONV state still needs its usual accept-rollback, so we keep the conv-window scatter and skip
> only the SSM scatter. GDN-only + linear-chain (topk<=1)."

### 2.4 Tier 2 — Bole (open PR, third-party, chain-only)

`sgl-project/sglang#46702` — **"[Feat] linear-attn: Bole spec-verify for GDN (drop per-draft SSM
snapshots)"**, author `waiwai-dev`, **open**, created 2026-10-27, `+918/-7` across 13 files,
labels `documentation, jit-kernel, memory-pool`, head branch `waiwai-dev:gdn-bole-spec-verify`.
Not merged, not in `main` (there is no `bole_spec_verify.py` in the tree).

New files it adds: `kernels/ops/attention/linear/bole_spec_verify.py` (Triton kernels
`bole_factor_kernel`, `bole_solve_kernel`, `bole_commit_kernel`) and
`srt/layers/attention/linear/bole_spec.py` (`BoleSpecVerifier`), plus flags
`--enable-linear-bole-spec`, `--linear-bole-workspace-mb` (default 384). It states the same identity
TreeWY uses:

> S_a = p_a·S_0 + Σ_{j≤a} (p_a/p_j)·k_j·u_jᵀ

Notably its own PR body says it does **not** deliver the paper's tree verification:

> "Not supported yet (rejected at startup or asserted): **EAGLE tree verify**, PD disaggregation, PP
> stable rows, unified memory pool, KDA models."
> "Validation: linear draft chain only (`--speculative-eagle-topk` in {None, 1}), triton verify
> backend, fp32/bf16 SSM dtype, not with ReplaySSM spec-verify, …"

So as of today SGLang has **no merged tree-capable low-memory recurrent verify**; tree verify goes
through the Triton recurrent kernel with per-node full-state snapshots.

### 2.5 Issue / PR table (SGLang, hybrid × spec-decode)

| # | title | type | state | date |
|---|---|---|---|---|
| [28511](https://github.com/sgl-project/sglang/issues/28511) | [RFC] Porting ReplaySSM to SGLang: faster decode and speculative decoding for hybrid (GDN/KDA) models | RFC | **closed** (implemented) | 2026-06-17 → 2026-07-21 |
| [28451](https://github.com/sgl-project/sglang/pull/28451) | [GDN][KDA] ReplaySSM buffered output-only decode for Linear Attention (Part A) | PR | **merged** `+1771/-14` | merged 2026-06-26 |
| [28695](https://github.com/sgl-project/sglang/pull/28695) | [GDN] Support ReplaySSM Ring Spec-Verify (Part B) | PR | **merged** `+1577/-48` | merged 2026-07-20 |
| [46702](https://github.com/sgl-project/sglang/pull/46702) | [Feat] linear-attn: Bole spec-verify for GDN (drop per-draft SSM snapshots) | PR | **open** `+918/-7` | 2026-10-27 |
| [27658](https://github.com/sgl-project/sglang/pull/27658) | Feat: add compact linear spec cache replay to reduce intermediate cache memory | PR | **open** `+1152/-77` | — |
| [28010](https://github.com/sgl-project/sglang/pull/28010) | Add DFLASH Mamba/GDN reduced-cache replay | PR | closed, **not merged** `+1205/-122` | — |
| [26520](https://github.com/sgl-project/sglang/pull/26520) | [GDN] Add MTP cache mode for final-state recompute | PR | closed, **not merged** `+409/-44` | — |
| [28730](https://github.com/sgl-project/sglang/issues/28730) | [Feature] Reduce Mamba/GDN verify cache memory overhead in DFLASH speculative decoding | issue | closed | 2026-06-19 → 2026-08-19 |
| [25587](https://github.com/sgl-project/sglang/issues/25587) | [Bug][NPU] Hybrid-GDN MTP speculative decoding **is not lossless** on Ascend NPU | bug | closed | 2026-05-18 → 2026-08-30 |
| [38485](https://github.com/sgl-project/sglang/issues/38485) | [Bug] ReplaySSM **CPU backup/restore** loses pending accepted updates | bug | **open** | 2026-09-08 |
| [37834](https://github.com/sgl-project/sglang/issues/37834) | [Bug] `--enable-linear-replayssm` forces `no_buffer`, degrading mamba prefix caching, TTFT up to 4.7× | bug | **open** | 2026-09-03 |
| [38398](https://github.com/sgl-project/sglang/issues/38398) | [Bug] `bench_gdn_replayssm_decode.py` never advances `write_pos`, overstates speedup by ≤21% | bug | closed | 2026-09-01 |
| [38975](https://github.com/sgl-project/sglang/issues/38975) | [Bug] GDN packed decode / KDA replay decode round `sigmoid(beta)` through bf16 → recurrent-state drift | bug | **open** | 2026-09-11 |
| [29857](https://github.com/sgl-project/sglang/issues/29857) | [Bug] EAGLE/MTP on: KV pool profiler leaves ~50 GB idle on hybrid GDN (Qwen3.6-27B NVFP4) | bug | **open** | 2026-07-01 |
| [34786](https://github.com/sgl-project/sglang/issues/34786) | [Bug] `set_mamba_track_indices_from_reqs` TypeError during NEXTN TARGET_VERIFY, hybrid-mamba + spec + lazy buffer | bug | closed | — |
| [42648](https://github.com/sgl-project/sglang/issues/42648) | [Bug] PP + EAGLE/MTP crashes with `--enable-mixed-chunk` | bug | **open** | 2026-10-05 |

### 2.6 Host-side "state backup/restore" hacks — what actually exists

The user asked specifically about memcpy-the-state-before-drafting hacks. Findings:

1. **The general "snapshot" approach is the Tier-0 scheme above** — `intermediate_ssm` is a
   device-side `[layers, slots+1, draft_tokens, HV, V, K]` tensor, written by the verify kernel and
   scattered back on commit. It is not a host memcpy, but it is exactly the "n+1 copies of the state
   per layer per request" cost model.
2. **A genuine host-side backup/restore exists for request retraction (KV-pressure eviction)**, and
   it is buggy: `sgl-project/sglang#38485`, verbatim:
   > "CPU retraction loses accepted GDN ReplaySSM updates when those updates are still in the
   > request's replay ring. **Backup copies the checkpoint and convolution state; reallocating the
   > request clears the replay cursor before restore.** […] Four of six regression cases fail,
   > including both backup routes and wrapped rings."
   Reproduction test lives in a user fork:
   `https://raw.githubusercontent.com/1sgtpepper/sglang/8062a14b.../test/registered/unit/managers/test_retraction_replayssm_state.py`.
3. **On Ascend NPU the rollback is a conv-state shift, and it is documented as incorrect** by the
   issue author. `hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py` imports
   `conv_state_rollback, move_intermediate_cache` from
   `sgl_kernel_npu.mamba.mamba_state_update_triton` and calls
   `conv_state_rollback(conv_states, dst_indices_tensor, last_steps, draft_token_num)` after
   `move_intermediate_cache(...)` gathers the accepted step's SSM state. Issue #25587's root-cause
   section, verbatim:
   > "The current code calls `conv_state_rollback` to right-shift `conv_states` by `(N - k)`
   > positions, intending to undo the effect of rejected tokens. This is **mathematically
   > incorrect** […] SiLU is non-linear and non-invertible […] the correct K-1 history after
   > accepting `k` tokens is the window state that existed after step `k` and before step `k+1` —
   > this intermediate state is **not preserved** anywhere."
   The issue is *closed*, but I did **not** verify the fix in the NPU code path (the file I read
   still performs the shift-style rollback). Treat "NPU GDN MTP is lossless" as **unverified**.

No symbols named `state_backup`, `backup_state`, `restore_state`, `mamba_state_buffer` exist in the
SGLang tree; the closest names are the ones above.

### 2.7 Bole / TreeWY / ReplaySSM / DeltaLog / DAMP / SpecLA referenced *inside* SGLang?

| name | found in sglang? | evidence |
|---|---|---|
| ReplaySSM | **yes, extensively** | RFC #28511, PRs #28451/#28695, 4 kernel files, 2 CLI flags, class attrs `replayssm_*` |
| Bole | **only as an open PR** (#46702) + its new files; not in `main` | `gh search prs "bole" --repo sgl-project/sglang` → 1 hit |
| TreeWY | **no** | `gh search issues "TreeWY"` → no sglang hit; PR search hits are unrelated forks |
| DeltaLog | **no** | issue search → only unrelated `deltalog` DB hits |
| DAMP | **no** (but adjacent feature exists) | `--enable-int8-mamba-checkpoint` / `--int8-mamba-ckpt-size` store radix-cached linear-attn states in int8; that is *uniform* int8 checkpointing, **not** decay-aware mixed precision. Do not conflate. |
| SpecLA | **no** | issue search → no sglang hit |
| DASC | **no** | issue search → no sglang hit |

---

## 3. Implementation status of the 7 papers

Legend: ✅ official public code · 🟡 third-party / partial · ❌ nothing public found

| # | paper | arXiv | official code? | third-party / engine status |
|---|---|---|---|---|
| 1 | **ReplaySSM** — "Cache SSM Inputs, Not State" (Liou, Nover, Naik, Skliar, Shovkun, Shahaf, Dao) | 2610.13778 (2026-10-09) | ✅ **https://github.com/Johnny-Liou/ReplaySSM** — 63★, 7 forks, Apache-2.0, created 2026-05-31, last push **2026-10-13**. It is a full **vLLM fork** carrying the patches (`vllm/model_executor/layers/fla/ops/...`). | **Strongest ecosystem uptake of the 7.** vLLM RFC [#47572](https://github.com/vllm-project/vllm/issues/47572) (open) + draft PR [#47576](https://github.com/vllm-project/vllm/pull/47576) (`+8551/-76`, 35 files, **closed, not merged**, last update 2026-10-12); a second RFC carrying the *same title*, #49232, is also open; feature request #46187 predates both. Known follow-up bugs: vLLM #63175 ("Triton ReplaySSM (V1) drops un-flushed ring tokens when a running request skips a step", open) and #60346 (`append_replayssm_ring` asserts for non-divisible Mamba group counts, closed). **TensorRT-LLM PR [#14203](https://github.com/NVIDIA/TensorRT-LLM/pull/14203) "[None][feat] Checkpointing variant of replay for MTP for mamba models" by `hnover-nv` is MERGED** (2026-08-04, `+6790/-1968`, 20 files). **SGLang merged both parts** (#28451, #28695). |
| 2 | **TreeWY** — "Speculative Verification for Gated DeltaNet Hybrids" (Ghantasala) | 2608.20961 (2026-08-21) | ❌ no standalone repo. Author's public work-in-progress: **fork branch** `sneha5gsm/vllm:treewy-gdn-spec-decode` (head `b073ed6` @ 2026-08-08), linked from the RFC. Fork = `sneha5gsm/vllm` (0★, Apache-2.0, fork of vllm). | vLLM RFC [#54080](https://github.com/vllm-project/vllm/issues/54080) **open**, labels `RFC, speculative-decoding`, author `sneha5gsm`, CC `@Johnny-Liou`. Body is candid about the limit: *"Draft trees are not a throughput win yet: acceptance rises monotonically with width (from 3.23 to 3.44 on Spec-Bench) but throughput falls at every width. The cause is CUDA graph capture: a branching verify step cannot be fully captured today."* Also a fork PR `MaCoredroid/vllm#3` "[Core] Tree-capable recurrent spec-state interfaces (no behavior change)" (open, 1964 files). |
| 3 | **Bole** — "Efficient Tree Speculation for Hybrid-Attention LMs" | 2608.01651 (2026-08-03) | ❌ **no public repo.** The "integration into SGLang" claim is a **private/in-paper implementation, not an upstream merge**. Verbatim from the paper: *"We integrate Bole into SGLang … with approximately 6.2 kLoC of Python and Triton code."*; *"We implement the complete design in SGLang [53], covering tree construction, target-model execution, sampling, and state commit."* The only repo URL in the paper is `https://github.com/sgl-project/sglang/tree/release/v0.5.12` — i.e. their **base version**, not a patch. | A **third-party reimplementation** is in flight: SGLang PR [#46702](https://github.com/sgl-project/sglang/pull/46702) (open, 2026-10-27, `waiwai-dev`). It is *chain-only, GDN-only, no tree* — it does **not** reproduce the paper's headline tree-verification result. **Verdict: the paper's SGLang integration claim is unverifiable from public artifacts; the only public code is a partial, single-author, unmerged PR.** |
| 4 | **DeltaLog** — "Deferred Materialization of Recurrent States" (Lin, Sun, Sun) | 2608.15533 (2026-08-16) | ❌ no repo, no GitHub link anywhere in the HTML. Paper says "we implement DeltaLog for GDN, KDA, and RWKV6 and integrate it into a **prototype serving stack**" — no stack named, no link. | Nothing. No mention in sglang/vllm issue trackers. ❌ |
| 5 | **DAMP** — "Decay-Aware Mixed-Precision Recurrent-State Quantization" (Zhang, Tan, Sun, Yu, Su, Jiang, Xie, Cai — Baidu) | 2608.27513 (2026-08-27) | ❌ no repo, no GitHub/HF link in the HTML (only a ShareGPT dataset URL). Paper says it "claims SGLang" only as the serving engine used for eval. | Nothing public. Adjacent-but-different shipped feature: SGLang `--enable-int8-mamba-checkpoint` (uniform int8 radix checkpoints). ❌ |
| 6 | **SpecLA** — "Efficient Speculative Decoding for Linear-Attention Models" (Wang, Han, Yang, Liu, Li, Gu, Zhong, Tian) | 2607.16673 (2026-07-18) | ❌ no repo, no links in HTML. | Nothing. `gh search repos "SpecLA …"` returns only unrelated name collisions (`SpecLab`, `SpeclatentGS`, …). ❌ |
| 7 | **DASC** — "Decay-Aware State Compression for Hybrid Linear-Attention Serving" (Yu, Sun, Tan, Zhang, Xie, Cai, Liu) | 2608.30386 | ❌ no repo, **zero** outbound links in the HTML. Same Baidu author cluster as DAMP. | Nothing. ❌ |

**Exact searches performed for the "nothing found" rows** (all run 2026-10-29):
`gh search repos "<name>"`, `gh search repos "<name> <keywords>"`, `gh api search/code -f q='<name>'`
(global, e.g. `ReplaySSM` → 22,400 hits but all derivative; `enable_linear_replayssm` → API error/0),
`gh search issues "<name>" --limit 8` (global), `gh search issues "<name>" --repo vllm-project/vllm`,
`--repo sgl-project/sglang`, `--repo vllm-project/vllm-ascend`, and grep of each paper's arXiv HTML
for `github.com` / `huggingface.co` / `zenodo.org`. Two later papers (DAMP, DASC) share authors and
venue style with no artifact release; treat their "we integrate into SGLang" phrasing as
**evaluation-environment**, not code contribution.

**Other third-party ReplaySSM work found** (for completeness, low stars — do not over-weight):
`ROCm/ATOM` has `atom/model_ops/fla_ops/replayssm.py` (AMD's engine, ported ReplaySSM);
`localai-org/vllm.cpp` has `.agents/specs/gdn-replayssm.md`; `zz774ever/fp8-replayssm-gdn-kernel`
(0★, MIT, 2026-09-14); `Dao-AILab/dao-ailab.github.io` publishes the ReplaySSM blog post at
`dao-lab.ai/blog/2026/replayssm` (`_news/2026-10-12-replayssm.md`).

---

## 4. Kernel libraries: GDN chunked kernels, and multi-state / tree support

### 4.1 FLA — `fla-org/flash-linear-attention` (MIT, 6,000★, 2026-10-28)

**GDN chunked entry point** — `fla/ops/gated_delta_rule/chunk.py:397`:

```python
def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                           output_final_state=False, use_qk_l2norm_in_kernel=False,
                           use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                           state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
                           chunk_indices=None, cp_context=None, **kwargs)
```
Shapes (docstring, verbatim):
* `q,k` `[B, T, H, K]`; `v` `[B, T, HV, V]` (GVA if `HV > H`, must divide); `g` `[B, T, HV]` in log space
  (or raw logits when `use_gate_in_kernel=True`, fused with `A_log`,`dt_bias`); `beta` `[B, T, HV]`.
* `initial_state`: **`[N, HV, K, V]` for `N` input sequences** — with varlen, `N = len(cu_seqlens)-1`,
  which need not equal `B`. `output_final_state` returns the same `[N, HV, K, V]`.
* `cu_seqlens` `[N+1]`, FlashAttention convention; `chunk_indices` precomputed for varlen;
  `state_v_first=True` switches the state to `[V, K]` layout (SGLang's layout).
* `chunk_size: int = 64` on `chunk_gated_delta_rule_fwd`.

**Step / single-token entry point** — `fla/ops/gated_delta_rule/fused_recurrent.py:309`:
`fused_recurrent_gated_delta_rule(...)`; inner kernel has `IS_VARLEN = cu_seqlens is not None` and
`N = B if cu_seqlens is None else len(cu_seqlens) - 1`, `h0` is `[N, HV, K, V]`.

**Answer on multi-state / tree:** FLA has **no tree or speculative-verification API** — grepping the
`fla/` path list for `spec|tree|verify` yields only `fla/ops/atk/...` (unrelated "attn-key" op) and
`benchmarks/ops/verify.py` (a numerics checker). **But the
"many states at once" capability you need is expressible**: because `initial_state`/`h0` is indexed
by *sequence*, not by batch token-block, you can verify K draft nodes of one request by packing
`N = B*(K+1)` sequences of length 1 (for `fused_recurrent`) or length ≥ chunk boundary (for
`chunk_*`) into one `cu_seqlens`. This gives a *batched multi-state* (not tree-aware) verification:
each node is an independent sequence, so you get no intra-window triangular solve and no factoring —
you pay `N × state` I/O, which is the very cost TreeWY/Bole remove. `fla/ops/cp/` is context
parallel (`compress_h0`, `expand_h0`), **not** tree.

**Ascend relevance (big):** FLA already ships Triton-Ascend backends — very relevant for the port:

```
fla/ops/gated_delta_rule/backends/triton_ascend/{chunk_fwd.py, gate.py, wy_fast.py}
fla/ops/common/backends/triton_ascend/{chunk_delta_h.py, chunk_h.py, chunk_o.py,
                                      chunk_scaled_dot_kkt.py, gate.py}
fla/ops/utils/triton_ascend/{cumsum.py, solve_tril.py}
fla/modules/norm/triton_ascend/{fused_norm_gate.py, l2norm.py, layernorm.py}
fla/modules/causal_conv1d/backends/triton_ascend.py
fla/ops/atk/backends/triton_ascend/..., fla/ops/gdn2/backends/triton_ascend/...
+ .github/workflows/ascend-a2-ci.yml, ascend-a2-benchmark-ci.yml
+ .agents/skills/fla-ascend-performance/{SKILL.md,references/{TRAPS,cases,g-contiguous-loading}.md}
```
i.e. `solve_tril` (the `(I+A)^{-1}` triangular inverse that both ReplaySSM and TreeWY need) already
has an Ascend Triton implementation in FLA.

**Other backends in FLA:** `fla/ops/common/backends/tilelang/chunk_bwd.py`,
`fla/ops/attn/backends/tilelang/parallel.py`,
`fla/ops/generalized_delta_rule/dplr/backends/tilelang/`,
`fla/ops/gated_delta_rule/backends/flash_qla.py`. Dispatch via `fla.backends.dispatch`.

### 4.2 FlashInfer (`flashinfer-ai/flashinfer`, Apache-2.0, 6,812★)

Python API surface SGLang actually binds (`linear/kernels/gdn_flashinfer.py:136-143`):

```python
from flashinfer.gdn_decode import (gated_delta_rule_decode_pretranspose,
                                   gated_delta_rule_mtp)
from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp as gated_delta_rule_mtp_bf16
from flashinfer.gdn_prefill import chunk_gated_delta_rule
```

`gated_delta_rule_mtp(...)` signature as called by SGLang, verbatim:

```python
self._mtp_fn(q=query_mtp, k=key_mtp, v=value_mtp,
             initial_state=ssm_states,
             initial_state_indices=cache_indices_fi,
             A_log=A_log_fi, a=a_mtp, dt_bias=dt_bias_fi, b=b_mtp,
             scale=None, output=None,
             intermediate_states_buffer=intermediate_states_buffer_mtp,
             disable_state_update=True, use_qk_l2norm=True)
```
with `q` viewed `[B, draft_token_num, H, K]`, `v` `[B, draft_token_num, HV, V]`, and
`intermediate_states_buffer` `[slots, cache_steps, HV, V, K]` (SGLang K-last layout; the buffer
offset in the Triton reference is `cache_idx * cache_steps * HV * K * V`).

**Tree support: none.** `gdn_flashinfer.py:600+` verbatim:
```python
if retrieve_parent_token is not None:
    raise RuntimeError("FlashInfer GDN verify kernel only supports topk=1 "
                       "(retrieve_parent_token must be None).")
```
Requires `flashinfer >= 0.6.14`; SGLang pins `0.7.2`. SM90 and SM100 both supported for
decode/prefill/MTP; a separate bf16-state MTP variant exists
(`gdn_kernels/gdn_decode_bf16_state`). Alignment: SGLang enforces 32-byte alignment of
`ssm_states`/`intermediate_states_buffer` and falls back to Triton otherwise
(`_mutable_inputs_are_aligned`, `_FLASHINFER_GDN_ALIGNMENT = 32`).

Other GDN material in FlashInfer relevant to state management:
`flashinfer/gdn_kernels/blackwell/{gated_delta_net_chunked.py, gated_delta_net_cp.py,
gated_delta_net_cp_prefill.py, gated_delta_net_tile_scheduler.py, gdn_prefill.py}`,
`flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm{80,90,120}.py`,
`flashinfer/experimental/b12x/sequence/gdn_decode/{_commit.py,_recovery.py,api.py}` (note
`_recovery.py` — an explicit recovery/commit split), plus
`benchmarks/bench_gdn_{decode,fused_decode,paged_state,prefix_materialize,ucache_flush}.py`.
`gdn_paged_state` + `prefix_materialize` + `ucache_flush` are the state-checkpoint primitives DASC
talks about.

### 4.3 TileLang (`tile-ai/tilelang`, 8,842★)

**GDN exists only as examples, not as a shipped kernel library.** `examples/gdn/`:
`example_chunk_delta_h.py`, `example_chunk_o.py` (+bwd), `example_chunk_scaled_dot_kkt.py`,
`example_chunk_delta_bwd.py`, `example_cumsum.py`, `example_wy_fast.py` (+bwd_split),
`test_example_gdn_compilation.py`. `examples/kda/`: `chunk_delta_h_fwd.py`, `chunk_delta_bwd.py`,
`FLA_KDA/fla_chunk_delta.py`. `examples/linear_attention/` has Mamba chunk scan/state.
No tree/multi-state/spec API. License metadata is `NOASSERTION` on the GitHub API (a LICENSE file
exists but is not SPDX-recognized) — check before vendoring.

SGLang pins `tilelang==0.1.12` but I found **no** TileLang-backed GDN path in SGLang's linear-attn
backends (the `--linear-attn-backend` choices contain no tilelang). TileLang appears used elsewhere
(diffusion / attention variants), so **do not** assume a TileLang GDN option in SGLang.

### 4.4 Summary matrix for the Ascend port decision

| need | FLA (Triton) | FlashInfer | TileLang | SGLang in-tree |
|---|---|---|---|---|
| GDN chunked prefill | ✅ `chunk_gated_delta_rule`, chunk_size 64, varlen via `cu_seqlens` | ✅ `gdn_prefill.chunk_gated_delta_rule` | 🟡 examples only | ✅ vendored FLA + cutedsl + flashinfer |
| GDN single-step decode | ✅ `fused_recurrent_gated_delta_rule` | ✅ `gated_delta_rule_decode_pretranspose` | ❌ | ✅ + ReplaySSM buffered |
| batched multi-state (`N` independent states) | ✅ `initial_state` `[N,HV,K,V]` indexed by `cu_seqlens` | ✅ `initial_state_indices` (paged state) | ❌ | ✅ via `cache_indices` |
| chain verify over K drafts | ❌ | ✅ `gated_delta_rule_mtp` (topk=1 enforced) | ❌ | ✅ Triton / flashinfer / cutedsl |
| **tree** verify | ❌ | ❌ (raises) | ❌ | 🟡 serial parent-walk over full snapshots (`retrieve_parent_token`) |
| triangular inverse `(I+A)^{-1}` | ✅ `fla/ops/utils/solve_tril.py` (+ **triton_ascend** variant) | ✅ in `delta_rule_dsl` / `chunk` kernels | 🟡 `example_chunk_scaled_dot_kkt.py` | ✅ vendored |
| Ascend/NPU backend | ✅ **`*/backends/triton_ascend/` throughout** | ❌ (CUDA only) | ❓ | 🟡 `hardware_backend/npu/` (rollback-style, lossless bug #25587) |

---

## 5. (Bonus) Ascend/NPU-specific state of play

Two independent Ascend efforts exist, both **open RFCs**, both working on exactly this problem:

* **`vllm-project/vllm-ascend#19987` — "[RFC]: Support GDN RecoverSSM for Speculative Decoding on
  Ascend"** (open, 2026-10-27, `wxsIcey`). "RecoverSSM" is their name for the checkpoint+replay
  family. Verbatim: *"This RFC proposes extending RecoverSSM to Gated DeltaNet (GDN) on Ascend Model
  Runner V2 (MRV2). […] RecoverSSM addresses this by maintaining a single recurrent-state checkpoint
  and reconstructing the accepted state after rejection sampling."* Explicit requirement list
  includes *"Immediate rejection of the first draft token"*, *"Multiple requests with different
  accepted-token counts"*, and *"Correct convolution-state restoration"*. It says a **KDA** RecoverSSM
  already exists (the reference), GDN is the gap. It also proposes moving recovery records out of the
  KV cache, following upstream vLLM PR #63045.
* **`vllm-project/vllm-ascend#18797` — "[RFC]: Tree Speculative Decoding"** (open, 2026-10-15,
  `chenaoxuan`). Contains a GDN plan, verbatim: *"**GDN.** Each chain updates its own recurrent state.
  Only the accepted chain is kept."* and *"Recurrent or compressed state is recomputed along the
  accepted path."* i.e. **per-chain state + recompute-on-accept**, the naive-but-simple approach;
  explicitly outside the official roadmap and not committed to upstream.

So on Ascend the ecosystem is roughly where CUDA was ~2 quarters ago: per-chain state fan-out +
recompute, with RecoverSSM/ReplaySSM-style single-checkpoint designs only at RFC stage.

---

## 6. Caveats / things I could not verify

1. **Bole's "integration into SGLang"** — no public patch, branch, or upstream PR from the authors.
   Only the base-tree URL `sgl-project/sglang/tree/release/v0.5.12` appears in the paper. The one
   public Bole implementation (`sgl PR #46702`) is a third party's chain-only subset. **Do not cite
   Bole as "available in SGLang".**
2. **Whether SGLang NPU GDN MTP is now lossless** — issue #25587 is *closed* but the NPU file I read
   still uses `conv_state_rollback` position-shifting, which the issue itself argues cannot be
   correct. Unverified; needs a run on hardware.
3. **FlashInfer `_recovery.py` / `gated_delta_rule_mtp` internals** — I only read the caller
   (`gdn_flashinfer.py`) and the file list, not the kernel bodies, so the exact `intermediate_states_buffer`
   semantics on the FlashInfer side are inferred from the Triton reference, not confirmed.
4. **`vllm-ascend` KDA RecoverSSM** — referenced as pre-existing by RFC #19987 but I did not locate
   the code (the RFC does not link a PR).
5. Code-search indexing of `sgl-project/sglang` is broken; anything you want to *prove* must be
   fetched by path.
