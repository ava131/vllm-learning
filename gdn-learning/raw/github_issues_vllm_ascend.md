# GitHub archaeology: the "n+1 recurrent state" problem in vLLM and vLLM-Ascend

Scope: how `vllm-project/vllm` and `vllm-project/vllm-ascend` handle the fact that speculative
decoding (EAGLE / MTP / n-gram / DFlash) must verify `n+1` tokens against a *sequential, non-rewindable*
recurrent state in hybrid models (Gated DeltaNet / Mamba-2 / KDA / ShortConv).
Light cross-check on `sgl-project/sglang` at the end.

All data below was pulled live via `gh api` / `gh search` / `raw.githubusercontent.com` on the date of
this run. Every number, title, state and date is from an actual API response; queries that returned
nothing are listed explicitly in §8.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Does vLLM allow spec decoding on GDN / Mamba hybrids **today**? | **Yes.** Grepping `vllm/config/speculative.py` for `mamba|hybrid|recurrent|gdn|linear_attn` returns only `bailing_hybrid*_mtp` model-type aliases — there is no rejection path. N-gram, MTP, EAGLE, DFlash and DSpark all run on Qwen3.5/3.6/3.8, Qwen3-Next, Nemotron-H, Kimi-K3, LFM2, GLM-5.3. |
| What is the **current mechanism** for the n+1 problem? | **Per-draft-token state snapshots.** The Mamba state page is replicated `1 + num_speculative_tokens` times per request per recurrent layer, indexed as block-table columns `[B, num_spec+1]`; the verify kernel writes one candidate state per draft position and the sampler picks slot `num_accepted_tokens - 1`. Conv state instead gets a widened window (`conv_kernel - 1 + num_spec`) that is rewound in place. |
| Is that still the mechanism on `main`? | **Yes** — unless `--use-replayssm` is on, in which case `num_speculative_blocks = 0`. |
| What superseded the ReplaySSM spec work? | **RecoverSSM** (verify-from-checkpoint + post-sampling commit from a small replay record). Merged for Kimi-K3 KDA (#63045, 2026-10-24). Open for Qwen4Exp GDN (#58863), Qwen3.5/Qwen3-Next (#61309), PP support (#59983). |
| What does vLLM-Ascend support today? | **Only the snapshot design.** Ascend mirrors upstream: `[B, K+1]` GDN state table, `num_accepted_tokens`, `spec_state_indices_tensor`. RecoverSSM on Ascend is **RFC-only** (#19733, #19987) and is item 10 of the Q4 2026 spec-decode roadmap (#18827). |
| Tree spec decoding on GDN? | vLLM: RFC only (#54080 TreeWY). Ascend: RFC only (#18797). |
| Where does the memory number come from? | `MambaSpec.max_memory_usage_bytes` returns `page_size_bytes * (1 + num_speculative_blocks)`; `mamba_cache_mode="align"` returns `page_size_bytes * (2 + num_speculative_blocks + num_prefill_checkpoint_blocks)`. |

---

## 1. Code-level ground truth: how vLLM implements n+1 today

This is the most important section — the issue/PR discussion below only makes sense against it.

### 1.1 The state page is replicated `1 + K` times

`vllm/model_executor/layers/mamba/abstract.py` (~L72–91), `MambaBase.get_kv_cache_spec`:

```python
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            # ReplaySSM verifies the whole window off one checkpoint, so it
            # never writes the baseline's per-draft-token state slots.
            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_replayssm
                else vllm_config.num_speculative_tokens
            ),
```

So: **baseline = `1 + K` full recurrent-state pages per request per recurrent layer**; RecoverSSM/ReplaySSM = **`1`**.

The accounting is explicit in `vllm/v1/kv_cache_interface.py` (~L1133–1141):

```python
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (
                2 + self.num_speculative_blocks + self.num_prefill_checkpoint_blocks
            )
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)
```

### 1.2 The `n+1` slots are block-table columns, not a tensor axis

`vllm/v1/attention/backends/gdn_attn.py`:

```
L107:  spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
L204:  self.spec_state_indices_tensor: torch.Tensor = torch.empty(
L205:      (self.decode_cudagraph_max_bs, self.num_spec + 1),
L447:  spec_state_indices_tensor = block_table_tensor[
L448:      :num_spec_decodes, : self.num_spec + 1
```

and the same shape assumption is documented in
`vllm/v1/attention/backends/utils.py` `mamba_get_block_table_tensor` (~L1299–1325):

```
    - "none":  input  (#requests, 1 + num_speculative_blocks);
               output (#requests, 1 + num_speculative_blocks).
    - "align": input  (#requests, cdiv(max_model_len, block_size));
               output (#requests, 1 + num_speculative_blocks), which are the last
               1 + num_speculative_blocks of each request.
```

The allocator really does hand out `1 + K` pages for one request —
`vllm/v1/core/single_type_kv_cache_manager.py`, `MambaManager`:

```
L1445:  self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
L1713:  if self.mamba_cache_mode != "align":
L1714:      # Allocate extra `num_speculative_blocks` blocks for
L1716:      if self.num_speculative_blocks > 0:
L1718:          self.kv_cache_spec.block_size * self.num_speculative_blocks
```

### 1.3 Conv state is handled differently (widened window, in-place rewind)

`vllm/model_executor/layers/mamba/mamba_utils.py`, `MambaStateShapeCalculator`:

```python
    # gated_delta_net_state_shape(...)
    conv_state_shape = cls._orient_conv_shape(
        divide(conv_dim, tp_world_size),
        conv_kernel_size - 1 + num_spec,          # <- one extra slot per draft token
    )
    temporal_state_shape = (divide(num_v_heads, tp_world_size), head_v_dim, head_k_dim)
```

i.e. conv history is a *single* tensor with `num_spec` extra columns, rewound by
`num_accepted_tokens` (`causal_conv1d_update`). Only the **recurrent/SSM matrix** state is
replicated as pages.

### 1.4 The GDN verify kernel writes a state per draft position

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`:

```
L113:  def supports_fused_mtp_decode() -> bool:
L116:      torch.ops._C, "fused_gdn_decode_post_conv_mtp"
L729:  self.enable_fused_gdn_spec_decode = (
L730:      self.gdn_decode_kernel == "cuda" and self.speculative_config is not None
L733:      logger.info_once("GDN spec decode kernel: %s", self.gdn_decode_kernel)
```

The `*_post_conv_mtp` kernel exists specifically because the verify step has to emit
`K+1` candidate states in one launch.

### 1.5 Who consumes the slots

`vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (`num_accepted_tokens_gpu`, `_scatter_num_accepted_kernel`
storing `max(num_sampled, 1)`), and `vllm/v1/worker/mamba_utils.py`
(`preprocess_mamba_align_fused_kernel`, `precopy_mamba_align_fused_kernel`, `accept_token_bias = aligned_new_computed - num_tokens_running_state`).

### 1.6 Config surface on `main`

`vllm/config/cache.py` (~L304–330):

```python
    mamba_cache_mode: MambaCacheMode = "none"
    replayssm_buffer_len: int = Field(default=16, gt=0)
    use_replayssm: bool = False
    """Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip
    the per-step full-state store, writing the checkpoint back only on flush."""
    use_kda_recoverssm: bool = field(default=False, init=False)
    """Whether Kimi-K3 KDA uses RecoverSSM speculative decode."""
```

`--use-replayssm` now means "the RecoverSSM/ReplaySSM family", not just the Dao-Lab kernel.

---

## 2. vLLM — the ReplaySSM → RecoverSSM lineage

This is the core of what vLLM is *doing* about n+1.

### 2.1 RFCs (discussion, not code)

| # | Title | State | Created | Notes |
|---|---|---|---|---|
| [47572](https://github.com/vllm-project/vllm/issues/47572) | [RFC]: ReplaySSM: cache SSM inputs instead of state for faster standard and speculative decode (Mamba2 + GDN) | OPEN | 2026-07-03 | The main RFC. Dao AI Lab × NVIDIA. |
| [49232](https://github.com/vllm-project/vllm/issues/49232) | [RFC]: ReplaySSM: cache SSM inputs instead of state … (Mamba2 + GDN) | OPEN | 2026-07-20 | Duplicate/companion RFC of #47572. |
| [46187](https://github.com/vllm-project/vllm/issues/46187) | [Feature]: Faster hybrid-SSM decode via input "replay" ring buffer (ReplaySSM) | OPEN | 2026-06-19 | Earlier feature request, assigned to the ReplaySSM authors. |
| [54080](https://github.com/vllm-project/vllm/issues/54080) | [RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models | OPEN | 2026-08-27 | The *tree* extension of the same idea. |

The RFC states the n+1 problem in exactly our terms. Verbatim from #47572:

> "Speculative decoding must roll back rejected draft tokens, but the SSM state update is irreversible:
> once history is summarized into the recurrent state, a token cannot be subtracted back out. To support
> rollback, the current implementation keeps a separate recurrent state per draft token, which costs extra
> memory traffic and extra HBM footprint (about (1 + num_spec)× the state), hurting both latency and maximum
> concurrency."

and the proposed fix:

> "ReplaySSM instead caches recent inputs, reconstructing the state only when needed … For speculative decode,
> caching recent inputs makes rollback a ring-buffer pointer move, cutting the per-step rollback cost from O(T)
> to O(1) (`T = num_spec_tokens + 1`) and removing the per-draft-token state storage."

Config flags in the RFC: `--use-replayssm`, `--use-replayssm-spec` (requires spec decoding **and**
`--mamba-cache-mode none`), `--replayssm-buffer-len`.

Prefix-caching interaction was raised in the RFC thread by @ZJY0516 and answered by @Johnny-Liou
(2026-08-01) with a "ring cache participates in the checkpoint" idea; @ZJY0516 pushed back
("it would add a lot of complexity to the KV cache manager, and the ring cache isn't managed by it now").

### 2.2 PRs — what actually merged, and what was closed

| # | Title | State | Created | Closed/Merged |
|---|---|---|---|---|
| [48018](https://github.com/vllm-project/vllm/pull/48018) | [Kernel] ReplaySSM: cache SSM inputs for faster Mamba2 **standard** decode | **MERGED** | 2026-07-08 | merged ~2026-07-24 |
| [48792](https://github.com/vllm-project/vllm/pull/48792) | [Kernel] ReplaySSM: … for faster Gated DeltaNet **standard** decode | CLOSED (unmerged) | 2026-07-15 | 2026-10-12 |
| [47576](https://github.com/vllm-project/vllm/pull/47576) | [Kernel] ReplaySSM: cache SSM inputs instead of state … (Mamba2 + GDN) — the umbrella draft PR | CLOSED (unmerged) | 2026-07-03 | 2026-10-12 |
| [49847](https://github.com/vllm-project/vllm/pull/49847) | [Kernel] ReplaySSM: cache SSM inputs for faster **Mamba2 speculative** decode | CLOSED (unmerged) | 2026-07-26 | 2026-10-12 |
| [49887](https://github.com/vllm-project/vllm/pull/49887) | [Kernel] ReplaySSM: cache SSM inputs for faster **Gated DeltaNet speculative** decode | CLOSED (unmerged) | 2026-07-26 | 2026-10-12 |

**Key reading:** the ReplaySSM *standard-decode* kernel for Mamba2 landed. The entire
**speculative-decode** branch of ReplaySSM (Mamba2 + GDN) was closed unmerged on 2026-10-12, the same
day as the umbrella PR. No closing rationale comment was posted (only a Mergify "merge conflicts" bot note
and the community bot greeting were available via the API).

### 2.3 RecoverSSM — the successor that is actually landing

RecoverSSM keeps the same insight (verify from one checkpoint, reconstruct on commit) but stores
per-token *replay records* instead of a ring cache. It is now the active design.

| # | Title | State | Created | Note |
|---|---|---|---|---|
| [63045](https://github.com/vllm-project/vllm/pull/63045) | [Kimi K3] KDA RecoverSSM: CuTe verify and commit kernels, records outside the KV cache | **MERGED** 2026-10-24 | 2026-10-22 | The reference implementation. |
| [58863](https://github.com/vllm-project/vllm/pull/58863) | [Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv | OPEN | 2026-09-26 | First GDN port. |
| [61309](https://github.com/vllm-project/vllm/pull/61309) | [Spec Decode][Qwen3.5] RecoverSSM for Qwen3.5 / Qwen3-Next GDN | OPEN | 2026-10-13 | Stacked on #58863; measured +8.6% @ K=5, c=32. |
| [59983](https://github.com/vllm-project/vllm/pull/59983) | [Spec Decode][Mamba] RecoverSSM: support pipeline parallelism | OPEN | 2026-10-04 | Removes the `pipeline_parallel_size > 1` rejection. |

#58863 states the current behaviour and the fix in one paragraph (verbatim):

> "Speculative decoding on Gated DeltaNet today snapshots the full recurrent state after every draft position,
> and reserves one Mamba block per draft position. This PR brings the **RecoverSSM** protocol that Kimi-K3 KDA
> already uses to the Qwen GDN layers of Qwen4Exp (Qwen3.8-Flash-Next), plus its PLE dilated short conv:
> **Verify** runs from a read-only checkpoint and stores a small per-token replay record
> `[correction (V), normalized key (K), gate (1)]` in fp32. **Commit** (after sampling) replays only the
> accepted tokens and writes the recurrent state once … `num_speculative_blocks = 0`: no per-draft Mamba blocks."

#63045 moved the replay records **out of the KV pages** (important for memory and for KV connectors):

> "They used to be appended to every Mamba page, which grew the page and the attention block size and made KV
> connectors copy them. Each KDA layer now keeps them in a buffer with one row per schedulable request
> (`max_num_seqs`), indexed by the request's row in the speculative batch. … (Kimi-K3 at TP8: 1024 -> 896 tokens,
> the same as without RecoverSSM)"

Relevant new code on `main`: `vllm/models/kimi_k3/nvidia/recoverssm/{kernels.py,cutedsl}`,
`vllm/models/kimi_k3/nvidia/ops/recoverssm.py`, `vllm/v1/worker/gpu/model_states/recoverssm.py`.

### 2.4 Tree speculative decoding on GDN — discussion only

| # | Title | State | Created |
|---|---|---|---|
| [54080](https://github.com/vllm-project/vllm/issues/54080) | [RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models | OPEN | 2026-08-27 |
| [18797](https://github.com/vllm-project/vllm-ascend/issues/18797) | (Ascend) [RFC]: Tree Speculative Decoding | OPEN | 2026-10-15 |

#54080 quantifies the n+1 cost for a real model (verbatim):

> "With `mamba_cache_mode="all"`, a draft of `N` positions needs `N+1` full state snapshots per sequence per
> layer. On Qwen3.5-397B that is 90 MiB/seq i.e. 360 MiB for a depth-3 chain and ~1.2 GiB for a `[2,2,2]` tree
> (14 nodes), out of the same budget as the KV cache."

and notes TreeWY is *slower* than ReplaySSM for chains:

> "I benchmarked ReplaySSM against my implementation on Qwen3.5-35B, and ReplaySSM is the more performant one for
> chain speculative decoding in hybrid models."

A detailed third-party design addendum (tree scan, ancestor-masked FA2 bias, accepted-path replay) was posted
by @MaCoredroid on 2026-09-10 in that thread.

`#52959` ([RFC]: Internal State Checkpoints for Mamba Align Mode, OPEN 2026-08-19) and
`#53401` ([RFC]: Experimental Recirculation for causal decoders, OPEN 2026-08-22) touch adjacent state-checkpoint
machinery but not the n+1 verify problem directly.

---

## 3. vLLM — MERGED facts about spec + recurrent state

Separated from discussion on purpose. All five are confirmed `merged=true` via the pulls API.

| # | Title | Merged | Why it matters for n+1 |
|---|---|---|---|
| [56531](https://github.com/vllm-project/vllm/pull/56531) | [Bugfix] Route every speculation-capable row through the speculative path so recurrent state stays consistent | 2026-10-05 | Defines the two state layouts and makes them agree. |
| [50272](https://github.com/vllm-project/vllm/pull/50272) | [Bugfix] Fix speculative decoding for short_conv (LFM2) models | 2026-08-22 | Adds `num_spec` to the short-conv state shape so the window can be rewound. |
| [45845](https://github.com/vllm-project/vllm/pull/45845) | [v1][kvcache] Honor prefix-cache retention interval for Mamba/linear attention | 2026-06-23 | Bounds how many full-state snapshots prefix caching retains. |
| [60050](https://github.com/vllm-project/vllm/pull/60050) | [Perf][GDN] Internal prefill checkpoints for the FlashInfer GDN prefill backend | 2026-10-14 | RFC #52959 implementation; avoids a second forward just to emit a boundary state. |
| [63045](https://github.com/vllm-project/vllm/pull/63045) | [Kimi K3] KDA RecoverSSM: CuTe verify and commit kernels, records outside the KV cache | 2026-10-24 | First merged `num_speculative_blocks = 0` speculative path. |

#56531 is the single best verbatim description of the current design (quote):

> "**Non-speculative path.** One slot per request: `block_table[:, 0]`. What the step writes is what the next
> step reads.
> **Speculative path.** A window of `num_spec + 1` slots per request (`spec_state_indices_tensor`). A verification
> step writes a candidate state per draft position, and the accepted prefix decides which one is authoritative.
> Kernels pick it as `num_accepted_tokens - 1` — the count includes the bonus token, so `1` is the minimum.
> The two layouts coincide only while `num_accepted_tokens == 1`, i.e. slot `0`."

#50272 (quote) confirms the design split:

> "`MambaStateShapeCalculator.short_conv_state_shape` sizes the conv state `conv_kernel - 1` (no `num_spec`), so
> there is no room to roll the window back (mirror: `mamba2_state_shape` adds `num_spec`)."

---

## 4. vLLM — open correctness / memory issues that are consequences of the n+1 design

Grouped, most-n+1-relevant first. These are the "what breaks because of snapshots" pile.

### 4.1 The state slot bookkeeping is wrong in specific situations

| # | Title | State | Created | 1–3 sentence summary |
|---|---|---|---|---|
| [62304](https://github.com/vllm-project/vllm/issues/62304) | [Bug]: Hybrid GDN (Qwen3.5) + n-gram spec: greedy outputs go wrong when drafts are partly accepted; the GDN state misses the last accepted token's update | OPEN | 2026-10-18 | 8/192 requests wrong at k=4, deterministic; needs *partial* acceptance to trigger. Diagnosed as a row taking the non-speculative path after an accepting step, so it read slot 0 which lacked the accepted tokens' writes. |
| [63620](https://github.com/vllm-project/vllm/issues/63620) | [Bug][Hybrid models] Padding rows of graph-replayed decode batches point recurrent state at `NULL_BLOCK_ID`, which aliases the shared null page | OPEN | 2026-10-25 | Padding rows in a replayed FULL graph get block 0, which is the shared null page for *every* group/layer — a state kernel that does not skip them scribbles over a page everyone aliases. |
| [55951](https://github.com/vllm-project/vllm/issues/55951) | [Bug] PP>1 + spec decode: mamba align postprocess indexes block table by batch row of a stale (pp_size-old) batch → recurrent state copied to wrong blocks | OPEN | 2026-09-08 | The align-mode copy uses a batch row from the wrong (already-advanced) batch, so states land in the wrong blocks → NaN / silent recall loss. |
| [61762](https://github.com/vllm-project/vllm/issues/61762) | [Bug]: Scheduler livelock with multi-module MTP + Mamba align prefix caching when `prompt_len % block_size < num_speculative_tokens` | OPEN | 2026-10-15 | Align block boundaries and the spec window collide. |
| [54392](https://github.com/vllm-project/vllm/issues/54392) | [Bug]: PD-admitted Mamba request is spec-padded before prefill completes, then align split truncates the 8-token window to 5 | OPEN | 2026-08-30 | PD + spec padding + align splitting interact. |
| [55951](https://github.com/vllm-project/vllm/issues/55951) / [53142](https://github.com/vllm-project/vllm/issues/53142) | align precopy: illegal memory access on prefix-cache resume with explicit `--block-size` (state column seeded with wrong block size) | OPEN / OPEN | 2026-08-20 | The `precopy_mamba_align_fused_kernel` src column uses the wrong block size. |
| [59933](https://github.com/vllm-project/vllm/issues/59933) | [Bug]: RecoverSSM align mode commits the final SSM state to an unwritten block at exact block boundaries (floor vs ceil-1) | CLOSED | 2026-10-04 | `floor(x/B)` vs `ceil(x/B)-1` disagree exactly when `x` is a multiple of `B`. Filed against a fork, but points at `vllm/v1/worker/mamba_utils.py` and `vllm/v1/worker/gpu/model_states/recoverssm.py` on main. |

### 4.2 Spec verify + prefix caching corrupts the cached recurrent state

| # | Title | State | Created | Summary |
|---|---|---|---|---|
| [53912](https://github.com/vllm-project/vllm/issues/53912) | [Bug]: prefix caching + MTP still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but unfixed) | OPEN | 2026-08-26 | The long-running tracker for "EAGLE peek-and-drop applied to an un-rewindable state". |
| [43559](https://github.com/vllm-project/vllm/issues/43559) | [Bug]: Accuracy drops ~20% when `--enable-prefix-caching` + MTP (Qwen3.6 35B-A3B) | CLOSED | 2026-05-25 | The original report; closed while #53912 says unfixed. |
| [50630](https://github.com/vllm-project/vllm/issues/50630) | No capability flag declares which KV-cache kinds may be peeked past a candidate boundary — each margin implementation excludes Mamba by hand | OPEN | 2026-07-31 | Architectural: `MambaManager.find_longest_cache_hit` takes `drop_eagle_block` and never reads it. |
| [48375](https://github.com/vllm-project/vllm/pull/48375) | [BugFix] Honor `drop_eagle_block` in `MambaManager` | OPEN PR | 2026-07-12 | First fix attempt. |
| [57128](https://github.com/vllm-project/vllm/pull/57128) | [Bugfix] `MambaManager.find_longest_cache_hit` ignores `drop_eagle_block`, silent corruption with prefix caching + MTP/EAGLE | OPEN PR | 2026-09-16 | Later fix attempt for the same bug. |
| [47861](https://github.com/vllm-project/vllm/pull/47861) | [BugFix] Fix MTP prefix cache correctness for hybrid Mamba models | CLOSED (unmerged) | 2026-07-07 | Took the opposite tack: "Avoid applying EAGLE/MTP cache peek-and-drop semantics to Mamba/GDN state cache groups, since recurrent state snapshots cannot be rewound like token KV blocks." Closed 2026-07-19. |
| [39273](https://github.com/vllm-project/vllm/issues/39273) | [Bug]: N-gram speculative decoding produces corrupted output on hybrid GDN (Qwen3.5) models | OPEN | 2026-04-08 | The oldest open report in this family (Qwen3.5-9B, 24 GDN + 8 full-attn layers). |
| [61738](https://github.com/vllm-project/vllm/issues/61738) | [Bug]: Wide speculative verify (`num_speculative_tokens ≥ 7`) samples EOS at `prompt_len+8` on hybrid GDN (Qwen3.5) | CLOSED | 2026-10-15 | Byte-clean until the EOS row; reproduced on both model runners. |
| [52873](https://github.com/vllm-project/vllm/issues/52873) | [Bug]: Qwen3-Next GDN + MTP: crossing position 32768 permanently kills draft acceptance engine-wide | CLOSED | 2026-08-19 | State-slot / position interaction. |
| [53505](https://github.com/vllm-project/vllm/issues/53505) | [Bug]: [SpecDecode] Hybrid Mamba (align) corrupts under spec decoding when a KV connector is attached — even with zero retrieved tokens | CLOSED | 2026-08-24 | Connector + align-mode state interaction. |

### 4.3 Memory / concurrency consequences of keeping `1+K` states

| # | Title | State | Created | Summary |
|---|---|---|---|---|
| [59117](https://github.com/vllm-project/vllm/issues/59117) | [RFC]: [Spec decode] KV-cache pool collapse when a draft model introduces a new cache spec type (hybrid GDN + DFlash) | OPEN | 2026-09-29 | Concurrency 22.53x → **2.05x** on 2×RTX 5090 with a 27B GDN hybrid + DFlash (K=7). `blocks_per_request` 14 → 86 @16k. Includes a dedicated-draft-pool POC. |
| [55533](https://github.com/vllm-project/vllm/issues/55533) | [Bug][Spec Decode] Hybrid GDN + MTP: scheduler runs only ~3 concurrent sequences at batch ≥ 4 — acceptance/throughput collapse | OPEN | 2026-09-06 | The `1+K` reserve dominates the pool at small batch. |
| [60008](https://github.com/vllm-project/vllm/issues/60008) | [Performance]: Hybrid Mamba prefix caching (`align`) costs 11-16% throughput on Nemotron-3.5-Lightning with no cache hits | OPEN | 2026-10-05 | Cost of align-mode snapshot machinery even when nothing hits. |
| [60838](https://github.com/vllm-project/vllm/issues/60838) | [Bug]: `--mamba-block-size` has no effect in any valid configuration after the removal of `mamba_cache_mode="all"` | OPEN | 2026-10-09 | Follow-on from dropping the "all" (every-step snapshot) mode. |
| [62926](https://github.com/vllm-project/vllm/pull/62926) | [Perf][Mamba] Avoid initial-state snapshots for single-sequence Mamba2 prefill | OPEN PR | 2026-10-21 | Same title/date as [#62922](https://github.com/vllm-project/vllm/pull/62922) (CLOSED same day) — i.e. a re-open, not a different change. |
| [55196](https://github.com/vllm-project/vllm/issues/55196) | [RFC]: fp8 KV cache gives little to no memory benefit on Mamba/GDN hybrid models | OPEN | 2026-09-03 | Because the recurrent page dwarfs the token KV page. |

### 4.4 ReplaySSM-in-production bugs (already-merged standard-decode path)

| # | Title | State | Created |
|---|---|---|---|
| [63175](https://github.com/vllm-project/vllm/issues/63175) | [Bug]: Triton ReplaySSM (V1) drops un-flushed ring tokens when a running request skips a step | OPEN | 2026-10-22 |
| [60346](https://github.com/vllm-project/vllm/issues/60346) | [Bug]: `append_replayssm_ring` asserts for supported non-divisible Mamba group counts | CLOSED | 2026-10-07 |

#63175 (quote): "When a request is unscheduled, the V1 runner drops it from the persistent batch and re-adds it
later via `InputBatch.add_request`, which always does `self.replayssm_decode_base[req_index] = request.num_tokens`.
So the ring origin jumps to 'now'. The decode tokens still sitting in the ring (not yet flushed into `ssm_state`)
get masked out and overwritten."

### 4.5 Related design discussion

| # | Title | State | Created |
|---|---|---|---|
| [52959](https://github.com/vllm-project/vllm/issues/52959) | [RFC]: Internal State Checkpoints for Mamba Align Mode | OPEN | 2026-08-19 |
| [53041](https://github.com/vllm-project/vllm/issues/53041) | [Discussion] Tiered SWA/Mamba checkpointing (HBM tail + store periodic) + recompute backfill for divergent hybrid prefix hits | OPEN | 2026-08-20 |
| [55524](https://github.com/vllm-project/vllm/issues/55524) | [RFC] Mamba2: exact-replay decode so that prefill, chunked prefill and decode produce identical bits | OPEN | 2026-09-06 |
| [64150](https://github.com/vllm-project/vllm/issues/64150) | [RFC]: Pluggable KV cache eviction policy for hybrid models | OPEN | 2026-10-27 |
| [53401](https://github.com/vllm-project/vllm/issues/53401) | [RFC]: Experimental Recirculation for causal decoders | OPEN | 2026-08-22 |

### 4.6 Still-unfinished plumbing

| # | Title | State | Created |
|---|---|---|---|
| [38898](https://github.com/vllm-project/vllm/issues/38898) | [Feature]: Mamba `DS` conv state layout \| Support speculative decoding with `mamba_cache_mode=align` | CLOSED | 2026-04-03 (closed 2026-08-29) |
| [42792](https://github.com/vllm-project/vllm/pull/42792) | [WIP][Model Runner V2] support spec decode + mamba align prefix caching | OPEN PR | 2026-05-15 |

---

## 5. vLLM-Ascend — what exists today

### 5.1 Current mechanism: identical to upstream (snapshots), with NPU kernels

Ascend does **not** have an independent solution. `vllm_ascend/ops/gdn.py` and
`vllm_ascend/ops/gdn_attn_builder.py` consume the upstream metadata contract
(`spec_state_indices_tensor`, `non_spec_state_indices_tensor`, `num_accepted_tokens`,
`spec_sequence_masks`, `spec_token_indx`), and the state shape comes from the upstream calculator:

```python
# vllm_ascend/ops/gdn.py (~L360)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size, self.num_k_heads, self.num_v_heads,
            self.head_k_dim, self.head_v_dim, self.conv_kernel_size,
            self.num_spec,          # <- same n+1 conv widening as upstream
        )
```

The `[B, K+1]` state table is documented in `vllm_ascend/ops/gdn.py` (~L162–186),
`_flatten_recurrent_state_indices` (verbatim):

> "`torch_npu.npu_recurrent_gated_delta_rule` requires `len(ssm_state_indices) == T` where T is the number of
> q/k/v rows. Block-table-derived index tensors carry one entry per request row … while **speculative rows
> carry one entry per draft slot**, including slots left unused when a sequence capped at max_model_len
> carries fewer than `num_spec + 1` query tokens."

and `causal_conv1d_update` is called with `max_query_len=self.num_spec + 1` (`vllm_ascend/ops/gdn.py` ~L536).

MTP/EAGLE on GDN hybrids is therefore **supported and widely used** on Ascend (A2/A3/310P), with no
RecoverSSM equivalent yet: `vllm_ascend/worker/v2/spec_decode/` contains
`dflash, dflash2, dspark, eagle, gemma4, mtp, multi_module_mtp, standalone_ar, target_dependent_ar` —
no `recoverssm`.

### 5.2 Ascend RFCs: RecoverSSM for GDN (the answer to "what are they doing")

| # | Title | State | Created | Author |
|---|---|---|---|---|
| [19987](https://github.com/vllm-project/vllm-ascend/issues/19987) | [RFC]: Support GDN RecoverSSM for Speculative Decoding on Ascend | OPEN | 2026-10-27 | wxsIcey |
| [19733](https://github.com/vllm-project/vllm-ascend/issues/19733) | [RFC]: RecoverSSM state recovery for speculative decoding on Ascend | OPEN | 2026-10-25 | drslark |
| [18827](https://github.com/vllm-project/vllm-ascend/issues/18827) | [RFC]: Speculative Decoding Roadmap Q4 2026 | OPEN | 2026-10-15 | drslark (item 10 = GDN SSM/Conv state recomputation) |
| [18797](https://github.com/vllm-project/vllm-ascend/issues/18797) | [RFC]: Tree Speculative Decoding | OPEN | 2026-10-15 | chenaoxuan |
| [15149](https://github.com/vllm-project/vllm-ascend/issues/15149) | [RFC]: add adaptive speculative (Dcut) verification for vllm-ascend | OPEN | 2026-08-27 | czfsdg |
| [19845](https://github.com/vllm-project/vllm-ascend/issues/19845) | [RFC]: Extend adaptive verification support across ascend attention backends | OPEN | 2026-10-26 | — |
| [19120](https://github.com/vllm-project/vllm-ascend/issues/19120) | [RFC]: Track vLLM speculative decoding issues and fixes | OPEN | 2026-10-19 | drslark |

#19987 states the problem and the plan (verbatim):

> "During speculative decoding, the target model needs to verify multiple draft tokens while maintaining
> recurrent states. **The conventional approach may retain intermediate SSM states for each speculative
> position, increasing Mamba/SSM cache consumption as the speculative window grows.**
> RecoverSSM addresses this by maintaining a single recurrent-state checkpoint and reconstructing the accepted
> state after rejection sampling."

and mandates following upstream #63045:

> "**3. Move Recovery Records Outside the KV Cache.** Follow the design introduced in upstream
> [vLLM #63045]. … allocate per-layer scratch buffers indexed by speculative-batch request rows."

#19733 (the parallel, more scheduling-oriented RFC) adds:

> "Follow upstream vllm-project/vllm#63045: allocate per-layer, per-request scratch records outside KV pages,
> indexed by speculative-batch row. Restore the persistent conv + SSM page layout and rebuild the commit context
> when caches are rebound. Adapt the NPU kernels and metadata interfaces; the upstream CuTe kernels target NVIDIA
> SM10x."

Roadmap #18827, item 10 verbatim: *"**10. GDN SSM State / Conv State recomputation** — Reduce GDN-layer cache
memory usage by recomputing selected SSM State and Conv State. … Adapt RecoverSSM state recovery, starting with
MRV2 KDA and extending to GDN SSM/Conv State. (#19733)"*

**Status: RFC-only. No Ascend RecoverSSM implementation PR exists yet** (searched `RecoverSSM`, `recover`,
`state recovery`, `ssm_state_indices` across the repo).

### 5.3 Ascend: the variable-verify-length problem (their version of the n+1 headache)

Because Ascend supports *trimmed* draft batches (structured output / D-Cut), per-request verify lengths differ,
which breaks the fixed `[B, K+1]` table.

| # | Title | State | Created | Summary |
|---|---|---|---|---|
| [15149](https://github.com/vllm-project/vllm-ascend/issues/15149) | [RFC]: add adaptive speculative (Dcut) verification for vllm-ascend | OPEN | 2026-08-27 | The clearest statement of the problem on Ascend. |
| [17763](https://github.com/vllm-project/vllm-ascend/pull/17763) | [Feature][GDN] Support adaptive verification for variable-length GDN | OPEN PR | 2026-09-29 | Implementation attempt. |
| [17837](https://github.com/vllm-project/vllm-ascend/issues/17837) | [Bug]: Qwen3.5 MTP3 structured output corruption from ragged GDN state indexing | OPEN | 2026-09-30 | Trigger = batch with different actual verify lengths. |
| [17839](https://github.com/vllm-project/vllm-ascend/pull/17839) | [BugFix][GDN] Preserve recurrent state rows for ragged MTP batches (v0.26) | OPEN PR | 2026-09-30 | Backport fix for #17837. |
| [18612](https://github.com/vllm-project/vllm-ascend/pull/18612) | [BugFix][GDN] Preserve recurrent state rows for ragged MTP batches | OPEN PR | 2026-10-13 | Main-branch fix for #17837. |
| [17856](https://github.com/vllm-project/vllm-ascend/issues/17856) | [Bug][310P] `recurrent_gated_delta_rule_v310` aborts the whole batch when `accepted_prev > current query length` (MTP + structured output) | OPEN | 2026-09-30 | 310P counterpart of #17837; #17839 does not cover it. |

#15149 (verbatim, the key quote):

> "Qwen3.5 and Qwen3.6 make this problem harder because they contain stateful Gated Delta Network (GDN) layers.
> GDN maintains causal-convolution state and recurrent state across decode steps. **Once different requests retain
> different draft-prefix lengths, the native fixed-length metadata can update the wrong state position or state
> slot.** Therefore, D-Cut cannot be implemented safely by only deleting draft tokens in the scheduler."

#18612 (verbatim):

> "Keeps `ssm_state_indices` as a `[B, K+1]` state table instead of flattening it. Preserves speculative history
> when one request has all drafts trimmed. Always supplies accepted-token metadata to the GDN builder."

### 5.4 Ascend: CRASH-class bugs in the GDN spec-decode path

| # | Title | State | Created |
|---|---|---|---|
| [19608](https://github.com/vllm-project/vllm-ascend/issues/19608) | [Bug]: qwen3.8-27B-w8a8 — AICore error (MTE invalid GM access) crash in GDN spec-decode path with MTP + FULL_DECODE_ONLY | OPEN | 2026-10-23 |
| [18239](https://github.com/vllm-project/vllm-ascend/issues/18239) | [Bug][MRV2] Qwen3.8-27B + MTP: GDN metadata `KeyError 'language_model.model.layers.0.linear_attn'` during FULL_DECODE_ONLY capture | OPEN | 2026-10-10 |
| [13863](https://github.com/vllm-project/vllm-ascend/issues/13863) | [Bug]: Async MTP can reuse stale accepted-token counts after batch condense | OPEN | 2026-08-09 |
| [18619](https://github.com/vllm-project/vllm-ascend/issues/18619) | [Bug]: Qwen3.8-27B (qwen3_5 GDN hybrid) on Atlas A2 — P/D disaggregation + MTP + prefix caching intermittently produces garbled output | OPEN | 2026-10-13 |
| [17338](https://github.com/vllm-project/vllm-ascend/issues/17338) | [Bug]: Qwen3.8-27B-w8a8-310P (GDN hybrid) — `--enable-prefix-caching` + MTP simultaneously → token-level corruption | OPEN | 2026-09-23 |
| [16979](https://github.com/vllm-project/vllm-ascend/issues/16979) | [Bug]: Qwen3.8-27B (qwen3_5 hybrid GDN+full) MTP output quality collapse (310P3, v0.23.0) | OPEN | 2026-09-20 |
| [10495](https://github.com/vllm-project/vllm-ascend/issues/10495) | [Bug]: mtp acceptance rate dropped to zero after several rounds with 64k input | OPEN | 2026-06-15 |

### 5.5 Ascend: state-aware prefix caching (the other half of the problem)

A *large* fraction of Ascend activity is about hybrid prefix-cache boundaries, because a snapshot is only
reusable if it lands exactly on a Mamba block boundary — the same alignment requirement spec decode needs.

| # | Title | State | Created |
|---|---|---|---|
| [8791](https://github.com/vllm-project/vllm-ascend/issues/8791) | [RFC]: GDN all-mode prefix caching on Ascend | OPEN | 2026-04-29 |
| [8160](https://github.com/vllm-project/vllm-ascend/pull/8160) | [Feature] Add GatedDeltaNet all-mode APC | OPEN PR | 2026-04-11 |
| [16451](https://github.com/vllm-project/vllm-ascend/issues/16451) | [RFC]: Fine-grained Prefix Cache for Hybrid Mamba Models | OPEN | 2026-09-13 |
| [17352](https://github.com/vllm-project/vllm-ascend/issues/17352) | [Feature] Ascend NPU follow-up: application-directed Mamba prefix checkpoints (port of vllm#55697 series) | OPEN | 2026-09-23 |
| [17454](https://github.com/vllm-project/vllm-ascend/pull/17454) | [Feature][Mamba] Application-directed Mamba checkpoint coordination in the Ascend scheduler (spike for #17352) | OPEN PR | 2026-09-24 |
| [20001](https://github.com/vllm-project/vllm-ascend/issues/20001) | [RFC]: Integrate lookahead block hashes for MTP prefix caching and align hybrid checkpoint boundaries | OPEN | 2026-10-27 |
| [12091](https://github.com/vllm-project/vllm-ascend/pull/12091) | [BugFix] Prevent Mamba EAGLE lookup from expanding hybrid prefix hits | OPEN PR | 2026-07-15 |
| [18517](https://github.com/vllm-project/vllm-ascend/pull/18517) | [BugFix][KV Cache] Align prefix cache writes with effective EAGLE drop | OPEN PR | 2026-10-13 |
| [19666](https://github.com/vllm-project/vllm-ascend/pull/19666) | [BugFix][KV Cache] Restore standalone MTP replay boundaries | OPEN PR | 2026-10-23 |
| [19940](https://github.com/vllm-project/vllm-ascend/pull/19940) | [BugFix][KV Cache] Avoid false Mamba external-state headroom within the same block | OPEN PR | 2026-10-27 |
| [18582](https://github.com/vllm-project/vllm-ascend/pull/18582) | [BugFix][Mamba] Fix prefix-cache hit cliffs at exact PMU multiples | OPEN PR | 2026-10-13 |
| [19400](https://github.com/vllm-project/vllm-ascend/pull/19400) | [v0.30.0][BugFix][Ops] Build GDN recurrent state indices in metadata | OPEN PR | 2026-10-21 |
| [13372](https://github.com/vllm-project/vllm-ascend/pull/13372) | [BugFix][SpecDecode][Mamba] Fix DSpark slot mapping and mamba postprocess signature on hybrid targets | OPEN PR | 2026-08-03 |
| [19769](https://github.com/vllm-project/vllm-ascend/pull/19769) | [Feature][SketchSSM] Ascend NPU backend for Mamba-2 SketchSSM | OPEN PR | 2026-10-26 |

### 5.6 Ascend: other drafters (context for "what else is landing")

* [17805](https://github.com/vllm-project/vllm-ascend/pull/17805) — [Feature][SpecDecode] Add `mamba_attn_hybrid` (H-Spec) speculative decoding support on Ascend (OPEN, 2026-09-30). H-Spec is a *parallel* drafter built from a Mamba-2/attention/MLP recipe that reads verifier KV in place, i.e. it sidesteps the drafter-side state-cache cost.
* [19458](https://github.com/vllm-project/vllm-ascend/pull/19458) — Eagle-4 spec decode (OPEN, 2026-10-22).
* [18787](https://github.com/vllm-project/vllm-ascend/pull/18787) — Domino speculative decoding (OPEN, 2026-10-15).
* [19682](https://github.com/vllm-project/vllm-ascend/pull/19682) — DSpine speculative decoding (OPEN, 2026-10-23).
* [19306](https://github.com/vllm-project/vllm-ascend/pull/19306) — [Refactor][SpecDecode] Unify speculative decoding dispatch and KV slot mapping (OPEN, 2026-10-20).

---

## 6. Top 5 most relevant items (n+1 recurrent state)

| Rank | Repo | # | Title | State | Why |
|---|---|---|---|---|---|
| 1 | vllm | [54080](https://github.com/vllm-project/vllm/issues/54080) | [RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models | OPEN, 2026-08-27 | Only place where "N positions need N+1 full state snapshots per sequence per layer" is stated with real MiB numbers, plus a working implementation that replaces snapshots with WY/UT reconstruction. Direct ReplaySSM comparison. |
| 2 | vllm | [47572](https://github.com/vllm-project/vllm/issues/47572) | [RFC]: ReplaySSM: cache SSM inputs instead of state … (Mamba2 + GDN) | OPEN, 2026-07-03 | The canonical upstream answer: rollback becomes an O(1) ring-buffer pointer move, removes `(1 + num_spec)×` state. Its spec PRs were closed 2026-10-12 but the idea survived as RecoverSSM. |
| 3 | vllm | [58863](https://github.com/vllm-project/vllm/pull/58863) (+ [61309](https://github.com/vllm-project/vllm/pull/61309), [59983](https://github.com/vllm-project/vllm/pull/59983)) | [Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN … | OPEN, 2026-09-26 | The *live* GDN implementation: verify from read-only checkpoint, store `[V, K, gate]` replay records, `num_speculative_blocks = 0`. Already measured +8.6% @ K=5. |
| 4 | vllm-ascend | [19987](https://github.com/vllm-project/vllm-ascend/issues/19987) (+ [19733](https://github.com/vllm-project/vllm-ascend/issues/19733), roadmap [18827](https://github.com/vllm-project/vllm-ascend/issues/18827) item 10) | [RFC]: Support GDN RecoverSSM for Speculative Decoding on Ascend | OPEN, 2026-10-27 | Ascend's official plan to stop keeping one state per draft position; explicitly ports upstream #63045 and moves records out of KV pages. RFC only — no code yet. |
| 5 | vllm | [56531](https://github.com/vllm-project/vllm/pull/56531) | [Bugfix] Route every speculation-capable row through the speculative path so recurrent state stays consistent | **MERGED** 2026-10-05 | The best single written description of today's slot layout and why slot 0 ≠ slot k. Merged, so it defines current behaviour. |

Runners-up worth flagging: [62304](https://github.com/vllm-project/vllm/issues/62304) (state misses the last
accepted token's update — the failure mode of the snapshot scheme), [63045](https://github.com/vllm-project/vllm/pull/63045)
(first merged `num_speculative_blocks = 0`), [59117](https://github.com/vllm-project/vllm/issues/59117) (2.05x
concurrency from GDN+spec memory pressure).

---

## 7. SGLang — light pass (3–5 bullets)

Another agent covers SGLang in depth; these are only the items that touch n+1 state sizing.

* **[#37066](https://github.com/sgl-project/sglang/issues/37066)** (OPEN, 2026-08-29) — plain EAGLE/NEXTN no longer charges `D` to the slot pool: "`memory_pool.py`: `intermediate_ssm_state_cache` shape `(layers, spec_state_size + 1, D, ...)`". So SGLang has **both** a shared pool *and* a separate per-request `D`-sized intermediate buffer — i.e. the n+1 copies are outside the paged pool.
* **[#36889](https://github.com/sgl-project/sglang/issues/36889)** (OPEN, 2026-08-28) — verbatim: "each running request needs `1 + speculative_num_draft_tokens` mamba state slots", and the scheduler silently caps concurrency: "`max_running_requests is capped to 2 by the mamba state cache (max_mamba_cache_size=10, 5 state slots per request)`" → +75% throughput hidden behind one flag. Same `1 + D` arithmetic as vLLM, but with a cap message.
* **[#40706](https://github.com/sgl-project/sglang/issues/40706)** (OPEN, 2026-09-22) — mitigation on the same axis: `--mamba-ssm-dtype float8` halves `state_bytes` per slot ("state/slot = 153.9 MB at float32"). vLLM has no FP8 SSM state equivalent (cf. vLLM [#55196](https://github.com/vllm-project/vllm/issues/55196)).
* **[#43514](https://github.com/sgl-project/sglang/issues/43514)** (OPEN, 2026-10-10) — "[Bug] DFlash + Qwen3.8-27B: speculative decoding accept rate drops to 0 during serving" — the same class of snapshot-lifecycle corruption vLLM tracks in #53912/#62304.
* **[#41351](https://github.com/sgl-project/sglang/issues/41351)** (OPEN, 2026-09-26) — "[Bug] Potential hybrid GDN Radix-cache selected-logprob drift on repeated branch scoring" — prefix cache + GDN state reuse, SGLang's analogue of vLLM's align-mode issues.

No SGLang issue was found that proposes reconstructing the accepted state instead of snapshotting it
(no ReplaySSM/RecoverSSM/TreeWY analogue) — but that is a statement about what my queries returned, not a
guarantee (see §8).

---

## 8. Method, and queries that returned nothing

Tooling: `gh search issues/prs`, `gh api "search/code"`, `gh issue view`, `gh api repos/.../pulls/N`
(for authoritative `merged` flags), and `curl https://raw.githubusercontent.com/...` for source.
`web_fetch` was not used (broken per task brief).

Files read from `main` (raw, 2026-10-30-ish `main`):
`vllm/v1/worker/mamba_utils.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/v1/kv_cache_interface.py`, `vllm/v1/core/single_type_kv_cache_manager.py`,
`vllm/v1/attention/backends/{utils.py,gdn_attn.py}`,
`vllm/model_executor/layers/mamba/{abstract.py,mamba_utils.py}`,
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`vllm/config/{cache.py,speculative.py}`, `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`,
`docs/features/automatic_prefix_caching.md`, `vllm_ascend/ops/{gdn.py,gdn_attn_builder.py}`, plus
directory listings of `vllm/v1/spec_decode`, `vllm/v1/worker/gpu/model_states`,
`vllm/models/kimi_k3/nvidia/**`, `vllm_ascend/worker/v2/spec_decode`.

Notable negative results (queries returning **zero** items):

| Query | Result |
|---|---|
| `gh search prs --repo vllm-project/vllm "recurrent speculative verification"` | no results |
| `gh search issues --repo vllm-project/vllm "mamba state align"` | no results |
| `gh search issues --repo vllm-project/vllm "mamba speculative decoding"` | 1 result, irrelevant (RFC on `torch.uint64` device pointers) |
| `gh search issues --repo vllm-project/vllm-ascend "spec decode hybrid"` | no results |
| `gh search issues --repo sgl-project/sglang "linear attention spec decode"` | no results |
| `gh search issues --repo sgl-project/sglang "gdn spec"` | no results |
| `gh search issues --repo sgl-project/sglang "hybrid speculative decoding recurrent"` | no results |
| `gh search issues --repo sgl-project/sglang "mamba state speculative"` | no results |
| `gh api search/code q=num_speculative_blocks repo:vllm-project/vllm` | 31 hits, listed in §1 |

Caveats / open gaps a follow-up should close:

1. **No closing rationale** is available via the API for the 2026-10-12 mass-closure of
   #47576 / #48792 / #49847 / #49887. It is plausible (but *not* evidenced here) that RecoverSSM
   superseded ReplaySSM-spec; the only supporting signal is that #58863 describes RecoverSSM as
   "the RecoverSSM protocol that Kimi-K3 KDA already uses".
2. Whether `--use-replayssm` currently works for **Mamba2** speculative decode on `main` (ReplaySSM-spec was
   closed, and `vllm/config/cache.py` says "Mamba2 speculative decode requires FlashInfer") was not
   end-to-end verified — only the config/code presence was.
3. GitHub code search is rate-limited and flaky here; two code-search calls timed out. Directory listings and
   raw file fetches were used instead, so a symbol I did not think to grep for could still exist.
4. Ascend `main` was read at a single point in time; the RecoverSSM RFCs (#19733, #19987) are 2–3 days old at
   the time of this run and may already have draft PRs that do not yet mention "RecoverSSM" in the title.
