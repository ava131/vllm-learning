# Speculative / Tree Decoding for Hybrid Linear-Attention Models + Ascend NPU Constraints

**Compiled:** 2026-10-29 · **Scope:** broad literature sweep (2025–2026) beyond the seven already-held papers
(ReplaySSM, TreeWY, Bole, DeltaLog, DAMP, SpecLA, DASC) + a constraint analysis for implementing such a
scheme on Huawei Ascend NPUs.

**Method / honesty notes**

- Literature found via `export.arxiv.org` Atom API (~35 distinct queries over `all:` fields), arXiv
  `/html/<id>` full-text fetches, and `web_search`. Every item below was opened and read (abstract +
  mechanism sections); ids, titles and dates are copied from the API, not from memory.
- Ascend facts come from: the live `vllm-project/vllm-ascend` tree (via GitHub API, commit on `main` as of
  2026-10-29), the live `flashserve/flash-linear-attention-npu` tree, open/merged vllm-ascend and vllm PRs,
  and two CANN-ecosystem reference docs. Anything I could not open is labelled **[UNVERIFIED]**.
- Two "known seven" items are *also* being productized in vLLM upstream right now; that is reported in
  §1.6 because it changes what is worth re-implementing.

---

# PART 1 — Literature sweep

## 1.1 The design space, as it actually appears in the 2026 literature

The "n+1 states" problem is never called that in print. The field's vocabulary is
**"one state per draft position"**, **"per-node state materialization"**, **"extra states"**, or
**"state snapshot per draft slot"**. Five distinct attack families are visible, and they are *not*
interchangeable — they differ in whether verification is exact, whether a tree is supported, and what the
commit path costs.

| Family | Core move | Who | Exact? | Tree? | Per-draft HBM |
|---|---|---|---|---|---|
| **F1. Snapshot everything** (baseline) | one full recurrent state per draft slot | vLLM/SGLang default; `num_speculative_blocks` | yes | yes | `(K+1)·S` |
| **F2. Factor log + replay accepted prefix** | store per-token low-rank factors `(v_new, k̂, g)`; after sampling, replay only accepted tokens and write state once | ReplaySSM, SpecLA, Bole, vLLM RecoverSSM, Ascend PR #18765 | yes (bit-exact replay) | only chain-ish (needs per-path replay) | `≈ K·(2·D)` ≪ `S` |
| **F3. Tree-structured closed form (WY / tree-scan)** | fold the whole tree's update into one triangular solve so *outputs* for all nodes come from one pass; keep one small pseudo-value matrix, reconstruct the single accepted state | TreeWY, STree, GDN Tree-Scan | yes in exact arithmetic | **yes** | one matrix, not `N_nodes·S` |
| **F4. Never materialize state during verify** | compute draft outputs from buffered recent K/V (or from a read-only checkpoint) instead of a state per node | KVBuffer, ReaDy (reads only the committed state), DeltaLog (deferred merge) | yes | yes (F4 is tree-agnostic) | ~0 |
| **F5. Compress / quantize the states you do keep** | make each state cheaper rather than fewer | DAMP, DASC, STEPQuant | lossy | n/a | `(K+1)·S/c` |

Two structural observations worth carrying forward:

1. **F2/F3/F4 all assume the linear-attention layer is a *linear* recurrence in the state**, so a token's
   contribution is an outer product and a path's contribution is a product of transitions. For GDN/KDA the
   transition is `(I − β k kᵀ)·diag(α)` — *rank-1 plus diagonal* — which is exactly why a WY/triangular-solve
   form exists. Mamba-2 is the diagonal special case. RWKV-6/7 is diagonal decay + rank-1, same shape.
2. **Verification exactness and commit cost decouple.** Every F2–F4 paper accepts that the *outputs* must be
   exact but lets the *state write* be deferred to the commit step, after rejection sampling. This is the
   single most important architectural insight for a hardware port: it moves work off the verify hot loop.

---

## 1.2 Table of newly-found items (beyond the seven), most-recent first

Priority: **A** = directly load-bearing for the project · **B** = useful technique/support · **C** = context only.

| # | arXiv | Date | Title | Family | How it attacks the n+1-state problem | Pri |
|---|---|---|---|---|---|---|
| 1 | [2610.26660](https://arxiv.org/abs/2610.26660) | 2026-10-19 | Tandem: Serving Hybrid Linear-Attention GUI Agents on Mobile NPUs | F2+F4 | explicit: "retaining rollback states for speculative decoding can move prompt processing off the NPU's matrix unit"; builds rollback states from a short per-token tail instead | B |
| 2 | [2610.20503](https://arxiv.org/abs/2610.20503) | 2026-10-14 | ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention LLMs | F4 (drafter side) | sidesteps n+1 entirely: drafter reads **only committed** states, one fixed-size source, position-conditioned low-rank reads | A |
| 3 | [2609.38169](https://arxiv.org/abs/2609.38169) | 2026-09-29 | STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization | F5 | reduces per-state bytes (6-bit nominal, >5× state compression) so `(K+1)` snapshots hurt less | B |
| 4 | [2609.24197](https://arxiv.org/abs/2609.24197) | 2026-09-21 | H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache | drafter side | removes the drafter's own cache; Mamba modules initialised from last-token target hidden state, attention modules reuse target KV in place | A |
| 5 | [2609.23900](https://arxiv.org/abs/2609.23900) | 2026-09-20 | GDN Tree-Scan: Served Tree Verification for Recurrent-Hybrid LMs (v3 retitled *LumoTree: Path-Parallel Speculative Verification for Hybrid Language Models*) | F2+F3 | **branch-local GDN scan/replay** + accepted-chain-only state publication; rejected branch state made non-durable in the cache | **A+** |
| 6 | [2609.06872](https://arxiv.org/abs/2609.06872) | 2026-09-06 | Exact Record Omission in Delta Attention: A Transport Criterion, Its Cost, and a Replay Certificate | theory (F2) | gives the *exact* condition under which a saved state difference ("receipt") plus transport reconstructs a target state — i.e. when a cheap delta beats replay. Empirically the receipt fails and checkpoint replay is exact | B |
| 7 | [2607.21535](https://arxiv.org/abs/2607.21535) | 2026-07-23 | Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context | drafter side | not an n+1 fix, but the key economic argument: on hybrids, cheap verification exposes the *draft's* full-attention read; windows the draft's KV to a constant | A (economics) |
| 8 | [2605.22791](https://arxiv.org/abs/2605.22791) | 2026-05-21 | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | architecture | decouples erase/write ⇒ changes what the "rank-1 record" must store; matters if targeting next-gen GDN | C |
| 9 | [2605.21325](https://arxiv.org/abs/2605.21325) | 2026-05-20 | Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers | kernel (F3 enabler) | **benchmarks on NPUs**; direct vs iterative triangular inverse, fp32/low-precision numerics, up to a large speedup over SGLang's implementation, layer-level gains at full accuracy | **A+ (Ascend)** |
| 10 | [2605.19049](https://arxiv.org/abs/2605.19049) | 2026-05-18 | KVBuffer: IO-aware Serving for Linear Attention | F4 | **"verifies draft tokens in parallel and avoids storing temporary states"**; chunkwise decode from buffered K/V; +serving capacity at K=4 | **A+** |
| 11 | [2605.05219](https://arxiv.org/abs/2605.05219) | 2026-04-17 | Sparse Prefix Caching for Hybrid and Recurrent LLM Serving | F5-adjacent (state checkpoints) | checkpoint *placement* DP: store exact states at sparse positions, resume from deepest checkpoint + exact suffix recompute. Orthogonal to spec decode but shares the "state checkpoint + exact suffix replay" machinery | B |
| 12 | [2605.01106](https://arxiv.org/abs/2605.01106) | 2026-05-01 | Component-Aware Self-Speculative Decoding in Hybrid Language Models | drafter side | isolates the SSM/linear-attn subgraph as a zero-cost internal draft; **negative result**: parallel hybrids (Falcon-H1) accept, sequential hybrids (Qwen3.5) don't (LayerSkip beats it) | B (important negative) |
| 13 | [2603.09555](https://arxiv.org/abs/2603.09555) | 2026-03-10 | Compiler-First State Space Duality and Portable O(1) Autoregressive Caching | compiler | SSD recurrence expressed in plain JAX, fused by XLA, cache as a PyTree; reaches roofline on TPU + L40S with no custom kernel. Template for "let the compiler tile it" | B |
| 14 | [2509.19873](https://arxiv.org/abs/2509.19873) | 2025-09-24 | SpecMamba: Accelerating Mamba Inference on FPGA with Speculative Decoding | F2 (hardware) | names the three obstacles explicitly: hidden-state backtracking, tree-verification incompatibility, draft/target workload mismatch; solves in an FPGA dataflow | B |
| 15 | [2506.01206](https://arxiv.org/abs/2506.01206) | 2025-06-01 | Mamba Drafters for Speculative Decoding | drafter side | Mamba as an *external* drafter (linear complexity draft, cross-model), + test-time tree search for candidates | B |
| 16 | [2505.14969](https://arxiv.org/abs/2505.14969) | 2025-05-20 | STree: Speculative Tree Decoding for Hybrid State-Space Models | F3 (origin) | **the origin of the closed form**: accumulates state-transition matrices along the tree, computes tree outputs in one packed pass, and *does not emit a state at all* — recomputes the correct state by activation replay after rejection | A |
| 17 | [2606.00601](https://arxiv.org/abs/2606.00601) | 2026-05-30 | ScanWeaver: Compiler-Driven Parallelization of Affine Recurrences via Associative Scan Lowering | compiler | associative-scan lowering as a compiler pass — relevant if the Ascend port wants scan-based rather than chunked verification | C |
| 18 | [2601.17768](https://arxiv.org/abs/2601.17768) | 2026-01-25 | LLM-42: Enabling Determinism in LLM Inference with Verified Speculation | systems | determinism/verification framing, useful for arguing bit-exactness of replay | C |

Adjacent (found, read abstract, **not** recommended as primary reading — listed so the sweep is honest):
prefix-cache-for-hybrid cluster `2609.33477` *Just Let Linear States Forget the Distant Past: Prefix Caching
via Suffix Replay*, `2608.30310` *Tail-Replay*, `2607.01299` *HYPIC*, `2610.18143` *RadixFlow* — these all
reuse the "state checkpoint + suffix replay" primitive, which is the same primitive as F2 commit; `2605.16640`
*Provably Shorter Scratchpads in Hybrid DeltaNet-Attention Decoders* (expressivity, not serving);
`2603.05931` persistent-state FPGA accelerator; `2607.25852` AngelSpec; `2610.21436` HyCo.

---

## 1.3 Per-item bullets (idea, and specifically the n+1-state move)

### 1. GDN Tree-Scan / LumoTree — [2609.23900](https://arxiv.org/abs/2609.23900) (2026-09-20)
- **Idea.** A *served* tree verifier for GDN hybrids inside vLLM. Four pieces: fork of FlashAttention-2 with
  a tree bias (attention ancestry), **branch-local GDN scan/replay**, device-side multidraft commitment, and
  **accepted-chain-only state publication** (rejected branch state is made non-durable rather than deleted).
- **n+1 attack.** Reframes the requirement: a tree row needs *two* path-local facts — attention ancestry
  **and** recurrent state lineage. Sibling rows must inherit the *root* state, not the previously packed row's
  state. They do not keep `N_nodes` states: they scan branch-locally during verify and **replay only the
  accepted chain** to publish state.
- **Numbers.** Qwen3.6-27B-FP8, B=1, T=0.6, SWE/Codex: six-node root-branch tree = 23.88 tok/s vs 18.80 for
  native 5-step MTP (+27.0% token-weighted decode throughput); committed tokens/event 4.11 → 4.82 (+17.2%) at
  near-native verify-forward time. Per-request-equal view only +4.0%.
- **Honest caveats (the authors state them).** Equivalence is "recurrent-oracle p-rescore closure within the
  observed native flip floor" (13.09–13.28% vs 12.90% native), **not** a distribution-distance proof; B=4,
  bootstrap, and TV-distance are open. Single author, no affiliation, one checkpoint.
- **Why it matters here.** It is the only paper that describes the *serving seams* (state publication,
  non-durable branch state, rejected-branch cleanup) rather than just the algebra. That seam list is exactly
  the checklist an Ascend port needs.

### 2. KVBuffer — [2605.19049](https://arxiv.org/abs/2605.19049) (2026-05-18)
- **Idea.** Buffer recent K/V and compute linear-attention outputs *from the buffer* instead of round-tripping
  the dense state every step. Decode becomes chunkwise: defer state updates and apply them in batch.
- **n+1 attack.** The strongest of all: **during spec verify, draft tokens are verified in parallel straight
  from buffered K/V and no temporary state is stored at all**. For short contexts it never creates the state.
  n+1 becomes 0.
- **Numbers.** SGLang + Qwen3-Next: −45.17% linear-attention decode latency; maximum concurrent requests up
  (reported for K=4 draft verification).
- **Caveat.** Buffer length is a hyperparameter; when the spec window exceeds the buffer you fall back to
  state. But for K=4–8 drafts this is comfortably inside any reasonable buffer.

### 3. H-Spec — [2609.24197](https://arxiv.org/abs/2609.24197) (2026-09-21)
- **Idea.** Hybrid Mamba-attention **parallel drafter** that needs *no drafter-side KV cache*: attention modules
  reuse target KVs in place; Mamba modules are initialised from the projected last-token target hidden state.
  Because Mamba's scan is parallel, block-parallel drafting survives.
- **n+1 attack.** Drafter-side, not verifier-side — it removes the drafter's cache, which is the other half of
  the memory bill on hybrids. Complementary to any F2/F3 verifier.
- **Numbers.** +5.0–13.3% mean accepted length, +5.3–12.6% batch-1 ITL speedup over best baseline on three targets.
- **Already on Ascend:** vllm-ascend PR [#17805](https://github.com/vllm-project/vllm-ascend/pull/17805)
  registers `method: "mamba_attn_hybrid"` under Model Runner V2, subclasses `AscendDSparkSpeculator`, and
  reports an e2e run on a **910B3** with Qwen3-4B + official H-Spec drafter.

### 4. ReaDy — [2610.20503](https://arxiv.org/abs/2610.20503) (2026-10-14)
- **Idea.** Give each draft position a learned query that reads the target's **committed** recurrent states, and
  inject the readout residually into the drafter's last layer (rank-8 query bottleneck, rank-8 value
  compression, fused gather + fused prep/pack).
- **n+1 attack.** Avoids it by construction: all draft positions read *the same* committed state, so the
  context source is one fixed-size tensor, independent of prefix length and of block size. Their theory
  (Lemma in §2) says a new query can extract information not in the existing readouts unless the exposed query
  span has full row rank — a nice justification for "one state, many queries".
- **Numbers.** Qwen3.8-27B on one B200: 3.16× over AR, +7.97% TPS and +8.50% acceptance length over DSpark.
  Also reports operator-level engineering: fused gather cut state-gather 16.388 → 4.622 µs (−71.8%).
- **Relevance.** The −71.8% on a *state gather* is a warning: on a bandwidth-starved part like Ascend, the
  gather/scatter of committed states is a first-class cost, not an afterthought (see §2.6).

### 5. Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers — [2605.21325](https://arxiv.org/abs/2605.21325) (2026-05-20)
- **Idea.** Systematic treatment of `inv(I − tril(diag(β) K Kᵀ))` for GDN/KDA: direct vs iterative algorithms,
  chosen for matrix-product richness (so tensor cores / Cube units are usable), with a heavy focus on
  numerical stability in low precision, plus **NPU benchmarks**.
- **Why it is the most Ascend-relevant paper in this list.** It reports **NPU** speedups over SGLang's
  triangular inversion and shows the whole GDN layer improving at unchanged end-to-end accuracy, and it
  explicitly ties inversion error accumulation to accuracy loss. If you port one kernel, port this one.
- **Caveat.** I read the abstract/intro and figure captions, not the full algorithm table; the exact NPU SKU
  and the numeric speedup factor did not survive the HTML→text conversion (math was stripped). Treat the
  multiplier as **[UNVERIFIED]** until read from the PDF.

### 6. STree — [2505.14969](https://arxiv.org/abs/2505.14969) (2025-05-20)
- **Idea.** The origin of the tree closed form for SSM/hybrid. Pack the draft tree into one sequence with a
  tree mask; accumulate the state-transition matrices `Ā` along tree edges so the packed pass yields every
  node's output. "When `M` is a lower-triangular causal mask, our method is the same as Mamba-2 chunk scan with
  a non-zero initial state."
- **n+1 attack.** Two moves: (a) the tree scan produces **no state output at all** — explicitly, "we only
  generate the output for the input sequence, but not the state after the input", because the end-of-sequence
  state is wrong unless everything is accepted; (b) after rejection, recompute the correct state by
  **activation replay**. So they keep *zero* per-node states and pay one short replay.
- **Numbers.** vs unrolled FSS on Mamba2-2.7B: tree forward grows 6→9 ms from 15→63 nodes while unrolled grows
  14→82 ms; tree+spec beats vanilla SSM spec on MT-Bench/HumanEval/GSM8K; advantage grows at T=1 (lower
  acceptance). Unrolled baselines OOM at 24 GB.
- **Why still worth reading.** It is the cleanest statement of the *cost model*: the acceptance-length ratio
  needed for tree decode to pay is derived as a function of tree size and the model-call intercept/slope.

### 7. STEPQuant — [2609.38169](https://arxiv.org/abs/2609.38169) (2026-09-29)
- **Idea.** Recurrent-state PTQ for delta-rule states along two axes: **temporal** (long-lived memory persists
  errors across many steps) and **spatial** (key rows differ in impact; magnitudes vary along rows *and*
  columns). Allocates precision by error magnitude × memory lifetime; jointly fits key-row and value-column
  scales.
- **n+1 attack.** F5: doesn't reduce the count, reduces the bytes. Nominal 6-bit matches FP32-state accuracy
  on Qwen3.8-27B and Kimi-Linear-48B-A3B; >5× state compression, up to −68.7% total serving memory, in SGLang.
- **Relevance.** On Ascend, where HBM is 64 GB @ ~1.6–1.85 TB/s (§2.1), a 5× state compression is a bigger
  lever than a 20% kernel win. Note the *combination* F2×F5 is unexplored: nobody has quantized the F2 factor
  log (which is only `K·2D` fp32 — already tiny, so probably not worth it) — but F5 on the *committed* state
  and F2 on the *draft* states is the obvious composition.

### 8. Sparse Prefix Caching — [2605.05219](https://arxiv.org/abs/2605.05219) (2026-04-17)
- **Idea.** Store exact recurrent states at a *sparse* set of checkpoint positions; on a hit, resume from the
  deepest checkpoint and recompute the suffix **exactly**. Checkpoint placement is an exact DP over a
  distribution of overlap depths.
- **n+1 attack.** Not a spec-decode paper, but it formalises the "checkpoint + exact suffix replay" cost model
  that F2's commit step is a special case of (`suffix = accepted tokens`). Worth stealing the DP if you ever
  have to choose *where* to place rollback checkpoints along a long generation.

### 9. Tandem — [2610.26660](https://arxiv.org/abs/2610.26660) (2026-10-19)
- **Idea.** Runtime for hybrid linear-attention GUI agents on **mobile NPUs** (Snapdragon 8 Elite, llama.cpp).
  Saves recurrent state where consecutive prompts diverge; produces **rollback states through a short per-token
  tail** instead of long replay; a verification kernel reads each weight block **once for all proposed tokens**.
- **n+1 attack.** The rollback-state-on-a-short-tail trick, plus the explicit warning that *retaining rollback
  states can push prompt processing off the NPU's matrix unit* and cost more than the decode saving on short
  outputs.
- **Why include it.** It is the only paper in the sweep that reasons about speculative state management on a
  **non-NVIDIA NPU** with a Cube/Matrix-unit-vs-vector split — structurally the same argument that governs
  Ascend, where the Cube:Vector ratio is 1:2 and the vector unit has 192 KB of UB.
- **Caveat.** Snapdragon Hexagon, not Ascend. Analogical, not transferable numbers.

### 10. Windowed-MTP — [2607.21535](https://arxiv.org/abs/2607.21535) (2026-07-23)
- **Idea.** Apply a StreamingLLM-style window + attention sink **to the MTP draft head's attention only**,
  leaving full-attention verification intact. Training-free, and *lossless by construction* because the target
  still decides every accepted token.
- **Why it matters for hybrids.** The paper's core claim is that the shift to hybrid/linear-attention targets
  **makes the draft the bottleneck**: cheaper verification leaves the draft's full-attention KV read exposed, so
  at 1 M context it dominates. Numbers: +28% to +44% per-decode-step improvement over shipping native MTP at
  1 M on Qwen GDN-MoE 35B/122B and a Mamba2-hybrid NoPE 120B (SGLang, single GPU); draft KV is 7.7–11% of
  total KV at 1 M and is reclaimed with a ring buffer.
- **n+1 relevance.** None directly — but it says the *right* thing about where the win is: on a hybrid target the
  verifier is cheap, so an expensive drafter eats the gain. Any Ascend design must budget drafter cost, not just
  verify cost.

### 11. Component-Aware Self-Speculative Decoding — [2605.01106](https://arxiv.org/abs/2605.01106) (2026-05-01)
- **Idea.** Use the SSM/linear-attention subgraph itself as a zero-cost internal draft. Evaluated on Falcon-H1
  (parallel: Mamba-2 + attention *per layer*) vs Qwen3.5 (sequential: interleaved) vs a Qwen2.5 control.
- **Finding (negative, and valuable).** Parallel hybrids accept (~high rate at draft length 4); sequential
  hybrids barely accept — an ~8× gap — and **generic LayerSkip beats component-aware self-speculation on
  sequential hybrids**. Scale-invariant (0.5B reproduces at 3B). They claim a perplexity-degradation ratio
  predicts viability without running spec decode.
- **n+1 relevance.** None, but it kills an idea cheaply: "just draft with the linear-attention layers" works for
  Falcon-H1-style and not for Qwen3.5/Kimi-Linear-style stacks. Do not build that.

### 12. Exact Record Omission — [2609.06872](https://arxiv.org/abs/2609.06872) (2026-09-06)
- **Idea.** Save the state difference right after a record (a "receipt"), carry it through later updates,
  subtract at the end; compare against the state built with the record never stored. Gives an **exact transport
  criterion**: the receipt reaches the never-stored state iff the extra differences created by later updates
  cancel after transport.
- **Empirical.** On Kimi Linear 48B (KDA + MLA) and a Qwen cohort, naive transport **fails** beyond the numerical
  error of matched bf16/8-bit controls; checkpoint replay matches the never-stored state exactly on every
  audited array and on the full next-token score vector across all 80 contexts. Replay cost is proportional to
  the *surviving suffix*.
- **n+1 relevance.** This is the theoretical reason F2 must be *replay*, not *subtract*: you cannot cheaply
  "undo" a rejected draft by subtracting its contribution, because later (accepted) tokens were computed from a
  state that already contained the rejected one. Any delta-log scheme must replay forward from the last accepted
  prefix, which is exactly what ReplaySSM/RecoverSSM/Bole/TreeWY commit do. Good citation for why F1→F2 is safe
  but "negative delta" is not.

### 13. Mamba Drafters / SpecMamba (2025, foundational)
- **Mamba Drafters** ([2506.01206](https://arxiv.org/abs/2506.01206)): Mamba as an *external* drafter exploiting
  linear complexity — faster drafting, lower memory, cross-target flexibility, plus test-time **tree search** for
  candidate generation. Comparable to self-speculation while using less memory.
- **SpecMamba** ([2509.19873](https://arxiv.org/abs/2509.19873)): the clearest statement of the three obstacles
  for SSM speculation — **hidden-state backtracking, tree-verification incompatibility, draft/target workload
  mismatch** — solved with an FPGA dataflow. Read it for the problem framing even if the hardware is irrelevant.

### 14. Compiler-First SSD — [2603.09555](https://arxiv.org/abs/2603.09555) (2026-03-10)
- **Idea.** SSD exposes four compiler-friendly properties (diagonal state matrix, chunkable recurrence,
  einsum-dominated compute, static control flow) which are *sufficient* for XLA fusion/tiling to reach roofline
  with **no custom kernels**; AR cache as a JAX PyTree. Matches Triton reference within tolerance, length-
  independent decode on five scales.
- **Relevance to Ascend.** This is the strongest existing argument for the "write it in a fused-graph compiler
  and let the backend tile it" route — which on Ascend means TileLang-Ascend or torch.compile/Inductor rather
  than hand-written AscendC. It also tells you the required properties to preserve (static control flow,
  einsum-shaped compute) if you want that route to work.

---

## 1.4 What the sweep did **not** find (real gaps — these are opportunities)

1. **No paper states or proves a lower bound on the number of states.** The "n+1" figure is universally treated
   as an implementation convention, not a requirement, and every F2–F4 paper implicitly proves it isn't one.
   There is no formal statement like "k states suffice for a tree of width w". Bole/TreeWY/GDN Tree-Scan give
   *constructions* (1 checkpoint + per-token factors = 1 extra state), not bounds.
2. **No RWKV-6/7 speculation paper.** Zero hits for `all:"RWKV" AND all:speculative` and for RWKV+verification.
   DeltaLog covers RWKV-6 for state *maintenance*, but nobody has published tree verification for RWKV-7's
   diagonal-plus-rank-1 (the "delta rule with in-context learning rate") form. This is a genuine hole.
3. **No KDA-specific tree verification paper.** KDA appears only via Kimi Linear in *record omission*
   ([2609.06872](https://arxiv.org/abs/2609.06872)) and in quantization ([STEPQuant](https://arxiv.org/abs/2609.38169)).
   KDA's *channel-wise* decay (vs GDN's head-wise) makes the WY/triangular form different — the per-channel
   decay does not commute with the rank-1 update the way a scalar-per-head decay does, so a KDA tree closed form
   is genuinely harder than GDN's. TreeWY/Bole/GDN Tree-Scan all say "GDN"; Kimi Linear (KDA) is analysed but
   never tree-verified in print.
4. **No Ascend/domestic-NPU spec-decode-for-linear-attention paper.** Tandem is Snapdragon. Nothing on
   Ascend, Cambricon, or MetaX. The Ascend work is all in PRs (§2), not papers.
5. **No treatment of the interaction between spec decode and *chunked prefill / PD disaggregation*** for hybrid
   state. vllm-ascend has Mooncake layerwise state transfer PRs, but no paper.
6. **MTP-for-hybrid is under-theorised.** Qwen3-Next/Qwen3.5/GLM-5.x/MiniMax-M3 all ship MTP or EAGLE-3 heads
   (evidenced by the serving configs in §2.4), but there is no paper on how an MTP head should be *trained*
   against a linear-attention verifier whose acceptance boundary differs from a Transformer's.

---

## 1.5 Production track: ReplaySSM/RecoverSSM is landing in vLLM right now (read this before re-implementing)

This is arguably the most actionable finding of the whole sweep. **ReplaySSM is not just a paper — it is being
merged into vLLM `main` as the `--use-replayssm` flag, with a "RecoverSSM" protocol, and it already covers GDN,
KDA and Mamba.** All PRs below are **open** as of 2026-10-29 (none merged), so treat as in-flight.

- [#58863](https://github.com/vllm-project/vllm/pull/58863) *[Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and
  the PLE short conv* — the canonical description: "Speculative decoding on Gated DeltaNet today **snapshots the
  full recurrent state after every draft position, and reserves one Mamba block per draft position**. This PR
  brings the RecoverSSM protocol that **Kimi-K3 KDA already uses** to the Qwen GDN layers." Verify runs from a
  read-only checkpoint and stores a small per-token replay record `[correction (V), normalized key (K), gate (1)]`
  in fp32; commit replays only accepted tokens and writes the state once; `num_speculative_blocks = 0`.
- [#61309](https://github.com/vllm-project/vllm/pull/61309) RecoverSSM for Qwen3.5 / Qwen3-Next GDN.
- [#61566](https://github.com/vllm-project/vllm/pull/61566) KDA RecoverSSM for GLM-5.3-Flash.
- [#62004](https://github.com/vllm-project/vllm/pull/62004) ReplaySSM for **standard (non-spec) decode** on GDN —
  i.e. ReplaySSM is being generalised beyond speculation.
- [#61124](https://github.com/vllm-project/vllm/pull/61124) *SketchSSM* — a different, sketching-based decode.
- [#59982](https://github.com/vllm-project/vllm/pull/59982) (XPU), [#59983](https://github.com/vllm-project/vllm/pull/59983)
  (pipeline parallelism), [#61017](https://github.com/vllm-project/vllm/pull/61017) (all GDN KV groups in one
  launch), [#56082](https://github.com/vllm-project/vllm/pull/56082) (Kimi-K3 commit overhead),
  [#56466](https://github.com/vllm-project/vllm/pull/56466) (prefix caching for GDN spec decode),
  [#52962](https://github.com/vllm-project/vllm/pull/52962) (quantized ReplaySSM checkpoints),
  [#55688](https://github.com/vllm-project/vllm/pull/55688) (unify FT/MTP/prefix-caching lifecycle),
  [#50021](https://github.com/vllm-project/vllm/pull/50021) (bound accepted-token state lookups),
  [#57128](https://github.com/vllm-project/vllm/pull/57128) (MambaManager prefix-cache corruption with MTP/EAGLE).
- Known cost of the approach, stated in #58863: the replay record makes the unified page **~6% larger**, which
  can make a pool that just fits `max_num_seqs` under native MTP fall short; and prefix reuse counts whole
  blocks so a repeated prompt can reuse up to one block less. **This is exactly the trade to re-measure on
  Ascend**, where HBM is the scarce resource.

**Two enablers that matter more than any single paper**

- [#62962](https://github.com/vllm-project/vllm/pull/62962) *Add a registry for out-of-tree speculative methods
  and a **linear-attention verify extension*** — introduces `LinearAttnVerifyExtension`: GDN and Mamba-2 layers
  hand their spec-decode rows to an *out-of-tree* verifier as typed zero-copy views (projections, states, gate
  parameters, indices, accepted counts, output buffer), so **a tree method can verify in the linear-attention
  layers without forking vLLM**. Also opens `SpeculativeConfig.method` and the MRV2 speculator if/elif chain to
  plugin registration. Touches `vllm/model_executor/layers/mamba/verify_extension.py`,
  `vllm/v1/attention/backends/gdn_attn.py`, `mamba2_attn.py`, `vllm/v1/spec_decode/registry.py`.
- [#62963](https://github.com/vllm-project/vllm/pull/62963) adds the MRV2 calls a non-chain verifier needs
  (custom rejection sampler, verify-input and accepted-path hooks).

**Ascend mirrors of that EAGLE-4/registry programme** (see §2.5): vllm-ascend
[#19603](https://github.com/vllm-project/vllm-ascend/pull/19603) (linear-attention verify extension),
[#19604](https://github.com/vllm-project/vllm-ascend/pull/19604) (speculator hooks for **non-chain verification**
and draft suspension), [#19600](https://github.com/vllm-project/vllm-ascend/pull/19600),
[#19602](https://github.com/vllm-project/vllm-ascend/pull/19602),
[#19606](https://github.com/vllm-project/vllm-ascend/pull/19606), tracked in RFC issue
[#19406](https://github.com/vllm-project/vllm-ascend/issues/19406).

**Implication.** Building a bespoke Ascend tree-verifier is now a *plugin*, not a fork. That dramatically
changes the cost of the project and should be the assumed integration path.

---

# PART 2 — Ascend NPU implementation constraints

## 2.1 Hardware baseline (per-chip, per-AI-Core)

Sources: [CANN-ecosystem arch facts (dav-2201 / dav-3510)](https://raw.gitcode.com/ascendc-agent/ascendc-skills/raw/980be3b0d7088f22fa4754908048e43c63f156c6/skills/sample/asc-perf-tuning/references/arch-facts.md),
[昇腾 910B/910C/950 硬件规格速查表](https://ai6s.net/6aa8af828f401d3626604ea0.html).

| | **A2 / 910B (dav-2201)** | **A3 / 910_93 (dav-2201)** | **A5 / 950PR (dav-3510)** |
|---|---|---|---|
| Cube / Vector cores | 24 AIC / 48 AIV (910B2 typical; B3/B4 cut to 20) · 1 AIC : 2 AIV | same family, `ASCEND910B` runtime SoC incl. `910_93` | 28–32 AIC / 56–64 AIV |
| Clock | 1.8 GHz | — | 1.65 GHz |
| BF16/FP16 Cube | 353 TFLOPS (910B2) | — | 378–432 TFLOPS |
| HBM capacity | **64 GB** (910B4 cut to 32 GB) | same family; exact per-package figure **[UNVERIFIED]** | 112 GB (PCIe) / 128 GB (Server) |
| HBM bandwidth | **1.6 TB/s nominal, ~1.85 TB/s measured peak** | ~same class | 1.4–1.6 TB/s |
| **UB per AIV** | **192 KB** (16 bank groups × 3 banks × 4 KB; 2 slots ≈ 64 KB each) | 192 KB | **256 KB physical / 248 KB usable** (8 groups × 2 banks × 16 KB) |
| L1 | 512 KB | 512 KB | 512 KB |
| L0A / L0B / L0C | 64 / 64 / 128 KB | same | 64 / 64 / **256** KB |
| L2 | ~192 MB @ ~7 TB/s | ~192 MB | 112–128 MB |
| Notable | physical AIC/AIV split; `MIX` blockDim must equal physical core groups | same arch ⇒ same Triton/AscendC constraints as A2 | regbase/SIMT entry; `SSBuffer` 256 KB; new AIC↔AIV mailbox |

**Direct consequences for this problem**

- **UB is 192 KB, i.e. ~48 K fp32 elements.** A GDN/KDA chunk of `BT=64`, `K=V=128`, `H` heads: a single
  `64×64` fp32 tile is 16 KB; a `64×128` fp32 tile is 32 KB. You get roughly 6–10 live fp32 tiles before you
  spill. Any kernel that holds `A` (the `64×64` in-block matrix) + `K` + `V` + the running state tile for
  several heads will not fit. **This is why the Ascend FLA kernels are written with explicit `LARGE_BLOCK_T`
  row-blocking and `extract_slice`/`insert_slice` sub-tile juggling** (see §2.3).
- **HBM at 1.6–1.85 TB/s with 64 GB** vs an H100's 3.35 TB/s / 80 GB. The state-update tax that DeltaLog
  attacks (write back `H·K·V` per token) is *relatively worse* on Ascend than on NVIDIA, which makes F2/F4
  (defer the state write) **more** valuable here, not less. Conversely, DASC/STEPQuant-style compression is
  worth more here too.
- **512 B transfer alignment**: the CANN reference notes up to **30% bandwidth loss** going from 512 B-aligned to
  32 B-aligned `DataCopy`. State tensors are `[H, V, K]` with `V=K=128` bf16 ⇒ one head's state is 32 KB (nicely
  aligned); but a *ragged* or strided gather of individual state rows is not. See §2.6.

## 2.2 The GDN/KDA op stack that already exists on Ascend

There are **three** layers, and knowing which one you are looking at matters.

**(a) `torch_npu` built-ins** — `torch_npu.npu_chunk_gated_delta_rule`, `torch_npu.npu_recurrent_gated_delta_rule`.
Referenced directly in `vllm_ascend/ops/gdn.py` (`_probe_fused_chunk`, `_chunk_gated_delta_rule_fused`).
**bfloat16 only** — the code comments that both fused prefill kernels (this one *and* `fla_npu`'s) only support
bf16, and fp16 models (e.g. ColQwen3.5) fall back to the Triton pipeline to avoid an aclnn dtype error.

**(b) `fla_npu` — a third-party AscendC linear-attention op library.** Repo:
[`flashserve/flash-linear-attention-npu`](https://github.com/flashserve/flash-linear-attention-npu), described as
Tianjin-University-led, "对标 Flash-Linear-Attention", prebuilt wheels for A2/A3/A5, CANN ≥ 8.5.2 (A2/A3) or
≥ 9.0.0 (A5), SoC knobs `ascend910b` / `ascend910_93` / `ascend950`.
Its AscendC op set (extracted from `op_kernel/` dirs and `npu_custom.yaml`) is **exactly the chunked GDN/KDA
pipeline**:

```
GDN:  chunk_gated_delta_rule_fwd_prepare   chunk_scaled_dot_kkt      solve_tri
      recompute_w_u_fwd / prepare_wy_repr_bwd{,_da,_full}
      chunk_gated_delta_rule_fwd_h  chunk_fwd_h  chunk_fwd_o
      chunk_local_cumsum  causal_conv1d  recurrent_gated_delta_rule
      chunk_gated_delta_rule_bwd_{dhu,finalize}  chunk_bwd_dqkwg  chunk_bwd_dv_local  chunk_gdn_bwd_intra
KDA:  chunk_kda_fwd_prepare  chunk_kda_fwd  chunk_kda_fwd_finalize  recurrent_kda  kda_gate_cumsum
      chunk_kda_bwd_prepare/bwd_recompute/bwd_intra/bwd/bwd_finalize  kda_gate_bwd_post[_varlen]
```

News log: 2026/09 added KDA fwd+bwd and "GDN 大融合 (fused forward / backward finalize)"; 2026/06 v26.6.0 wheels.
**This is the single most important fact for Part 2:** `solve_tri` (triangular inverse), the WY construction
(`recompute_w_u_fwd`, `prepare_wy_repr_*`), `chunk_local_cumsum` (decay cumprod), and the chunk fwd passes
**already exist as AscendC kernels**. You are not writing these from scratch.

**(c) `vllm-ascend`'s own kernels.** Two parallel implementations of the same math:
- **AscendC, in-tree, for the two hot ops**:
  `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/` (the sequential state pass) and `csrc/moe/chunk_fwd_o_vllm/`
  (the output pass), each with `op_host/` tiling + `op_kernel/{arch20,arch22,arch35}/` variants and
  `op_host/op_api/aclnn_*.{h,cpp}` aclnn wrappers. SoC config registration in
  `op_host/chunk_gated_delta_rule_fwd_h_vllm_def.cpp` lists **`ascend910b` and `ascend910_93`**.
  Also `csrc/attention/recurrent_gated_delta_rule_v310/` for 310P, and KDA-side AscendC ops
  `csrc/attention/kda_gate_cumsum_vllm/` and `csrc/attention/kda_layout_swap12/`.
  `csrc/attention/chunk_kda_fwd/` is **docs-only** (README + `docs/api.md` + `docs/design.md`, no `op_kernel/`)
  and specifies a single fused L0 kernel covering Gate → Prepare → PostWu → FwdH → Finalize
  (with `gk_i = cumsum(gate)_i / ln 2` so `exp2(gk)` is bound to the natural-log gate) — a design contract, not
  yet a shipped kernel.
- **Triton-Ascend, a full FLA port** — `vllm_ascend/ops/triton/fla/`: `chunk.py`, `chunk_delta_h.py`,
  `chunk_delta_hupdate.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`, **`solve_tril.py`**, **`wy_fast.py`**,
  **`cumsum.py`**, `l2norm.py`, `layernorm_guard.py`, `fused_qkvzba_split_reshape.py`, `utils.py`.
  And a **KDA** port — `vllm_ascend/ops/triton/kda/`: `kda.py`, `chunk_delta_h.py`, `solve_tril.py`, `cumsum.py`,
  `fused_recurrent_kda.py`, `recoverssm.py`, `conv_state.py`, `gate.py`, `l2norm.py`, `output_writeback.py`.

Dispatch order in `vllm_ascend/ops/gdn.py` (prefill): `fla_npu` fused op (if hardware capability
`FLA_GDN_PREFILL` and PCP world size 1) → `torch_npu.npu_chunk_gated_delta_rule` (if
`_probe_fused_chunk()`) → Triton pipeline. Decode: `recurrent_gated_delta_rule` (switchable backend
`ascend_gdn_decode_backend: fla_npu | native`, per PR
[#17393](https://github.com/vllm-project/vllm-ascend/pull/17393), which measured **538 µs → 148 µs** average
recurrent-decode latency for the fla_npu path across 50 scenarios).

## 2.3 So: does the triangular-inverse / WY / cumprod machinery exist? Yes — with caveats

Reading the actual Ascend Triton kernels answers this precisely.

- **`solve_tril` — yes, implemented, formula-shape is portable.** `vllm_ascend/ops/triton/fla/solve_tril.py`
  contains `solve_tril_16x16_kernel` (in-block forward substitution) + `merge_16x16_to_32x32_inverse_kernel`
  (block merge). The 16×16 inverse is a **serial loop over 16 rows** doing
  `b_a = b_a + sum_j (b_a[:,None,:] * b_A)` — i.e. 16 dependent vector ops — then the 32×32 merge is two
  `tl.dot(..., input_precision="ieee")` products (`Ai_21 = -Ai_22 · A_21 · Ai_11`). Recursion continues up to
  `BT=64`. **Ascend-specific deviations from the upstream GPU kernel:** the code uses `extract_slice` /
  `insert_slice` helpers (from `vllm_ascend/ops/triton/triton_utils.py`) instead of `tl.gather`-style indexing,
  and it packs `N_BLOCKS` 16×16 blocks into one `(N_BLOCKS*16, 16)` operand "to avoid for loops UB oom" — a
  *direct, in-code* statement that the UB is the binding constraint.
- **Tuning knob is Ascend-specific and lives in the caller:** `vllm_ascend/ops/gdn_attn_builder.py` sets
  `_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2` with the comment "Keep this aligned with solve_tril.LARGE_BLOCK_T",
  and `NTASKS = 2` inside the kernel splits that into two 608-row tasks. So each program handles **1216 rows**
  of the chunk-local matrix per launch. This is a hand-tuned number for Ascend, not an upstream default.
- **`cumsum` — yes, `tl.cumsum` works** and is used as the decay cumprod
  (`vllm_ascend/ops/triton/fla/cumsum.py`, `chunk_local_cumsum_scalar_kernel`, with `reverse=` supported). The
  working-set heuristic is explicit: `OPTIM_BLOCK_SIZE = next_pow2(2**18 // (H * chunk_size))` and
  `_GDN_CUMSUM_WORKING_SET = 2**18` in the builder — i.e. **the kernel is sized so `H × chunk × BT` elements fit
  in a ~256 K-element budget**. That budget is the UB, expressed in elements.
- **WY representation — yes**, `vllm_ascend/ops/triton/fla/wy_fast.py` plus `chunk_scaled_dot_kkt.py`
  (the `β K Kᵀ` build), and on the AscendC side `recompute_w_u_fwd` / `prepare_wy_repr_*`.
- **`cumprod` of a *vector* decay per head — yes** (scalar `g` per head per token). **Per-channel decay (KDA)
  is a different kernel family** (`vllm_ascend/ops/triton/kda/cumsum.py`, `kda_gate_cumsum`), and it is the part
  where a tree closed form is mathematically harder (§1.5 item 3).
- **fp32 is supported for `tl.dot`** on Ascend Triton — `input_precision="ieee"` is used in the merge kernel,
  and the dtype matrix confirms `tl.dot` supports fp32/fp16/bf16/int8. So the fp32 triangular solve is legal.
- **Graph capture works**: `gdn_attn_builder.py` builds metadata for full ACL graph capture (`use_full_cuda_graph`,
  `_can_pad_spec_decode`, graph-padding dummy rows with `_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1`).

## 2.4 Speculative decoding status in vllm-ascend (2026-10-29)

**Supported / shipped** (`docs/source/user_guide/feature_guide/speculative_decoding.md`): methods
`eagle`, `eagle3`, `mtp`, `dflash`, `dspark`, `medusa`, `draft_model`, `ngram`, `suffix`,
`extract_hidden_states`, `uno`. Dynamic draft-length K, adaptive verification, `rejection_sample_method: block`,
draft-parallel, `disable_padded_drafter_batch`.

**Hybrid/linear-attention models with MTP are first-class in CI.** Configs present in the tree:
`tests/e2e/cases/models/configs/Qwen/Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml` (+ perf/weekly, w4a8-A2),
`Qwen3.6-35B-A3B-w8a8-310p-MTP.yaml`, `GLM/GLM-5.2-W4A8-MTP-A3.yaml`, `GLM-5.1-w8a8c8-MTP-offload`,
`MiniMax/MiniMax-M3-W8A8-A3-EAGLE3.yaml`, `DeepSeek_V3.1T_MTP{1,3}_PD.yaml`; and
`tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py`,
`tests/e2e/pull_request/one_card/spec_decode/{test_dflash,test_dspark,test_eagle,test_ngram_npu,test_draft_parallel}.py`.
The doc has a hybrid-specific warning: *"Hybrid Mamba models (e.g., Qwen-Next and Qwen3.5 series):
`num_speculative_tokens` should be equal on P nodes and D nodes."*

**State handling for spec rows — the n+1 states, in code.** `vllm_ascend/ops/gdn.py::_flatten_recurrent_state_indices`
documents it exactly: *"`torch_npu.npu_recurrent_gated_delta_rule` requires `len(ssm_state_indices) == T` where T
is the number of q/k/v rows. Block-table-derived index tensors carry one entry per request row … while
**speculative rows carry one entry per draft slot**, including slots left unused when a sequence capped at
`max_model_len` carries fewer than `num_spec + 1` query tokens."* And the decode buffer is sized
`max_num_seqs * (num_spec + 1)` (`gdn_attn_builder.py`). So today: **`K+1` physical state rows per request,
`K+1` full state writes per step.** This is the F1 baseline, on Ascend, in production.

**Ascend's ReplaySSM equivalent is an open PR, and it states the memory bill.**
[vllm-ascend#18765](https://github.com/vllm-project/vllm-ascend/pull/18765) *"Add opt-in GDN checkpoint/replay
for eager MTP on Ascend"*: *"GDN speculative verification reserves a full FP32 recurrent state for every draft
position. This limits resident requests in hybrid KV pools. This PR adds an opt-in Ascend checkpoint/replay path:
verification keeps one checkpoint and small FP32 correction/key/decay records; rejection sampling commits only the
processed input prefix and matching convolution history."*
**Numbers given:** at 48 value heads, K=V=128, MTP3 → `SSM + records ≈ 3.188 MiB` vs **12 MiB for four
snapshots** (≈3.77× reduction), and "GDN speculative block reservations become zero".
**Gating is severe:** requires A2/A3, single device (TP/PP/DP/PCP/DCP = 1), **eager Model Runner V1**, dense
Qwen3.5-family target, eager native MTP with 1–5 draft tokens, BF16 model/conv + FP32 SSM, **prefix caching off**,
Mamba cache mode `none`; KV transfer and sleep mode rejected. Flags: `--additional-config '{"enable_gdn_recover":true}'`,
default off. New files include `vllm_ascend/ops/gdn_recover.py`,
`vllm_ascend/ops/triton/mamba/gdn_recover.py`, `gdn_recover_batch.py`, and design docs
`docs/source/developer_guide/Design_Documents/gdn_recover_primitives.md`,
`docs/source/user_guide/feature_guide/gdn_recover.md`. It explicitly follows upstream RecoverSSM + Qwen wiring and
leaves "FIA aliasing and the native ragged-state fix" to upstream #18376 / #17854.

**Tree / non-chain verification: not yet supported, being built right now.**
[vllm-ascend#19604](https://github.com/vllm-project/vllm-ascend/pull/19604) (2026-10-23) adds
`vllm_ascend/worker/v2/spec_decode/hooks.py` — *"speculator hooks for **non-chain verification** and draft
suspension"* — mirroring upstream [#62963](https://github.com/vllm-project/vllm/pull/62963).
[vllm-ascend#19603](https://github.com/vllm-project/vllm-ascend/pull/19603) (2026-10-27) adds *"a registry for
out-of-tree speculative methods and a **linear-attention verify extension**"*, mirroring upstream
[#62962](https://github.com/vllm-project/vllm/pull/62962). Both are `todo` on testing, both touch
`vllm_ascend/ops/gdn.py` + `gdn_attn_builder.py` + `patch/worker/patch_qwen3_5.py`. **Until these land, the
GDN path verifies chains only.**

**Other relevant open/merged-in-flight Ascend work:**
- [#17805](https://github.com/vllm-project/vllm-ascend/pull/17805) H-Spec (`mamba_attn_hybrid`) drafter on 910B3.
- [#17069](https://github.com/vllm-project/vllm-ascend/pull/17069) / [#19753](https://github.com/vllm-project/vllm-ascend/pull/19753)
  UNO (gated draft-LoRA) with `FULL_DECODE_ONLY` **tree** draft graphs, `vllm_ascend/ops/triton/uno_tree.py`;
  MATH500 tree 1.73× vs AR on 910B. Note: UNO trees work because the *target* is a plain Transformer — this is
  **not** evidence of tree verify in GDN layers.
- [#17393](https://github.com/vllm-project/vllm-ascend/pull/17393) FLA RecurrentGDN decode backend (538→148 µs).
- [#19898](https://github.com/vllm-project/vllm-ascend/pull/19898) GDN decode: fuse gating + recurrent update.
- [#18567](https://github.com/vllm-project/vllm-ascend/pull/18567) reuse GDN spec-decode metadata (V1);
  [#20094](https://github.com/vllm-project/vllm-ascend/pull/20094) graph-padded query rows in GDN spec
  `actual_seq_lengths`; [#17854](https://github.com/vllm-project/vllm-ascend/pull/17854) preserve recurrent state
  rows for **ragged MTP** batches; [#18655](https://github.com/vllm-project/vllm-ascend/pull/18655) backport of same.
- [#19954](https://github.com/vllm-project/vllm-ascend/pull/19954) fuse GDN metadata for mixed batches;
  [#16246](https://github.com/vllm-project/vllm-ascend/pull/16246) share group-invariant metadata across KV groups.
- KDA state copy: `vllm_ascend/ops/kda_state_copy_plan.py`, `ops/triton/kda_state_copy_kernel.py`
  (doc: `ops/triton/docs/kda_state_copy.md`) — a *fused Triton gather/scatter plan* on A5/950, masked byte-copy
  fallback on A3. Explicitly a tuned state-copy path with 4 compiled variants (INT32/INT64 × gather/scatter),
  fixed block 8192, and a lifecycle (`prepare` → `seal` → `publish`) that forbids JIT during serving.

## 2.5 Primitive availability matrix (what a chunked-GDN tree verifier needs)

| Primitive needed | Exist on Ascend? | Form | Constraint / evidence |
|---|---|---|---|
| `inv(I − tril(diag(β)KKᵀ))`, BT=64 | **yes** | Triton `solve_tril.py`; AscendC `solve_tri` | 16×16 is a 16-step serial substitution; 32/64 merge via fp32 `tl.dot` ieee. Tunable `LARGE_BLOCK_T=1216` rows/launch |
| WY construction / `w,u` recompute | **yes** | Triton `wy_fast.py`; AscendC `recompute_w_u_fwd`, `prepare_wy_repr_*` | — |
| decay cumsum / cumprod | **yes** | `tl.cumsum` (fwd + `reverse=`); AscendC `chunk_local_cumsum`, `kda_gate_cumsum` | working set `2**18` elements → sizes the grid, not the math |
| chunk fwd state pass (`h`) | **yes, two impls** | AscendC `csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/` (arch20/22/35); Triton `chunk_delta_h.py` | AscendC def registers `ascend910b` + `ascend910_93` |
| chunk fwd output pass (`o`) | **yes, two impls** | AscendC `csrc/moe/chunk_fwd_o_vllm/`; Triton `chunk_o.py` | — |
| recurrent decode (1 token) | **yes** | `torch_npu.npu_recurrent_gated_delta_rule`, `fla_npu.recurrent_gated_delta_rule`, Triton `recurrent_gated_delta_rule` | `fla_npu` backend 538→148 µs (PR #17393) |
| state gather / scatter by row id | **yes, tuned** | Triton `gather_ssm_states` / `scatter_ssm_states_` (`ops/triton/docs/mamba_state_index.md`), KDA fused plan | tile = 1024 elements; row stride may be padded but **inner dims must be contiguous**; **duplicate scatter destinations race and are not validated** |
| strided / gather copies fast? | **partly** | see above + `ops/triton/batch_memcpy.py`, FIA `fused_scatter_copy_sparse_flash_attention` | 512 B alignment matters (≤30% BW); `tl.gather` exists but `extension.gather` is skipped "waiting for the compiler" |
| fp32 matmul for the solve | **yes** | `tl.dot(input_precision="ieee")` | `tl.dot` supports fp32/fp16/bf16/int8; int16/int32/int64 dot **not** supported |
| `int64` index arithmetic | **partial** | — | `tl.arange` is **int32 only**; `tl.trans` does not support int64; the KDA state-copy doc notes address math must widen to INT64 *in the wrapper*, not in-kernel |
| atomics for scatter-add | **limited** | `atomic_add`, `atomic_cas`, `atomic_max/min` | `atomic_or/xor/and/xchg/cas` **not supported in loops**; `atomic_add` **does not support multi-core add + saving intermediate** |
| 3-D non-adjacent-axis `trans` | **fragile** | `tl.trans` supports it in pytest but generalization cases are commented out: *"not support yet: need bisheng support later"* | explains the `extract_slice`/`insert_slice` style in `solve_tril` |
| dynamic shapes | **restricted** | `do_not_specialize=[...]` annotations, `IS_VARLEN` heuristics, pre-computed `chunk_indices` on host | vllm-ascend pre-computes `chunk_indices`, `cu_seqlens` compacted, on host; kernels are compiled per layout, and the KDA copy doc forbids JIT at serving time |
| `dot_scaled` (MXFP8 etc.) | **not supported** | — | matters if you were hoping to do the solve in low-precision scaled formats |
| fp16 dense GDN | **falls back** | bf16-only fused kernels | `supports_fused_dtype = query.dtype == torch.bfloat16`; fp16 → Triton pipeline |

## 2.6 Ranked blockers

**B1 — Verify-time state multiplicity is still the default and the fix is not merged.**
Today's Ascend GDN spec path allocates `max_num_seqs × (K+1)` state rows and writes `K+1` full states per step
(`_flatten_recurrent_state_indices`, `gdn_attn_builder.py`). With `S = H·V·K` and 48 heads × 128 × 128 fp32
that is **12 MiB/request at MTP3** just for snapshots. The checkpoint/replay fix exists only as
open PR #18765, gated to *eager V1, single device, prefix caching off, dense Qwen3.5, bf16, no KV transfer*.
Nothing in `main` reduces it. This is blocker #1 for residency, and it is a **memory** blocker, not a compute one.

**B2 — No non-chain (tree) verification in the GDN path.** `hooks.py` / `LinearAttnVerifyExtension` equivalents
are open PRs #19603/#19604. Until they land, any tree scheme requires forking `ops/gdn.py` +
`gdn_attn_builder.py`, exactly the thing upstream #62962 was created to avoid. Also: **nothing in the Ascend tree
implements a branch-local GDN scan** — the closest reference implementation is GDN Tree-Scan (2609.23900, vLLM
CUDA/FA2) and `vllm_ascend/ops/triton/kda/recoverssm.py` for the chain case.

**B3 — UB (192 KB) turns every "just fuse it" plan into a tiling project.** The in-tree Ascend `solve_tril`
already needed `extract_slice`/`insert_slice` and block packing *to avoid UB OOM*, and the cumsum kernel sizes
itself off a `2**18`-element budget. A tree verifier keeps, per node or per chunk: `A` (64×64 fp32 = 16 KB),
`w`,`u` (64×128 each = 32 KB), `k`,`v`, the running state tile, plus the tree bias mask for attention layers.
Realistically 2–4 live tiles. **Consequence: on Ascend you cannot port a GPU kernel that assumes 96–228 KB of
shared memory per block; you must re-block, and the block size becomes a per-SKU tuning parameter (arch20 vs
arch22 vs arch35 have different UB sizes: 192 KB vs 192 KB vs 248 KB usable).**

**B4 — The triangular solve is serial in 16 steps and fp32-only, so it is a latency floor, not a throughput
problem.** `solve_tril_16x16_kernel` does 16 dependent row updates (`for i in range(1,16)`) before any `tl.dot`.
On a part with 1 AIC : 2 AIV and a 192 KB UB, that loop is 16 sequential vector ops of a 64-wide-ish reduction
per program — hard to hide. The mitigation is arithmetic: use the *matrix-product-rich* direct/iterative
algorithms from [2605.21325](https://arxiv.org/abs/2605.21325) (which benchmarks exactly this on NPUs), and/or
shrink `BT` for the solve while keeping `BT=64` for the state pass. **This is the one place where a paper gives
you a concrete, NPU-validated alternative.**

**B5 — State gather/scatter cost and alignment are a first-class tax, and the fast path is A5-only.**
ReaDy measured a 71.8% win purely from fusing a state gather (16.4 → 4.6 µs) on a B200. On Ascend,
`gather_ssm_states` copies 1024-element tiles per program with row-stride addressing (fine for a `[H,V,K]`
bf16 state of 32 KB/head), but: inner dims must be contiguous, scatter destinations must be **unique or they
race silently**, and the genuinely fast fused plan (`KDAStateCopyPlan`) is **A5/950-only** with a masked
byte-copy fallback on A3. A `K`-way gather + accepted-only scatter per layer per step is `2·K·S` bytes of
D2D traffic; at 48 heads × 32 KB that is 1.5 MB per layer per step at K=1 — this is why B1's "write once at
commit" matters more than shaving the kernel.

**B6 — Dynamic shapes + graph capture fight each other, and the workarounds are visible.** Spec decode needs
`K+1` rows per request with ragged lengths; Ascend needs static shapes for ACL graph. The tree shows the scars:
host-precomputed `chunk_indices`/`cu_seqlens`, `do_not_specialize` on `T`/`H`, `_compact_empty_segments`,
graph-padding rows with a zero-length query segment and `_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1`,
`_flatten_recurrent_state_indices` carefully avoiding any device→host sync ("the output length is host-known,
so this path stays sync-free, unlike masked_select"). **A tree verifier has a *data-dependent* accepted-path
length**, which is the worst case for this regime. This is why GDN Tree-Scan's "device-side multidraft
commitment" is the interesting bit, and why a naive tree port will lose to the chain implementation.

**B7 — Library-support risk: `fla_npu` is third-party and small.** 57 stars, Tianjin-University-led, prebuilt
wheels tied to CANN version + SoC. vllm-ascend treats it as optional (`_get_fla_gdn_prefill_op` returns `None`
on ImportError; `HardwareCapability.FLA_GDN_PREFILL` gate). If your design depends on `solve_tri`/
`recompute_w_u_fwd` from this library, you inherit its CANN-version matrix. The in-tree fallback (Triton FLA port
in `vllm_ascend/ops/triton/fla/`) is the safer dependency and is what fp16 models already use.

**B8 — KDA is *kernel-covered* but *math-uncovered*.** Ascend kernel coverage for KDA is real: Triton port in
`vllm_ascend/ops/triton/kda/`, `fla_npu` AscendC `chunk_kda_fwd{,_prepare,_finalize}` + `recurrent_kda` +
`kda_gate_cumsum` + full bwd, plus in-tree AscendC `csrc/attention/kda_gate_cumsum_vllm/` and
`csrc/attention/kda_layout_swap12/`. Notably `csrc/attention/chunk_kda_fwd/` ships **design + api docs only**
(no `op_kernel/` yet) and specifies a *fully fused* KDA chunk forward: *"Gate、Prepare、PostWu、FwdH and Finalize
all complete inside one physical `ChunkKdaFwd` L0; L2 no longer concatenates or launches several stage-L0s"* —
i.e. the direction of travel is **one fused L0 kernel for the whole chunk pipeline**, which is exactly what a
spec-decode inner loop needs. The gap is not kernels, it is **math**: there is no published tree/WY closed form
for KDA (§1.5 item 3), because KDA's *per-channel* decay does not commute with the rank-1 update the way GDN's
per-head scalar does. If the target is Kimi-Linear/Kimi-K3 rather than Qwen3.5, expect to derive it yourself.

*(Not a blocker but worth stating: `triton-ascend` maturity is the enabling constraint behind all of the above.
The evidence is indirect but consistent — the Triton FLA port exists and is used in production for fp16, yet
`tl.gather`/`extension.gather` is skipped pending compiler support, 3-D non-adjacent `trans` is "need bisheng
support later", `dot_scaled` is absent, and int64 `arange`/`trans` are absent. **Verdict: Triton-Ascend is
production-usable for elementwise + fp32 `tl.dot` + cumsum + 2-D broadcast, and not yet usable for exotic
indexing, int64, or scaled-dot formats.** **[UNVERIFIED: exact triton-ascend release version in the wheel set
vllm-ascend CI pins — I did not open the pinned requirements file.]**)*

## 2.7 What a realistic port looks like

Ordered by leverage. **(0) Do not write a verifier from scratch** — write it as a plugin against the upstream
`LinearAttnVerifyExtension` from PR 62962 + MRV2 hooks, so you inherit the rejected-branch state publication and
rejection-sampling interfaces instead of re-deriving them. Ascend mirrors #19603/#19604 are the merge points.

1. **Ship the chain path first (F2), because the plumbing already exists on both sides.** The Ascend version of
   GDN checkpoint/replay is PR #18765 (3.188 MiB vs 12 MiB, `num_speculative_blocks = 0`). Start from it, relax
   its gating (it is eager-V1-only, single-device-only), and treat "records in fp32, state write once at commit"
   as the contract. Effort: integration + validation, not new math.
2. **Reuse `solve_tri` + `recompute_w_u_fwd` verbatim for the tree (F3).** TreeWY/Bole/STree all reduce to
   "build the `64×64` in-block matrix with a tree-structured mask instead of a causal mask, then one solve".
   On Ascend that means editing the *mask construction* (`chunk_scaled_dot_kkt` equivalent) and the *scan*
   (branch-local instead of sequential), not the inverse. This is the cheapest possible tree.
3. **Prefer the Triton-Ascend path over AscendC for the first version**, and AscendC only for the two ops that
   dominate time. Rationale: the Triton FLA port (`ops/triton/fla/`) already exists, is bit-comparable to
   `fla_npu` (there is a UT named `test_recurrent_gated_delta_rule_fla_compare.py`), supports fp16 *and* bf16,
   and is iterable. Move `chunk_gated_delta_rule_fwd_h` and `chunk_fwd_o` to AscendC only after profiling —
   those are the two ops vllm-ascend already chose to hand-write.
4. **Budget the solve as a latency floor and attack it with 2605.21325's algorithms, not with more kernels.**
   If `solve_tril` is >25% of verify time (plausible: 16 serial steps), switch to a matrix-product-rich direct
   inverse and/or decouple `BT_solve` from `BT_state`.
5. **Treat the state gather/scatter as its own optimization.** On A3 the KDA fused plan degrades to byte-copy;
   for a `K`-draft tree you want a *single* gather of the `K+1` slots and a *single* scatter of the accepted slot,
   with `int32` indices (int64 is a compiler liability) and 512 B-aligned inner dims. Consider `block=8192`
   like the KDA plan rather than 1024.
6. **Only then** consider F4 (KVBuffer-style "no state during verify") — it is the biggest win conceptually
   (0 temporary states) but requires reworking the decode kernel to read from a K/V ring buffer instead of a
   state, which is a much deeper change to `ops/gdn.py` and the cache spec.
7. **Do not** attempt the "negative delta / receipt" shortcut (subtract a rejected draft's contribution).
   [2609.06872](https://arxiv.org/abs/2609.06872) shows it fails empirically on Kimi Linear beyond bf16 noise,
   and gives the exact condition for when it would work. Replay is the correct commit.

## 2.8 Uncertainty ledger

| Claim | Confidence |
|---|---|
| GDN chunk pipeline (solve_tri, WY, cumsum, fwd_h, fwd_o) exists as AscendC in `fla_npu` + Triton in vllm-ascend | **High** — read the trees, op names, and the import sites |
| Ascend GDN spec path allocates one state row per draft slot (`K+1`) | **High** — `_flatten_recurrent_state_indices` docstring + `num_spec+1` buffer sizing |
| PR #18765 gives 3.188 MiB vs 12 MiB at 48 heads / K=V=128 / MTP3 | **High** (PR text), but the PR is **open, unmerged**, and the author says "historical 27B throughput is not presented as performance of this integrated revision" |
| A2/A3 UB = 192 KB, A5 = 248 KB usable; HBM 1.6–1.85 TB/s / 64 GB on 910B | **Medium-high** — two independent community/CANN-ecosystem sources agree; not Huawei whitepapers. Individual SKUs are cut (910B3/B4 = 20 cores, B4 = 32 GB) |
| Ascend 950 family "self-developed HBM up to 144 GB / 4 TB/s" | **Low** — press coverage of a Huawei announcement; SKU mapping and availability unknown. Do not size a design on it |
| `arch20/arch22/arch35` = 910B / 910_93 / 950 | **Medium** — directory names + the SoC strings in `def.cpp`; I did not find an explicit mapping table |
| triton-ascend limits (`tl.arange` int32-only, no int64 trans, atomic loop restriction, no `dot_scaled`) | **Medium-high** — from an official-ish "Ascend Triton API dtype matrix" doc in the CANN skills ecosystem, dated before 2026-10-29; version not pinned |
| TreeWY/Bole/SpecLA one-line mechanism summaries | **High** — read abstracts + intro; full derivation not re-derived |
| "No RWKV-6/7 speculation paper exists" | **Medium** — 0 hits across ~35 queries including `all:"RWKV" AND all:speculative`; absence of evidence |
| The 2605.21325 NPU speedup multiplier | **Low** — math stripped by HTML→text; the *claim* (NPU benchmarks, large speedup over SGLang, full accuracy) is high confidence |
| cann-recipes / ascend-tribe contents | **Unverified** — `github.com/Ascend/{cann-recipes,ascend-tribe}` 404; `gitee.com/ascend/cann-recipes` 403; `gitcode.com/Ascend/*` 404. I could not read them, so I make **no** claim about whether they contain GDN kernels |

---

# Sources

**Papers (arXiv abs)**
- 2610.26660 Tandem — https://arxiv.org/abs/2610.26660
- 2610.20503 ReaDy — https://arxiv.org/abs/2610.20503 · code https://github.com/StartLuxLabs/ReaDy
- 2609.38169 STEPQuant — https://arxiv.org/abs/2609.38169
- 2609.24197 H-Spec — https://arxiv.org/abs/2609.24197 · code https://github.com/weifanjiang/H-Spec
- 2609.23900 GDN Tree-Scan / LumoTree — https://arxiv.org/abs/2609.23900
- 2609.06872 Exact Record Omission — https://arxiv.org/abs/2609.06872 · code https://github.com/vishrmsh/delta-attention-record-omission
- 2607.21535 Windowed-MTP — https://arxiv.org/abs/2607.21535
- 2605.22791 Gated DeltaNet-2 — https://arxiv.org/abs/2605.22791
- 2605.21325 Fast and Stable Triangular Inversion — https://arxiv.org/abs/2605.21325
- 2605.19049 KVBuffer — https://arxiv.org/abs/2605.19049
- 2605.05219 Sparse Prefix Caching — https://arxiv.org/abs/2605.05219
- 2605.01106 Component-Aware Self-Speculation — https://arxiv.org/abs/2605.01106
- 2603.09555 Compiler-First SSD — https://arxiv.org/abs/2603.09555 · code https://github.com/CosmoNaught/mamba2-jax
- 2509.19873 SpecMamba — https://arxiv.org/abs/2509.19873
- 2506.01206 Mamba Drafters — https://arxiv.org/abs/2506.01206
- 2505.14969 STree — https://arxiv.org/abs/2505.14969 · code https://github.com/wyc1997/stree
- (context) 2609.33477 Suffix Replay · 2608.30310 Tail-Replay · 2607.01299 HYPIC · 2610.18143 RadixFlow ·
  2605.16640 Shorter Scratchpads · 2603.05931 FPGA persistent-state · 2607.25852 AngelSpec · 2610.21436 HyCo ·
  2601.17768 LLM-42 · 2606.00601 ScanWeaver — all `https://arxiv.org/abs/<id>`

**vLLM upstream (spec decode / ReplaySSM / linear-attention verify)**
- #58863 RecoverSSM for Qwen GDN — https://github.com/vllm-project/vllm/pull/58863
- #61309 RecoverSSM Qwen3.5/Qwen3-Next — https://github.com/vllm-project/vllm/pull/61309
- #61566 KDA RecoverSSM GLM-5.3-Flash — https://github.com/vllm-project/vllm/pull/61566
- #62004 ReplaySSM for standard decode — https://github.com/vllm-project/vllm/pull/62004
- #62962 out-of-tree spec method registry + `LinearAttnVerifyExtension` — https://github.com/vllm-project/vllm/pull/62962
- #62963 MRV2 calls for non-chain verifiers — https://github.com/vllm-project/vllm/pull/62963
- also #59982 #59983 #61017 #56082 #56466 #52962 #55688 #50021 #57128 #59366 #62625 #63176 #63918 #61124

**vLLM-Ascend**
- repo — https://github.com/vllm-project/vllm-ascend
- #18765 GDN checkpoint/replay for eager MTP (3.188 MiB vs 12 MiB) — https://github.com/vllm-project/vllm-ascend/pull/18765
- #19603 linear-attention verify extension — https://github.com/vllm-project/vllm-ascend/pull/19603
- #19604 speculator hooks for non-chain verification — https://github.com/vllm-project/vllm-ascend/pull/19604
- #17805 H-Spec on Ascend (910B3) — https://github.com/vllm-project/vllm-ascend/pull/17805
- #17393 FLA RecurrentGDN decode backend (538→148 µs) — https://github.com/vllm-project/vllm-ascend/pull/17393
- #17069 UNO tree draft graphs — https://github.com/vllm-project/vllm-ascend/pull/17069
- #19406 RFC: EAGLE-4 spec decode (PR pairing table) — https://github.com/vllm-project/vllm-ascend/issues/19406
- spec-decode feature guide — https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/speculative_decoding.md
- FLA Triton port — https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/fla
  (`solve_tril.py`, `wy_fast.py`, `cumsum.py`, `chunk.py`, `chunk_delta_h.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`)
- KDA Triton port — https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/kda
- GDN layer — https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn.py
- GDN metadata/builder — https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn_attn_builder.py
- AscendC chunk GDN `fwd_h` op — https://github.com/vllm-project/vllm-ascend/tree/main/csrc/moe/chunk_gated_delta_rule_fwd_h_vllm
- AscendC chunk `fwd_o` op — https://github.com/vllm-project/vllm-ascend/tree/main/csrc/moe/chunk_fwd_o_vllm
- KDA state copy doc — https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/docs/kda_state_copy.md
- mamba state gather/scatter doc — https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/docs/mamba_state_index.md

**Ascend hardware / toolchain**
- arch facts (dav-2201: UB 192 KB, 1.85 TB/s; dav-3510: UB 256 KB) — https://raw.gitcode.com/ascendc-agent/ascendc-skills/raw/980be3b0d7088f22fa4754908048e43c63f156c6/skills/sample/asc-perf-tuning/references/arch-facts.md
- Ascend Triton API dtype matrix — https://raw.gitcode.com/Ascend/agent-skills/raw/5a44ddfa63e333c0761fd7810f17b80fa6b384e6/skills/triton-operator-code-review/references/ascend-api-dtype-matrix.md
- 910B/910C/950 spec table (cores, UB/L0, HBM) — https://ai6s.net/6aa8af828f401d3626604ea0.html
- CANN AscendC basic architecture (AIC/AIV split) — https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/programug/Ascendcopdevg/docs/guide/编程指南/高级编程/硬件实现/基本架构.md
- TPU/Triton-Ascend faq — https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/FAQ.md
- `fla_npu` (AscendC FLA library, A2/A3/A5) — https://github.com/flashserve/flash-linear-attention-npu
- TileLang-Ascend adapter — https://github.com/tile-ai/tilelang-ascend
- Atlas 800T A3 product page — https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3
- Ascend 950 self-developed HBM (144 GB / 4 TB/s, press) — https://www.eet-china.com/mp/a439210.html
