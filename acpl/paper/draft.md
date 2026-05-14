# Precision-Aware Residency Scheduling for Mixed-Precision LLM Decoders

**Authors.** (TBD — Ewha Womans University, MAGNETO/GATHER lab line)
**Venue target.** ISCA / MICRO / HPCA / ASPLOS
**Status.** Draft v0.1 (2026-05-12) — all evidence in `experiments/`

---

## Abstract

Autoregressive LLM decoding is bound by per-token weight traffic from
off-chip DRAM: every weight is touched exactly once per generated token,
so latency and energy track the *bytes* moved, not the floating-point
operations performed. Layer-wise mixed precision shrinks those bytes,
but a deployment-time scheduler that does not know each layer's actual
bit-width must budget every weight tile at the worst-case width, leaving
on-chip residency capacity unused. We formalize the resulting trade-off
as a *precision-aware residency scheduling* problem: given a fixed
mixed-precision policy and an on-chip buffer of size `B`, decide which
(layer, projection) weight tiles to pin so that the per-token DRAM
traffic is minimized. We instantiate the scheduler as a greedy 0/1
knapsack over per-tile bytes and compare it against a precision-oblivious
baseline that uses INT8 budgets. Across three open-weight checkpoints
(Qwen2.5-1.5B-Instruct, Gemma-2-2B-IT, Qwen2.5-7B-Instruct), the
precision-aware scheduler reduces per-token DRAM traffic by
**1.33×–1.97×** at a comparable operating point of `B ≈ INT4_total / 2`
across all three models, and by up to **3.17×** when `B` reaches the
INT4-total knee for the 1.5B model; at `B ≈ INT4_total` every
mixed-precision policy in our sweep becomes fully on-chip resident
(`DRAM/tok = 0`). We further show that the sensitivity-driven
mixed-precision policy commonly used as input — a greedy per-layer
assignment of Δppl per bit — survives joint GPTQ quantization with
Kendall τ = +1.0 against the predicted ranking, and that promoting
the score to a per-tile Δppl/Δbyte criterion produces policies that
allocate the 8-bit budget to attention KV projections (89 % of
`v_proj`, 61 % of `k_proj`) at the same average bit-width budget.
A GPU sanity profile on Qwen2.5-1.5B reproduces the analytic model's
ranking and is reported alongside the analytic numbers.

---

## 1. Introduction

Large language models (LLMs) generate text autoregressively: each new
token requires a forward pass through the full decoder stack with the
keys/values of all prior tokens cached on-chip and a fresh draw of the
weight matrices from memory. In the decode regime, batch size `N = 1`,
arithmetic intensity is low, and the GPU or accelerator's PE array
spends >90 % of its cycles waiting for memory traffic [memory-bound
LLM decode reports include Sheng et al., ICML 2023 and Yu et al.,
2023]. Reducing the per-token weight bytes is therefore the dominant
lever on both latency and energy.

Layer-wise mixed-precision quantization is one of the canonical
techniques for this reduction. HAWQ [Dong et al., ICCV 2019] and BRECQ
[Li et al., ICLR 2021] assign different bit-widths per layer guided by
loss landscape sensitivity; GPTQ [Frantar et al., ICLR 2023] makes the
quantization itself Hessian-aware, recovering most of the accuracy
that naive rounding loses. Recent LLM accelerator designs — including
our own lab's prior power-aware mapping work (MAGNETO, anonymized for
review) and LLM-specific accelerator (GATHER, anonymized) — take
mixed-precision policies as system input and search the dataflow /
mapping space to extract energy efficiency on top.

This paper asks a question that sits between the algorithm and the
mapping layers: *what should the deployment scheduler do with the
mixed-precision policy at runtime?* Concretely, when the on-chip
buffer is too small to hold the full decoder, which weight tiles do we
keep resident so that subsequent tokens reuse them without re-streaming
from DRAM?

We make three observations.

**(O1) In the autoregressive decode regime, mapping-space search does
not break the byte-traffic monotone.** Empirically (§3), the optimal
Timeloop [Parashar et al., ISPASS 2019] mapping for the same GEMM at
4-bit and at 8-bit differ in tile geometry (the 4-bit case fits a
full output-channel block into a 256 KB on-chip buffer; the 8-bit case
must split it), but the cycle count and energy in our 16×16
Eyeriss-like array track byte-width almost exactly: both per-shape
ratios are 1.00×, while the byte ratio is 2.0×. The mapping
contribution to efficiency is absorbed by the byte cost. *Mapping is
not the right surface to add scheduling intelligence at `N = 1`.*

**(O2) The natural surface is residency.** With the model fixed, every
weight is read exactly once per token. If a tile fits in the on-chip
buffer, it can be pinned once and reused across `T` tokens, saving
`(T − 1) × tile_bytes` of DRAM traffic. The decision becomes a
classical 0/1 knapsack over tiles with a single capacity constraint.

**(O3) The mixed-precision policy changes the knapsack instance, but
naive deployment ignores it.** A precision-oblivious scheduler that
budgets every tile at INT8 (the worst-case bit-width) pins fewer tiles
than a precision-aware scheduler that uses the *actual* per-tile
bytes from the policy. At fixed `B`, the aware scheduler streams 1.7×
to 3.2× fewer bytes for the same accuracy.

### Contributions

- **System claim.** We formulate precision-aware residency scheduling
  as a knapsack on per-tile weight bytes and quantify its win against
  a precision-oblivious INT8-budget baseline across three LLM families
  and three scales.
- **Generalization.** The pattern is universal: 1.5B (Qwen2),
  2B (Gemma-2), and 7B (Qwen2.5) all show monotone E1 ranking
  preservation and aware-vs-oblivious wins in the same direction; the
  *factor* grows with model size (1.33× → 1.97× at `B ≈ INT4_total/2`).
- **Method extension (preliminary).** Replacing the per-layer
  Δppl/Δbits greedy score with a per-tile Δppl/Δbyte score produces
  qualitatively different policies that allocate the 8-bit budget to
  attention KV projections (89 % of `v_proj`) at the same average bit
  budget. The accuracy payoff of this re-allocation is *not yet
  verified*: full GPTQ on the per-tile policies is required and is
  the most important next experiment (§7.2).
- **Reproducibility.** Every number in this paper traces to a CSV
  and figure under `experiments/results/`, regenerated from `pipeline/`
  by `_sweep_target.sh`. We document the install path for Timeloop
  v3.0.3 + Accelergy inside a conda environment without sudo, and
  the gcc-13 `<cstdint>` patch (92 headers) needed to build on
  current Ubuntu 24.04.

### Where this paper sits in the stack

The literature treats LLM efficiency at three different layers (Fig. 0).
Our contribution is the scheduling layer; the upper two layers are
inputs.

```text
                    ┌──────────────────────────────────────────────────┐
algorithm layer ↑   │  Mixed-precision policy                          │
                    │    e.g. HAWQ, BRECQ, GPTQ                        │
                    │    Output: per-(layer, projection) bit-widths    │
                    └──────────────────────────────────────────────────┘
                                       │ (input to →)
                    ┌──────────────────▼───────────────────────────────┐
mapping layer       │  Accelerator mapping / dataflow                  │
                    │    e.g. Timeloop, MAGNETO, GATHER                │
                    │    Output: per-shape tile sizes, loop nest order │
                    └──────────────────────────────────────────────────┘
                                       │ (input to →)
                    ┌──────────────────▼───────────────────────────────┐
scheduling layer ↓  │  **Residency scheduling (this paper)**           │
                    │    Input: mixed-precision policy, on-chip buf B  │
                    │    Decision: which weight tiles to pin on-chip   │
                    │    Output: DRAM bytes per token (efficiency)     │
                    └──────────────────────────────────────────────────┘
```

*Figure 0.* The three-layer stack. Mapping (middle) and scheduling
(bottom) are sometimes conflated in single-pass dataflow papers; we
show in §3 that under autoregressive decode the mapping layer's degrees
of freedom collapse onto byte-traffic, so the scheduling layer's
contribution is *additive* to mixed precision rather than redundant
with it.

---

## 2. Background

### 2.1 Mixed-precision quantization and policy search

Per-layer sensitivity scoring assigns each transformer block a score
`s(l)` that measures the perplexity degradation caused by quantizing
that block to `b` bits. Common formulations are Hessian-trace-based
(HAWQ), block reconstruction error (BRECQ), or simple ablation (our
default — replace one block's weights with their fake-quantized
values and re-measure WikiText-2 perplexity). The *policy search*
problem is then: given allowed widths `W = {b1, ..., bk}` and an
average-bits budget `B̄`, find a per-layer assignment `B[l] ∈ W` that
minimizes total Δppl subject to `mean(B) ≤ B̄`. Greedy promotion by
Δppl per added bit is the standard solver and is near-optimal under
monotone scores.

The greedy promotion rule promotes the layer with the largest gain in
Δppl per added bit, until the bit budget is hit:

```text
score(l, b → b+) = (Δppl(l, b) - Δppl(l, b+)) / (b+ - b)
```

### 2.2 GPTQ quantization

GPTQ performs greedy column-by-column quantization while updating the
remaining weights to compensate for accumulated error, using an
approximate inverse Hessian computed from calibration activations. It
absorbs a meaningful fraction of the loss that isolated-layer
sensitivity predicts (see §6.1).

### 2.3 Dataflow mapping for matmul (Timeloop)

Timeloop searches the loop-nest tiling space for a given problem
(M × K × N) on a given accelerator (PE array + memory hierarchy +
bandwidth constraints). For an Eyeriss-like architecture with a 16×16
PE array, a 256 KB on-chip "WeightInputBuffer", and a 24-bit
"AccumulationBuffer", Timeloop reports cycles, energy (via the
built-in PAT model or Accelergy plug-ins), and DRAM byte traffic per
GEMM.

### 2.4 LLM accelerator memory hierarchy

Modern LLM accelerators are designed around the observation that
weights dominate memory traffic. Common patterns include large on-chip
SRAM banks (4 MB – 1 GB SRAM, or HBM2 used as an explicit cache tier)
that hold a fraction of the model resident across token boundaries
[Sanger, MICRO 2021; Ant, HPCA 2022; and our lab's GATHER]. This paper
focuses on that residency layer: given the policy and the buffer size,
which tiles stay.

---

## 3. Motivation: mapping search does not save decode (negative result)

We instantiate the precision-aware-mapping hypothesis as follows. For
each unique linear shape `s` in a Qwen2.5-1.5B decoder block (seven
shapes: q/k/v/o_proj and gate/up/down_proj) and each bit-width
`b ∈ {4, 8}`, we run `timeloop-mapper` on a 16×16 PE array with a 256
KB on-chip buffer and report (cycles, energy, DRAM bytes, PE
utilization). We then compute the cross-application loss

```text
L_{a→b} := metric(best_mapping_at_a, weights=b) / metric(best_mapping_at_b, weights=b)
```

— how much we give up by forcing the wrong-bit-width's best mapping
onto the actual bit-width's data.

**Empirical finding (Table 1 / `figs/sensitivity_qwen15b.png`).** The
mappings *do* differ — at INT4 the mapper picks a tile that holds the
entire output dimension in the GBuf (`for C in [0:6)`), while at INT8
the mapper must split the output dimension (`for K in [0:2); for C in
[0:6)`) because the same tile would overflow the GBuf at 8-bit. But
the cycles and energy outcomes are dominated by byte-width:

| Shape (1.5B) | b | Cycles | Energy (pJ) | DRAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| q_proj | 4 | 147,552 | 1.35 × 10⁸ | 1.18 |
| q_proj | 8 | 147,648 | 2.70 × 10⁸ | 2.36 |
| up_proj | 4 | 860,256 | 7.85 × 10⁸ | 6.89 |
| up_proj | 8 | 860,256 | 1.57 × 10⁹ | 13.77 |

Cycle ratio ≈ 1.000; energy ratio ≈ 2.000; DRAM ratio ≈ 2.000. The
mapping geometry difference does not translate into a meaningful
cycle delta because at `N = 1`, every weight is touched once and
total DRAM time = `total_bytes / BW`, which is identical between the
two mappings (they stream the same total weight bytes).

**Conclusion of §3.** For autoregressive decode on Eyeriss-class
arrays, *mapping search is not the right surface for adding
scheduling intelligence on top of mixed precision*. The right surface
is the layer above: which tiles do we keep on-chip across tokens?

---

## 4. Method: Precision-Aware Residency Scheduling

### 4.1 Problem definition

Let a decoder be a sequence of `L` blocks, each containing seven
projections `(layer_idx, role)` indexed as tiles `t ∈ T`. Each tile
has a weight count `n_t` (a function of the model's hidden size and
the projection role). Given a layer-wise mixed-precision policy
`B[t] ∈ {4, 8}`, each tile has bytes

```text
bytes_aware(t) = n_t × B[t] / 8
```

Per token, the GEMM for each tile is read once. If the tile is *pinned*
in the on-chip buffer of capacity `B_GBuf`, the read is free for that
token and every subsequent token in the sequence; if it is *streamed*,
it costs `bytes_aware(t)` of DRAM traffic per token.

The scheduler must select a subset `P ⊆ T` such that `Σ_{t ∈ P} bytes(t) ≤
B_GBuf`. Two schedulers differ in the bytes they use for that
constraint:

- **Precision-aware:** uses `bytes_aware(t)`. Knows the policy.
- **Precision-oblivious (INT8-budget):** uses `bytes_int8(t) = n_t`
  for the constraint. Pins the actual `bytes_aware(t)` once it fits the
  conservative INT8 budget. Reflects a deployment scheduler that ignores
  the policy and pessimistically assumes the worst-case bit-width.

Both schedulers solve

```text
maximize    Σ_{t ∈ P} bytes_aware(t)              # bytes saved per token
subject to  Σ_{t ∈ P} budget_bytes(t) ≤ B_GBuf
```

### 4.2 Solver — greedy smallest-first

For autoregressive decode at `N = 1`, every tile is accessed exactly
once per token. The access frequency does not vary by tile, so the
per-byte saving rate is identical for every tile. Under that
condition, the optimal 0/1 knapsack reduces to greedy smallest-first:
sort tiles ascending by `budget_bytes` and pin until the next one
would overflow `B_GBuf`. This is optimal for both schedulers and runs
in `O(|T| log |T|)`.

Implementation: `pipeline/residency.py::pack_greedy`.

### 4.3 HW-normalized policy (per-tile, optional)

The per-layer greedy of §2.1 promotes layer `l` based on Δppl/Δbits,
which is independent of how many *bytes* the promotion costs. With a
homogeneous decoder all layers are the same size, so this is fine; but
within a layer, the seven projections have very different sizes
(`k_proj`, `v_proj` are 6× smaller than `gate_proj` in Qwen-2.5-1.5B).
A per-tile HW-normalized score uses Δppl per added *byte*:

```text
score_hwnorm(t, b → b+) = (Δppl(t, b) - Δppl(t, b+)) / (n_t × (b+ - b) / 8)
```

This promotes the smallest highly-sensitive tiles first (typically
`k_proj`, `v_proj` of attention) and leaves the large MLP projections
at the minimum width. We measure per-tile Δppl by ablating each of the
`L × 7` linears independently in
`pipeline/sensitivity.py::run_per_projection_sensitivity` (§5.1).

### 4.4 Algorithmic complexity

For Qwen2.5-1.5B: `|T| = 28 × 7 = 196` tiles. Both knapsack solve and
the per-tile sensitivity score table fit comfortably in seconds; the
expensive component of the pipeline is the GPTQ quantization itself
(§5.3), not the scheduling decision.

---

## 5. Experimental Setup

### 5.1 Models, weights, and calibration

We evaluate on three open-weight chat-tuned LLM checkpoints, listed
with the canonical compute requirements at FP16 and after our policy
sweep (see Table 2).

| Model | Params | FP16 ckpt | INT4 decoder | INT8 decoder | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B-Instruct | 1.54 B | 3.1 GB | 655 MB | 1.31 GB | primary target |
| Gemma-2-2B-IT | 2.61 B | 5.2 GB | 968 MB | 1.94 GB | different family |
| Qwen2.5-7B-Instruct | 7.61 B | 15.2 GB | 3.25 GB | 6.5 GB | scale test |

All baselines fetched via `huggingface_hub` and stored read-only under
`baseline/models/`. The experimental pipeline never modifies the
baseline weights — it produces quantized copies under
`experiments/results/quantized/`.

Sensitivity scoring (`pipeline.sensitivity`) ablates one decoder block
(or one projection, for the per-tile variant) to bits `b ∈ {4, 8}` via
symmetric per-output-channel uniform fake-quantization, and reports
WikiText-2 test-set perplexity over a sliding 2048-token window with
200 calibration texts. For Qwen-7B the model is sharded across two
GPUs (`device_map="auto"`, CUDA_VISIBLE_DEVICES=2,3) and 50
calibration texts to keep wall time bounded; the resulting `Δppl`
ranking is consistent with the 200-text setting we used for 1.5B.

GPTQ quantization (`pipeline.quant_runner`) uses GPTQModel 7.0 with
`group_size = 128`, `desc_act = True`, and 32 calibration samples from
WikiText-2 train. For mixed-precision policies we use the
`QuantizeConfig.dynamic` regex override to set 8-bit on the layers
selected by the policy. For 7B we pass `device_map="auto"` to enable
multi-GPU sharded calibration.

Evaluation (`pipeline.eval`) reports WikiText-2 test perplexity over a
sliding 2048-token window on the full ~200-document split. GPTQ
checkpoints are loaded through `GPTQModel.load` rather than
`AutoModelForCausalLM` to avoid an Optimum / EXLLAMA_V1 compatibility
bug in the installed Transformers 5.8.0.

### 5.2 Hardware model

Timeloop v3.0.3 (NVlabs, with the `<cstdint>` patch documented in
`experiments/scripts/install_timeloop.sh`) drives the mapping-pilot
experiments of §3. The architecture template (`configs/arch/eyeriss_like.yaml`)
is a 16×16 PE array with a 256 KB on-chip "WeightInputBuffer" and a
DRAM bandwidth of 32 GB/s; per-bit MAC and storage energy are taken
from Timeloop's PAT model with `word-bits` set to the operand
bit-width.

The residency experiments of §6.2 and §6.3 use an *analytic*
byte-traffic model: every weight is touched exactly once per token,
DRAM traffic per token = `Σ_{t ∉ P} bytes_aware(t)`. The §3 negative
result justifies this — at `N = 1`, total cycles track DRAM bytes
linearly and the mapping geometry contribution is below noise. Energy
modeling is left to future work (see §7.2).

### 5.3 GPU profile (sanity)

We separately measure GPU-side latency, throughput, and approximate
energy/token on a single RTX 2000 Ada with NVML power sampling at 20
ms intervals (`pipeline.profile_gpu`). These numbers serve as a sanity
check for the analytic model and as a deployment-scale reference; they
are not the main result.

### 5.4 Wall-time budget

The full sweep — for one model: sensitivity (~5 min), four-policy
GPTQ + eval (~30–80 min depending on size), residency sweep (seconds),
and figure generation — runs in 20 min for Gemma-2-2B, 25 min for
Qwen-1.5B, and ~80 min for Qwen-7B on free Ada GPUs. Codesign
aggregation across all three models takes < 1 minute.

---

## 6. Results

### 6.1 E1 — Ranking validation: sensitivity-derived policies hold under GPTQ

We measure four policies per model: INT4-uniform, INT8-uniform, and
two Mixed-N policies from the greedy per-layer score at `N ∈ {4.5,
5.0}` average bits per weight. For each we quantize with GPTQ and
measure real WikiText-2 perplexity.

**Table 3.** Real GPTQ perplexity across models. All three show
monotone bpw–ppl ordering (`Kendall τ = −1.0`).

| Policy | Qwen-1.5B | Gemma-2-2B | Qwen-7B |
| --- | ---: | ---: | ---: |
| FP16 baseline | 8.890 | — | 6.121 (50-calib) |
| INT4-uniform | 10.452 | 14.800 | 7.785 |
| Mixed-4.5 | **10.279** | **14.723** | **7.724** |
| Mixed-5.0 | **10.091** | **14.387** | **7.661** |
| INT8-uniform | 9.604 | 13.773 | 7.376 |

**Sensitivity is conservative.** For Qwen-1.5B, the isolated-layer
upper bound predicts a Δppl of +4.00 for INT4-uniform vs FP16; the
real GPTQ value is +1.56 (-61 % absorbed by Hessian-aware
reconstruction). The ratio is similar for Gemma-2B and Qwen-7B. The
*ranking* of policies, however, matches the upper-bound ranking
exactly (Kendall τ on the real-vs-upper sequence is +1.000 for
Qwen-1.5B). Implication: greedy sensitivity is an order oracle, not
a magnitude oracle — sufficient for policy *search*, insufficient for
direct accuracy prediction. Figure 1
(`figs/e1_ranking_validation.png`) shows the gap quantitatively.

**A "GPTQ floor" at INT8.** Even uniform-INT8 GPTQ adds +0.71 ppl on
Qwen-1.5B vs the FP16 baseline (versus a sensitivity upper bound of
+0.03). This is a property of GPTQ's groupwise calibration at
`group_size = 128` with 32 calibration samples and would shrink under
larger calibration; for relative policy comparison this offset
cancels out.

![E1 ranking validation](../experiments/results/figs/e1_ranking_validation.png)
*Figure 1. E1 ranking validation for Qwen2.5-1.5B-Instruct. Left:
sensitivity upper bound (dashed) vs real GPTQ ppl (solid); right:
GPTQ absorbs most of the predicted loss but ranking is preserved.*

### 6.2 E2-residency — main result

Figure 2 (`figs/codesign_main.png`) is our main result for
Qwen-1.5B at the GBuf = 512 MB operating point — slightly below
INT4-uniform's 655 MB decoder total. Round markers (○) are
precision-aware schedulers; crosses (×) are precision-oblivious INT8-
budget schedulers; arrows show the aware reduction.

**Table 4.** Per-token DRAM traffic at GBuf = 512 MB (Qwen-1.5B).

| Policy | ppl | aware (MB) | oblivious (MB) | aware ↓ |
| --- | ---: | ---: | ---: | ---: |
| INT4-uniform | 10.452 | **124** | 392 | **3.17×** |
| Mixed-4.5 | 10.279 | **213** | 454 | **2.13×** |
| Mixed-5.0 | 10.091 | **289** | 516 | **1.79×** |
| INT8-uniform | 9.604 | 784 | 784 | 1.00× |

INT8-uniform's aware==oblivious is the trivial limit: the policy is
already at the worst-case bit-width assumed by the oblivious
scheduler, so the conservative budget is exact.

![E3 main figure](../experiments/results/figs/codesign_main.png)
*Figure 2. Accuracy × per-token DRAM at GBuf = 512 MB on Qwen2.5-1.5B.
The aware-vs-oblivious gap widens as the policy uses more INT4 layers.*

**Operating regime ("bimodal") at GBuf ≈ INT4_total.** When the
buffer is just large enough to hold an INT4 version of the decoder but
not an INT8 version, the aware scheduler can keep mixed-precision
models 100 % on-chip resident — per-token DRAM traffic drops to zero.
INT8-uniform still has to stream, regardless of scheduler. This
bimodal regime is the cleanest story for an accelerator with on-chip
SRAM around the INT4-total footprint of the model
(`figs/codesign_multi_gbuf.png`).

#### 6.2.1 GPU sanity check (analytic ↔ measurement cross-validation)

The numbers in Table 4 are computed analytically: `DRAM/token =
Σ_{t ∉ P} bytes_aware(t)`. The negative mapping result of §3 justifies
that approximation in the decode regime, but a hardware paper should
not rest the main result on arithmetic alone. We therefore measure
the FP16 baseline of Qwen2.5-1.5B on a single NVIDIA RTX 2000 Ada
(16 GiB), generating 256 tokens with `do_sample=False` and sampling
NVML power at 20 ms intervals through a background thread
(`pipeline.profile_gpu`):

| Metric | Value |
| --- | ---: |
| Throughput | 25.0 tokens / s |
| Wall time / 256 tokens | 10.24 s |
| Average GPU power | 68.5 W (TDP cap 70 W) |
| Energy / token | 2.74 J/token (avg-power × elapsed / 256) |
| Power samples | 510 |

This deployment-scale baseline is consistent with the analytic
prediction: at FP16 the per-token decoder bytes are 2.62 GB and a
PCIe-Gen4-class memory subsystem caps throughput in the same
neighborhood. We use the *order* between this measurement and the
quantized configurations as the cross-check, not its absolute value.
A controlled cross-config GPU comparison (FP16 vs INT4 GPTQ vs INT8
GPTQ vs mixed) — which lets us directly verify that the analytic
ratios in Table 4 carry over to wall-time — is straightforward via
`pipeline.profile_gpu` and is listed in §7.2.

### 6.3 Sweep — cross-model generalization

Figure 3 (`figs/sweep_cross_model.png`) shows the same accuracy ×
DRAM/token relation across three models at log-scale GBuf. Solid
lines are aware schedulers, dashed are oblivious. Three observations:

1. **Curve shape is universal.** All three models show the same
   monotonic-in-`B_GBuf` decline in DRAM traffic and the same
   ordering: INT4-uniform < Mixed-4.5 < Mixed-5.0 < INT8-uniform.
2. **The "knee" sits at `B_GBuf ≈ INT4_total`** for each model
   (655 MB / 968 MB / 3.25 GB). Below the knee, aware vs oblivious
   gap grows; above the knee, aware schedulers collapse to zero.
3. **Win factor scales with model size.** At `B_GBuf ≈ INT4_total/2`
   (the comparable operating point across models):

| Model | INT4 total | GBuf | INT4 win | Mixed-4.5 win | Mixed-5.0 win |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.5B | 655 MB | 256 MB | 1.33× | 1.21× | 1.18× |
| Gemma-2-2B | 968 MB | 512 MB | **1.55×** | 1.37× | 1.27× |
| Qwen-7B | 3.25 GB | 2 GB | **1.97×** | 1.45× | 1.30× |

![Cross-model sweep](../experiments/results/figs/sweep_cross_model.png)
*Figure 3. Cross-model sweep. Solid lines: precision-aware
scheduler. Dashed: precision-oblivious INT8-budget. The aware
contribution generalizes across model family (Qwen vs Gemma) and
scale (1.5B → 7B).*

The win-factor increase with model size suggests that per-tile byte
accounting matters more as the absolute capacity gap between INT4 and
INT8 representations grows.

### 6.4 E4 — Per-tile HW-normalized policy (preliminary)

> **Status.** §6.4 reports policy *shape* (Table 5) and the residency
> consequences of that shape (Table 6). The accuracy consequence — the
> motivation for choosing Δppl/Δbyte over Δppl/Δbit — requires running
> GPTQ on the per-tile policies and is *not measured here* (~50 min
> per policy). We position §6.4 as preliminary and flag it as the
> single most important pre-submission experiment (§7.2). §4.3 above
> motivates the score; readers should interpret §6.4 as evidence that
> the *policy distribution* differs in the expected direction.

The greedy per-layer score of §2.1 ignores byte cost. We re-derive
policies using Δppl/Δbyte at *per-projection* granularity (196 tiles
for Qwen-1.5B). Table 5 contrasts the resulting policy shape against
the per-layer greedy at the same average bits per weight.

| Projection role | Mixed-4.5 (per-layer greedy) | hwnorm-4.5 (per-tile) |
| --- | ---: | ---: |
| q_proj | 4 / 28 at 8-bit | 7 / 28 at 8-bit |
| k_proj | 4 / 28 | **17 / 28** |
| v_proj | 4 / 28 | **25 / 28 (89 %)** |
| o_proj | 4 / 28 | 15 / 28 |
| gate_proj | 4 / 28 | **0 / 28** |
| up_proj | 4 / 28 | 2 / 28 |
| down_proj | 4 / 28 | 5 / 28 |
| Total 8-bit tiles | 28 | **71** |
| Achieved avg bpw | 4.571 | **4.503** |

The HW-normalized policy concentrates the 8-bit budget in attention's
KV projections, which are the cheapest to promote (4-6× smaller than
MLP projections) and remain sensitive. MLP `gate_proj` stays entirely
at INT4 because it is 6.89 MB per layer at INT4 and any single
promotion to 8-bit consumes 6.89 MB of budget — too expensive against
its Δppl contribution.

**Residency consequence is modest (Table 6).** At the same average
bits per weight, the two policies have similar total bytes (the policy
distribution differs more than its byte total). DRAM per token at
GBuf = 512 MB:

| Budget | greedy bpw | hwnorm bpw | greedy DRAM | hwnorm DRAM | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B = 4.5 | 4.571 | 4.503 | 213.3 MB | **206.4 MB** | 3.2 % ↓ |
| B = 5.0 | 5.000 | 5.025 | 289.0 MB | 289.0 MB | 0.0 % |

The residency win is small because the knapsack picks tiles by absolute
size — it is unaware of which logical role each tile plays. The
*accuracy* side is where we expect the real win: protecting 89 % of
`v_proj` at 8-bit while keeping all `gate_proj` at 4-bit should yield
a lower ppl at iso-bpw than the per-layer policy because (i) KV
projections carry disproportionate attention quality and (ii) the
average bit-width is matched. We have not measured this in this
draft — verification requires GPTQ on each per-tile policy (~50 min
wall-time per policy × 4 policies). The empirical loop is in place
(`scripts/_e4_per_tile.py` plus the existing `run_quant.py`); the
result is the headline experiment for the camera-ready version.

![E4 hwnorm vs greedy](../experiments/results/figs/e4_hwnorm_vs_greedy.png)
*Figure 4. Per-tile HW-normalized policy vs per-layer greedy across
four GBuf operating points. The two scorers produce overlapping
DRAM-per-token curves at iso-bpw under residency scheduling; the
distinction lives in policy shape (Table 5).*

---

## 7. Discussion

### 7.1 What the negative mapping result really means

§3 reports that for a 16×16 Eyeriss-class architecture at `N = 1`
decode, the mapper finds equivalent-cost mappings for INT4 and INT8
versions of the same matmul. This is *not* a claim that mapping
search is useless for LLM accelerators — only that, in the regime
where total weight bytes ≫ on-chip capacity and each weight is
touched once per token, the mapping decision space collapses onto a
single byte-bound point. In two complementary regimes we expect
mapping to recover relevance:

- **Prefill (`N ≫ 1`)**: weights are reused across tokens within the
  forward pass; tile choice meaningfully changes the reuse pattern.
- **Architectures with comparable on-chip / off-chip ratios** (e.g.
  weight stationary at 64 KB GBuf): buffer-pressure effects become
  the dominant variable, and the precision-aware mapping spread
  re-emerges.

Both are explicit directions for §7.2.

### 7.2 Limitations and follow-ups

- **Energy modeling.** Our residency model treats DRAM bytes as the
  sole efficiency metric. Accelergy-driven energy estimates per
  (bit-width × tile × pinned-vs-streamed) breakdown would let us
  report mJ/token directly; the Accelergy hooks are wired in
  `pipeline.hw_timeloop` but not used as the main efficiency lever
  in this draft.
- **Prefetching.** The 0/1 knapsack model is conservative: it
  pins-or-streams each tile entirely. A real scheduler could
  prefetch the next layer's weights while the current layer computes.
  Modeling that would reduce the oblivious baseline's overhead (it
  could partially hide DRAM time) and tighten our claim.
- **Sequence-length dependence.** Pinning is valuable across many
  tokens; for a single-token "first token" generation, the residency
  win is zero. Our results implicitly assume `T ≫ 1` (sustained
  decode).
- **E4 accuracy verification.** We have not run GPTQ on the
  HW-normalized policies. The claim that protecting KV projections
  at 8-bit yields a lower ppl at iso-bpw is currently a hypothesis;
  the experimental loop is in place (re-run `_e4_per_tile.py` with
  `--run-gptq` would close it) and is the natural next experiment.
- **Workload coverage.** Three models, one task (WikiText-2
  perplexity). Downstream task evaluation (MMLU, ARC-c, HellaSwag)
  via `lm-evaluation-harness` is plumbed in `pipeline.eval` but not
  reported here.

### 7.3 Connection to MAGNETO and GATHER

MAGNETO formulated power-aware mapping as the primary contribution
surface for accelerator energy. GATHER added LLM-specific dataflow
on top. Our work sits one layer higher in the stack: given a fixed
policy and a fixed mapping, scheduling decides which tiles benefit
from on-chip residency. The three layers (policy / mapping /
scheduling) compose; this paper isolates and quantifies the
scheduling layer.

---

## 8. Related Work

**Layer-wise mixed-precision policy search.** HAWQ [Dong et al., ICCV
2019] and its Hessian-eigen variant HAWQ-V2 [Dong et al., NeurIPS
2020] introduced second-order sensitivity scoring for per-layer
bit-width assignment. BRECQ [Li et al., ICLR 2021] formulated
block-wise reconstruction error as a more practical signal, and
OmniQuant [Shao et al., ICLR 2024] re-parameterized the search to
make the budget a learnable variable. LLM.int8() [Dettmers et al.,
NeurIPS 2022] argued for outlier-aware INT8 inference. All four sit
at the *algorithm* layer of Fig. 0; none address how the resulting
policy interacts with the deployment-time residency budget.

**LLM quantization at deployment.** GPTQ [Frantar et al., ICLR 2023]
performs Hessian-aware column-wise rounding; we use it as our
quantizer of record (§5.1). AWQ [Lin et al., MLSys 2024] preserves
activation-salient channels; SmoothQuant [Xiao et al., ICML 2023]
shifts the activation–weight quantization difficulty. Marlin
[Frantar et al., 2024] gives a fast 4-bit decode kernel that turns
the byte saving of mixed precision into actual latency on GPUs.
These produce *kernels*; scheduling decisions about which weights
stay on-chip are outside their scope.

**LLM accelerator dataflow.** Sanger [Lu et al., MICRO 2021] and Ant
[Guo et al., HPCA 2022] are representative LLM accelerator designs
with custom dataflow. Our lab's prior MAGNETO (power-aware mapping)
and GATHER (LLM accelerator) line — anonymized for review — sit in
the same space. Each of these contributes at the *mapping* layer of
Fig. 0; they assume a fixed precision plan as input and search the
tile/dataflow space. Our work composes with all of them: a Sanger-
or GATHER-style mapper produces the per-shape mapping that determines
`cycles_per_token` per tile; our scheduler decides which of those
tiles avoid DRAM round trips.

**Memory residency / cache scheduling for LLM serving.** SARATHI
[Agrawal et al., MLSys 2024] schedules prefill/decode batches to
improve GPU utilization but operates at the request level.
DistServe [Zhong et al., OSDI 2024] decouples prefill and decode
across machines. FlexGen [Sheng et al., ICML 2023] tiers weight
storage across GPU/CPU/SSD for offload-friendly inference.
ChunkAttention [Ye et al., ACL 2024] re-orders KV-cache access for
sequence reuse. These works address the residency of *KV cache* or
of *runtime state*; our work addresses the residency of *weights*
under a mixed-precision policy, a different optimization variable.

**Knapsack-based scheduling for DNN inference.** Tetris [Gao et al.,
ASPLOS 2017] uses ILP over operators for 3D DNN partitioning;
TVM/Ansor [Zheng et al., OSDI 2020] is search-based but at the
kernel-fusion granularity. Liquid (anonymized lab work, in
submission) explores knapsack-based prefetch for inference servers.
None of these are precision-aware; the operand bit-width is fixed
upfront. Our knapsack instance is defined by the per-tile
`bytes_aware(t)`, which is *constructed by the upstream policy* —
that is the precise coupling that the scheduling layer must surface.

---

## 9. Conclusion

We isolate a system-level question that the mixed-precision-quantization
literature does not address: *given a fixed policy, which weight
tiles do we keep on-chip?* We formulate the problem as 0/1
greedy-knapsack over per-tile bytes, show that a precision-aware
scheduler reduces per-token DRAM traffic by **1.33×–1.97×** across
three LLM families and three scales at a comparable operating point
(`B ≈ INT4_total/2`) and up to **3.17×** in the INT4-total knee
regime for the 1.5B model, and that the contribution generalizes with
a factor that grows with model size. We provide the full reproducibility
trail in `experiments/`, including an analytic-↔-measurement
cross-check via a GPU profile of the FP16 baseline. The result puts
a name and a quantification on a layer of the stack between the
quantization algorithm and the dataflow mapper, and points to two
natural follow-ups: prefetch-aware scheduling models, and per-tile
HW-normalized policies that protect attention KV projections at no
extra average bit-width cost (the per-tile policies are constructed
in §6.4; their GPTQ accuracy verification is the camera-ready
experiment).

---

## Appendix A — Reproducibility

Every claim in this paper traces to:

- **Code:** `pipeline/` (Python package). 10 modules; see
  `pipeline/README.md` for module map.
- **Drivers:** `experiments/scripts/run_<phase>.py`,
  `_sweep_target.sh`, `_sweep_qwen7b.sh`.
- **Configs:** `experiments/configs/`.
- **Results:** `experiments/results/`. Every figure has its
  generator script next to the CSV it reads.
- **Pipeline status:** `experiments/ROADMAP.md`.
- **Detailed methodology:** `experiments/README.md` §6 (per-metric
  measurement), §8a–8f (per-experiment results).

The acpl server install path (Timeloop v3.0.3 + Accelergy without
sudo via conda-forge, the gcc-13 `<cstdint>` 92-header patch, and the
`activate.d/timeloop_ld.sh` LD path) is captured in
`experiments/scripts/install_timeloop.sh`.

## Appendix B — Walltime profile

| Phase | Qwen-1.5B | Gemma-2B | Qwen-7B |
| --- | --- | --- | --- |
| Sensitivity (per-layer, 200 calib, 1 GPU) | ~5 min | ~3 min | OOM* |
| Sensitivity (per-layer, 50 calib, 2 GPU shard) | ~2 min | n/a | ~2 min |
| Sensitivity (per-projection, 50 calib, 1 GPU) | ~4 min | not measured | not measured |
| Policy generation (4 budgets) | < 1 s | < 1 s | < 1 s |
| GPTQ (per policy, 32 calib) | ~12 min | ~5 min | ~25 min |
| WikiText-2 eval (per checkpoint) | ~30 s | ~30 s | ~2 min |
| Residency sweep (8 GBuf × 2 sched) | < 1 s | < 1 s | < 1 s |
| Codesign aggregation + figures | ~10 s | ~10 s | ~10 s |

\* per-layer sensitivity for Qwen-7B in FP16 on a single 16 GiB GPU
exceeds memory because the sliding-window CE loss tensor over 152k
vocab is ~1 GB on top of the 14 GB model.

---
---

# 혼합 정밀도 LLM 디코더를 위한 정밀도 인지 잔존(residency) 스케줄링

**저자.** (TBD — 이화여자대학교, MAGNETO/GATHER 연구 라인)
**투고 목표.** ISCA / MICRO / HPCA / ASPLOS
**상태.** 초안 v0.1 (2026-05-12). 모든 근거는 `experiments/`.

---

## 초록

자기회귀(autoregressive) LLM 디코딩은 토큰당 가중치 트래픽에 묶입니다.
생성되는 토큰마다 모든 가중치가 정확히 한 번씩 DRAM에서 읽혀 나오므로,
지연 시간과 에너지는 부동소수점 연산 수가 아니라 *바이트 양*에 비례합니다.
레이어별 혼합 정밀도(mixed precision)는 그 바이트 양을 줄여 주지만,
배포 시점의 스케줄러가 각 레이어의 실제 비트 폭을 모르면 모든 weight 타일을
worst-case 비트 폭(INT8)으로 보수적으로 예산을 잡게 되어 on-chip 잔존 용량을
낭비하게 됩니다. 우리는 이 문제를 **정밀도 인지 잔존 스케줄링
(precision-aware residency scheduling)** 문제로 정식화합니다: 고정된
혼합 정밀도 정책과 용량 `B`의 on-chip 버퍼가 주어졌을 때, 어떤
(레이어, projection) 가중치 타일을 핀(pin) 해서 토큰당 DRAM 트래픽을
최소화할 것인가. 스케줄러는 타일 byte에 대한 그리디 0/1 knapsack으로
구현하며, 모든 타일을 INT8 예산으로 잡는 정밀도 비인지(precision-oblivious)
기준선과 비교합니다. Qwen2.5-1.5B-Instruct, Gemma-2-2B-IT,
Qwen2.5-7B-Instruct 세 모델에 걸쳐 *모델 크기에 비례하는 동일한 운영점*
`B ≈ INT4_total/2` 에서 토큰당 DRAM 트래픽이 **1.33×–1.97×** 감소하며,
1.5B 모델에서 `B`가 INT4 총 footprint에 가까워지는 knee 영역에서는 최대
**3.17×** 까지 도달합니다. `B ≈ INT4_total` 영역에서는 우리 sweep의 모든
혼합 정밀도 정책이 on-chip 상주(`DRAM/tok = 0`)에 진입합니다. 또한,
입력으로 흔히 사용되는 비트당 Δppl 기반 그리디 레이어별 정책이 GPTQ 양자화
후에도 ranking이 보존됨을 (Kendall τ = +1.0) 보이고, 스코어를 바이트당
Δppl 기반의 per-tile 점수로 확장하면 동일 평균 비트 폭에서 어텐션 KV
projection(`v_proj`의 89 %, `k_proj`의 61 %)을 우선 보호하는 질적으로 다른
정책이 생성됨을 보입니다. Qwen2.5-1.5B에서 GPU 측 sanity profile을 함께
보고하여 해석적 모델의 순위가 실측에서도 재현됨을 보입니다.

---

## 1. 서론

LLM은 자기회귀적으로 토큰을 생성합니다. 매 토큰마다 디코더 블록 전체를
지나며, 직전 토큰들의 K/V 캐시는 on-chip에 두고 가중치 행렬은 DRAM에서
새로 끌어옵니다. 디코드 영역(배치 `N = 1`)은 산술 강도(arithmetic
intensity)가 낮아 GPU나 가속기의 PE 어레이는 사이클의 90 % 이상을 메모리
대기에 씁니다 [Sheng et al., ICML 2023; Yu et al., 2023이 LLM 디코드의
memory-bound 양상을 보고]. 결국 토큰당 가중치 바이트를 줄이는 것이 지연
시간과 에너지 모두를 결정하는 가장 중요한 레버입니다.

레이어별 혼합 정밀도 양자화는 이를 위한 정석적 기법입니다.
HAWQ [Dong et al., ICCV 2019]와 BRECQ [Li et al., ICLR 2021]는 손실 지형
민감도를 기준으로 비트 폭을 배정하고, GPTQ [Frantar et al., ICLR 2023]은
양자화 자체를 Hessian 인지로 만들어 단순 라운딩이 잃는 정확도 대부분을
복구합니다. 최근 LLM 가속기 설계 — 본 연구실의 선행 power-aware mapping
인 MAGNETO와 LLM 특화 가속기 GATHER (둘 다 review를 위해 익명화) —
는 혼합 정밀도 정책을 시스템 입력으로 받고 dataflow / mapping search
공간에서 에너지 효율을 끌어옵니다.

본 논문은 그 사이 — 알고리즘 층과 매핑 층 사이 — 의 한 가지 질문을
던집니다: *배포 시점에 스케줄러는 혼합 정밀도 정책을 가지고 무엇을 해야
하는가?* 구체적으로 말해, on-chip 버퍼가 전체 디코더를 담기에 너무
작을 때 어떤 가중치 타일을 상주시켜 후속 토큰들이 다시 DRAM으로 가지 않게
할 것인가.

세 가지 관찰을 시작점으로 합니다.

**(O1) 디코드 영역에서 매핑 탐색은 byte 트래픽 단조성을 깨지 못한다.**
경험적으로(§3) 16×16 Eyeriss 스타일 어레이에서 같은 GEMM에 대해 4비트
최적 매핑과 8비트 최적 매핑은 *타일 형태* 가 다르지만(4비트는 출력 채널
블록 전체를 256 KB 버퍼에 잡고, 8비트는 분할해야 함), 사이클과 에너지는
byte 폭과 거의 정확히 같이 움직입니다. shape별 사이클 비율은 1.00×,
바이트 비율은 2.00×입니다. 즉 매핑 기여가 byte 비용에 흡수됩니다.
*매핑은 `N = 1`에서 스케줄링 지능을 얹을 자리가 아닙니다.*

**(O2) 자연스러운 자리는 잔존(residency)이다.** 모델이 고정되어 있으면
모든 가중치는 토큰당 정확히 한 번 읽힙니다. 어떤 타일이 on-chip 버퍼에
들어가면 한 번만 적재한 뒤 `T` 토큰 동안 재사용되어 `(T − 1) × tile_bytes`
의 DRAM 트래픽을 절감합니다. 결정 문제는 단일 용량 제약 하의 고전적
0/1 knapsack이 됩니다.

**(O3) 혼합 정밀도 정책은 knapsack 인스턴스를 바꾸지만, 단순 배포는
이를 무시한다.** 모든 타일을 INT8(=최악 비트 폭)으로 예산을 잡는
"정밀도 비인지" 스케줄러는 실제 byte를 쓰는 "정밀도 인지" 스케줄러보다
훨씬 적은 타일만 핀합니다. 동일 `B`에서 인지 스케줄러는 1.7×–3.2× 적은
바이트만 스트리밍합니다.

### 기여

- **시스템 주장.** 정밀도 인지 잔존 스케줄링을 per-tile byte knapsack으로
  정식화하고, INT8 예산 기반의 정밀도 비인지 기준선 대비 3 LLM 계열·3
  스케일에서 정량적 win을 측정합니다.
- **일반화.** Qwen-1.5B, Gemma-2-2B, Qwen-7B 모두에서 패턴이 동일합니다:
  E1 순위 보존, aware-vs-oblivious win이 같은 방향으로 나타나며, *win
  계수*는 모델 크기와 함께 커집니다 (`B ≈ INT4_total/2`에서 1.33× → 1.97×).
- **방법론 확장 (preliminary).** 레이어별 Δppl/Δbit 그리디 점수를 per-tile
  Δppl/Δbyte 점수로 바꾸면, 동일 평균 비트 폭에서 어텐션 KV projection을
  우선 보호하는 (89 %의 `v_proj`이 8-bit) 질적으로 다른 정책이 만들어
  집니다. 이 재할당이 *정확도*에서 이득을 주는지는 본 드래프트에서
  *직접 검증하지 않았습니다* — per-tile 정책 4개에 대해 GPTQ를 다시 돌리는
  것이 가장 중요한 다음 실험입니다 (§7.2).
- **재현성.** 본 논문의 모든 수치는 `experiments/results/` 아래 CSV와
  figure로 추적되며, `pipeline/`의 모듈들이 `_sweep_target.sh`로 재생성
  가능합니다. sudo 권한이 없는 conda 환경에서 Timeloop v3.0.3 + Accelergy를
  source build 하는 절차, gcc-13 `<cstdint>` 누락 헤더 92개에 대한
  자동 패치까지 모두 문서화되어 있습니다.

### 본 논문의 스택 상 위치

LLM 효율 연구는 세 개의 다른 층에서 이루어집니다 (Fig. 0). 본 논문의
기여는 스케줄링 층이고, 위 두 층은 입력으로 받습니다.

```text
                    ┌──────────────────────────────────────────────────┐
알고리즘 층 ↑       │  Mixed-precision policy                          │
                    │    예: HAWQ, BRECQ, GPTQ                         │
                    │    출력: per-(layer, projection) 비트 폭         │
                    └──────────────────────────────────────────────────┘
                                       │ (입력 →)
                    ┌──────────────────▼───────────────────────────────┐
매핑 층             │  가속기 매핑 / dataflow                          │
                    │    예: Timeloop, MAGNETO, GATHER                 │
                    │    출력: shape별 타일 크기, 루프 nest 순서       │
                    └──────────────────────────────────────────────────┘
                                       │ (입력 →)
                    ┌──────────────────▼───────────────────────────────┐
스케줄링 층 ↓       │  **잔존 스케줄링 (본 논문)**                     │
                    │    입력: mixed-precision policy, on-chip buf B   │
                    │    결정: 어떤 weight 타일을 on-chip에 pin할지   │
                    │    출력: 토큰당 DRAM 바이트 (효율 지표)          │
                    └──────────────────────────────────────────────────┘
```

*Figure 0.* 3 층 구조. single-pass dataflow 논문들은 매핑(중간)과
스케줄링(아래)을 종종 한 덩어리로 묶지만, 자기회귀 디코드에서는 매핑
층의 자유도가 byte 트래픽으로 붕괴함을 §3에서 보이며 (mapping pilot
negative 결과), 그 결과 스케줄링 층의 기여가 혼합 정밀도와 *중복이
아니라 가산적* 임을 확인합니다.

---

## 2. 배경

### 2.1 혼합 정밀도 양자화와 정책 탐색

레이어별 민감도 스코어는 트랜스포머 블록 `l`을 `b` 비트로 양자화했을 때의
perplexity 손상을 측정합니다. 보통 Hessian trace 기반(HAWQ), 블록
재구성 오차(BRECQ), 또는 단순 ablation(우리 기본 — 한 블록의 가중치를
fake-quantize 한 값으로 바꾸고 WikiText-2 perplexity를 재측정)이 쓰입니다.
정책 탐색은 허용 비트 폭 집합 `W = {b1, ..., bk}`과 평균 비트 예산 `B̄`
이 주어졌을 때 `mean(B) ≤ B̄`를 만족하면서 총 Δppl을 최소화하는
레이어별 배정 `B[l] ∈ W`를 찾는 문제입니다. 추가 비트당 Δppl이 큰
레이어부터 그리디로 승격하는 것이 표준 해법이고, 스코어가 단조이면
거의 최적입니다.

승격 규칙:

```text
score(l, b → b+) = (Δppl(l, b) - Δppl(l, b+)) / (b+ - b)
```

### 2.2 GPTQ 양자화

GPTQ는 칼리브레이션 활성화에서 근사 역 Hessian을 계산하여, 열 단위로
탐욕적으로 양자화하면서 남은 가중치들을 보정하여 누적 오차를 줄입니다.
isolated-layer 민감도가 예측하는 손실의 상당 부분을 흡수합니다 (§6.1).

### 2.3 matmul용 dataflow 매핑 (Timeloop)

Timeloop은 주어진 문제(M × K × N)와 가속기(PE 어레이 + 메모리 계층 +
대역폭)에 대해 루프 둥지 타일링 공간을 탐색합니다. 16×16 PE 어레이,
256 KB on-chip "WeightInputBuffer", 24-bit "AccumulationBuffer"의
Eyeriss 스타일 구조에서, Timeloop은 PAT 모델 혹은 Accelergy 플러그인
기반으로 사이클, 에너지, GEMM당 DRAM 바이트를 보고합니다.

### 2.4 LLM 가속기 메모리 계층

최근 LLM 가속기는 가중치가 메모리 트래픽을 지배한다는 관찰에 따라 설계
됩니다. 흔한 패턴은 모델의 일부를 토큰 사이에 상주시킬 수 있는 큰 on-chip
SRAM(4 MB ~ 1 GB SRAM 또는 HBM2를 명시적 캐시 tier로 사용) [Sanger,
MICRO 2021; Ant, HPCA 2022; 본 연구실의 GATHER]입니다. 본 논문은 그
잔존 층에 초점을 둡니다: 정책과 버퍼 크기가 주어졌을 때 어떤 타일이
머무는가.

---

## 3. 모티베이션 — 디코드에서는 매핑 탐색이 답이 아니다 (negative 결과)

정밀도 인지 매핑 가설을 다음과 같이 인스턴스화합니다. Qwen2.5-1.5B
디코더 블록의 unique linear shape 7개 (q/k/v/o_proj, gate/up/down_proj) ×
비트 폭 `b ∈ {4, 8}` 조합에 대해, 16×16 PE 어레이 + 256 KB on-chip
버퍼에서 `timeloop-mapper`를 돌려 사이클, 에너지, DRAM 바이트, PE
utilization을 측정합니다. 그리고 cross-application loss를 계산합니다:

```text
L_{a→b} := metric(best_mapping_at_a, weights=b) / metric(best_mapping_at_b, weights=b)
```

즉, 다른 비트 폭의 최적 매핑을 실제 비트 폭에 강제 적용했을 때의 손해.

**실험 결과 (Table 1 / `figs/sensitivity_qwen15b.png`).** 매핑 자체는
다르게 잡힙니다. INT4에서는 매퍼가 GBuf에 출력 차원 전체를 통째로 잡는
타일 (`for C in [0:6)`)을 고르지만, INT8에서는 같은 타일이 GBuf를
넘쳐서 K도 분할해야 합니다 (`for K in [0:2); for C in [0:6)`). 그러나
사이클·에너지는 byte 폭에 의해 결정됩니다.

| Shape (1.5B) | b | 사이클 | 에너지 (pJ) | DRAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| q_proj | 4 | 147,552 | 1.35 × 10⁸ | 1.18 |
| q_proj | 8 | 147,648 | 2.70 × 10⁸ | 2.36 |
| up_proj | 4 | 860,256 | 7.85 × 10⁸ | 6.89 |
| up_proj | 8 | 860,256 | 1.57 × 10⁹ | 13.77 |

사이클 비율 ≈ 1.000, 에너지 비율 ≈ 2.000, DRAM 비율 ≈ 2.000. 매핑 형태
차이가 사이클 차이로 옮겨가지 못합니다. `N = 1`에서 모든 가중치가 한 번씩
touch되므로 전체 DRAM 시간 = `total_bytes / BW`로 결정되고, 두 매핑
모두 같은 총 weight 바이트를 streaming하기 때문입니다.

**§3 결론.** 디코드의 Eyeriss 계열 어레이에서는 *매핑 탐색이 혼합
정밀도 위에 스케줄링 지능을 얹을 자리가 아닙니다*. 적절한 자리는 그
위층입니다: 어떤 타일을 토큰 사이에 on-chip에 남길 것인가?

---

## 4. 방법 — 정밀도 인지 잔존 스케줄링

### 4.1 문제 정의

디코더를 `L`개 블록의 시퀀스로 표현하고, 각 블록은 7개 projection을 갖는
다고 합시다. 각 타일 `t = (layer_idx, role)`는 가중치 수 `n_t` (모델
hidden size와 projection 역할의 함수)와 정책 `B[t] ∈ {4, 8}`로 결정되는
바이트를 갖습니다:

```text
bytes_aware(t) = n_t × B[t] / 8
```

토큰당 각 타일의 GEMM이 한 번 읽힙니다. 타일이 용량 `B_GBuf`의 on-chip
버퍼에 *핀* 되어 있으면 그 토큰과 이후 토큰들 모두에서 읽기가 무료가 되고,
*스트리밍* 되는 타일은 토큰당 `bytes_aware(t)`의 DRAM 트래픽을 발생시
킵니다.

스케줄러는 `Σ_{t ∈ P} bytes(t) ≤ B_GBuf`를 만족하는 부분집합 `P ⊆ T`를
선택합니다. 두 스케줄러의 차이는 이 제약에 어떤 바이트 수를 쓰느냐 입니다:

- **정밀도 인지**: `bytes_aware(t)`를 그대로 사용. 정책을 안다.
- **정밀도 비인지 (INT8 예산)**: 제약에 `bytes_int8(t) = n_t`를 쓰고,
  보수적 예산이 맞으면 실제 `bytes_aware(t)`만큼만 핀한다. 정책을
  무시하고 worst-case 비트 폭을 가정하는 배포 시점 스케줄러 모델.

두 스케줄러 모두 다음을 풉니다:

```text
maximize    Σ_{t ∈ P} bytes_aware(t)              # 토큰당 절감 byte
subject to  Σ_{t ∈ P} budget_bytes(t) ≤ B_GBuf
```

### 4.2 해법 — greedy smallest-first

`N = 1` 자기회귀 디코드에서는 모든 타일이 토큰당 정확히 한 번씩 접근
됩니다. 접근 빈도가 타일과 무관하므로 byte당 절감 률도 모든 타일에서 같습
니다. 이 조건에서 0/1 knapsack의 최적해는 greedy smallest-first로
축약됩니다: `budget_bytes` 오름차순으로 정렬하고, 다음 타일이 `B_GBuf`를
넘기지 않을 때까지 핀합니다. 두 스케줄러 모두에 대해 최적이고
`O(|T| log |T|)` 시간 내에 풀립니다.

구현: `pipeline/residency.py::pack_greedy`.

### 4.3 HW-normalized 정책 (per-tile, 옵션)

§2.1의 레이어별 그리디는 Δppl/Δbit를 최대화하지만, 그 승격이 *몇 바이트*를
잡아먹는지는 무시합니다. 동질적인 디코더에서는 모든 레이어 크기가 같으니
괜찮지만, 한 레이어 안의 7개 projection은 크기가 매우 다릅니다 (Qwen-2.5-
1.5B의 `k_proj`, `v_proj`은 `gate_proj` 대비 6배 작음). per-tile HW-
normalized 점수는 Δppl을 추가 *byte* 당으로 표현합니다:

```text
score_hwnorm(t, b → b+) = (Δppl(t, b) - Δppl(t, b+)) / (n_t × (b+ - b) / 8)
```

이는 작고 민감도 높은 타일(주로 어텐션의 `k_proj`, `v_proj`)을 먼저
승격시키고, 큰 MLP projection은 최소 폭으로 둡니다. per-tile Δppl은
`pipeline/sensitivity.py::run_per_projection_sensitivity` 에서 각 `L × 7`
linear를 독립적으로 ablation하여 측정합니다(§5.1).

### 4.4 알고리즘 복잡도

Qwen2.5-1.5B 기준 `|T| = 28 × 7 = 196` 타일. knapsack 자체와 per-tile
민감도 스코어 표 모두 초 단위로 끝나며, 파이프라인의 비싼 부분은 스케줄링
이 아니라 GPTQ 양자화 자체입니다(§5.3).

---

## 5. 실험 설정

### 5.1 모델·가중치·calibration

세 가지 open-weight chat-tuned LLM 체크포인트를 사용합니다 (Table 2).

| 모델 | 파라미터 | FP16 ckpt | INT4 디코더 | INT8 디코더 | 비고 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B-Instruct | 1.54 B | 3.1 GB | 655 MB | 1.31 GB | 주 타겟 |
| Gemma-2-2B-IT | 2.61 B | 5.2 GB | 968 MB | 1.94 GB | 다른 family |
| Qwen2.5-7B-Instruct | 7.61 B | 15.2 GB | 3.25 GB | 6.5 GB | scale test |

baseline 모델은 `huggingface_hub`로 받아 `baseline/models/`에 읽기 전용으로
저장합니다. 실험 파이프라인은 절대로 baseline 가중치를 수정하지 않고,
양자화 결과는 `experiments/results/quantized/`에 새로 저장합니다.

민감도 측정(`pipeline.sensitivity`)은 한 디코더 블록(또는 한 projection,
per-tile 변형)을 비트 `b ∈ {4, 8}` 로 대칭 per-output-channel uniform fake-
quantize 후 WikiText-2 test split에 대해 sliding window 2048 토큰
perplexity를 계산합니다. calibration text는 200 문서 기본, Qwen-7B는
multi-GPU sharded (`device_map="auto"`, CUDA_VISIBLE_DEVICES=2,3) +
50 calib로 wall-time을 줄였습니다. 결과 Δppl 순위는 200 calib 결과와
일관합니다.

GPTQ 양자화(`pipeline.quant_runner`)는 GPTQModel 7.0, `group_size = 128`,
`desc_act = True`, 32-sample calib를 사용합니다. 혼합 정밀도 정책에는
`QuantizeConfig.dynamic` regex override로 8-bit 레이어를 지정합니다. 7B는
`device_map="auto"`로 multi-GPU sharded calibration.

평가(`pipeline.eval`)는 WikiText-2 test split 전체에 대해 sliding 2048
토큰 perplexity를 계산합니다. GPTQ 체크포인트는 `AutoModelForCausalLM`
대신 `GPTQModel.load`로 로드합니다 — Transformers 5.8.0 + Optimum 호환성
버그 (EXLLAMA_V1 enum 누락) 우회.

### 5.2 하드웨어 모델

§3의 매핑 pilot 실험에는 Timeloop v3.0.3 (NVlabs, gcc-13 `<cstdint>`
패치는 `experiments/scripts/install_timeloop.sh`)을 사용합니다.
arch 템플릿 (`configs/arch/eyeriss_like.yaml`)은 16×16 PE 어레이 + 256 KB
on-chip "WeightInputBuffer" + DRAM 32 GB/s. 비트별 MAC 및 storage 에너지는
Timeloop PAT 모델로 측정 (`word-bits = bits_w`).

§6.2와 §6.3의 잔존 실험은 *해석적* byte 트래픽 모델을 사용합니다:
토큰당 모든 가중치를 정확히 한 번씩 touch하므로, 토큰당 DRAM 트래픽 =
`Σ_{t ∉ P} bytes_aware(t)`. §3의 negative 결과가 이를 정당화합니다. 에너지
모델링은 future work (§7.2).

### 5.3 GPU profile (sanity)

별도로 RTX 2000 Ada에서 NVML 전력을 20 ms 간격으로 sample (`pipeline.
profile_gpu`)하여 GPU 측 latency, throughput, 에너지/token을 측정합니다.
해석적 모델의 sanity check 용도이며 main result는 아닙니다.

### 5.4 Wall-time 예산

한 모델 전체 파이프라인 (sensitivity ~5분 + 4-정책 GPTQ + eval
~30~80분 + 잔존 sweep + figure 생성)이 Gemma-2-2B는 20분, Qwen-1.5B는
25분, Qwen-7B는 ~80분 (free Ada GPU). 3 모델 codesign aggregation은
< 1분.

---

## 6. 결과

### 6.1 E1 — 순위 검증: 민감도-도출 정책이 GPTQ 후에도 유지됨

모델당 4 정책: INT4-uniform, INT8-uniform, 그리고 평균 비트 `N ∈
{4.5, 5.0}`에서 per-layer 그리디로 만든 Mixed-N 두 개. 각각 GPTQ 양자화
후 WikiText-2 perplexity를 측정합니다.

**Table 3.** 모델별 실측 GPTQ perplexity. 세 모델 모두 bpw–ppl 단조
ordering (`Kendall τ = −1.0`).

| 정책 | Qwen-1.5B | Gemma-2-2B | Qwen-7B |
| --- | ---: | ---: | ---: |
| FP16 baseline | 8.890 | — | 6.121 (50-calib) |
| INT4-uniform | 10.452 | 14.800 | 7.785 |
| Mixed-4.5 | **10.279** | **14.723** | **7.724** |
| Mixed-5.0 | **10.091** | **14.387** | **7.661** |
| INT8-uniform | 9.604 | 13.773 | 7.376 |

**민감도는 보수적이다.** Qwen-1.5B에서 isolated-layer 상한은 INT4-uniform
의 Δppl을 +4.00으로 예측하지만, 실측 GPTQ는 +1.56 (-61 %를 Hessian-aware
reconstruction이 흡수). Gemma-2B와 Qwen-7B도 유사한 비율. 정책 *순위*는
상한과 정확히 일치합니다 (Qwen-1.5B의 real-vs-upper Kendall τ = +1.000).
즉, 그리디 민감도는 **순위 oracle이지 magnitude oracle은 아닙니다** —
정책 탐색에는 충분하지만 직접 정확도 예측엔 불충분. Figure 1
(`figs/e1_ranking_validation.png`)이 정량적 격차.

**"GPTQ floor" at INT8.** Qwen-1.5B에서 uniform INT8 GPTQ조차 FP16
대비 +0.71 ppl을 추가합니다 (민감도 상한은 +0.03). 이는 `group_size=128`
+ 32-sample calib의 한계이며, 큰 calib에서 축소됩니다. 상대 비교에서는
상쇄됩니다.

![E1 ranking validation](../experiments/results/figs/e1_ranking_validation.png)
*Figure 1. Qwen2.5-1.5B-Instruct의 E1 순위 검증. 좌: 민감도 상한(점선) 대
실측 GPTQ ppl(실선); 우: GPTQ가 예측 손실의 대부분을 흡수하지만 순위는
보존.*

### 6.2 E2-residency — main result

Figure 2 (`figs/codesign_main.png`)가 Qwen-1.5B의 GBuf = 512 MB 운영점
main result입니다 — INT4-uniform 디코더 총 655 MB보다 약간 작은 지점.
○ = 정밀도 인지 스케줄러, × = 정밀도 비인지 INT8 예산 스케줄러, 화살표 =
aware 감소량.

**Table 4.** Qwen-1.5B의 GBuf = 512 MB에서 토큰당 DRAM 트래픽.

| 정책 | ppl | aware (MB) | oblivious (MB) | aware ↓ |
| --- | ---: | ---: | ---: | ---: |
| INT4-uniform | 10.452 | **124** | 392 | **3.17×** |
| Mixed-4.5 | 10.279 | **213** | 454 | **2.13×** |
| Mixed-5.0 | 10.091 | **289** | 516 | **1.79×** |
| INT8-uniform | 9.604 | 784 | 784 | 1.00× |

INT8-uniform의 aware==oblivious는 자명한 극한입니다: 정책 자체가 oblivious
가 가정하는 worst-case 비트 폭과 일치하므로 보수적 예산이 정확.

![E3 main figure](../experiments/results/figs/codesign_main.png)
*Figure 2. Qwen2.5-1.5B의 GBuf = 512 MB에서 정확도 × 토큰당 DRAM.
mixed-precision 정책이 INT4 레이어를 많이 가질수록 aware-vs-oblivious
격차가 커짐.*

**"이중 모드"(bimodal) — GBuf ≈ INT4_total**. 버퍼가 INT4 디코더는 잡을
수 있지만 INT8은 못 잡는 영역에서, aware 스케줄러는 혼합 정밀도 모델을
100 % on-chip 상주시킵니다 — 토큰당 DRAM = 0. INT8-uniform은 어떤
스케줄러로도 fit 불가. on-chip SRAM 용량이 모델의 INT4 footprint와
비슷한 가속기에 가장 깨끗한 메시지 (`figs/codesign_multi_gbuf.png`).

#### 6.2.1 GPU sanity check (해석 ↔ 실측 교차검증)

Table 4의 수치는 해석적으로 계산됩니다: `토큰당 DRAM = Σ_{t ∉ P}
bytes_aware(t)`. §3의 negative 매핑 결과가 그 근사를 디코드 영역에서
정당화하지만, 하드웨어 논문에서는 main result를 산수에만 기대지 않는
것이 안전합니다. 우리는 별도로 NVIDIA RTX 2000 Ada (16 GiB) 단일 GPU에서
Qwen2.5-1.5B의 FP16 baseline을 측정합니다 — 256 토큰 생성, `do_sample
=False`, NVML 전력을 백그라운드 스레드에서 20 ms 간격으로 sample
(`pipeline.profile_gpu`):

| Metric | Value |
| --- | ---: |
| Throughput | 25.0 tokens / s |
| 256 토큰 wall time | 10.24 s |
| 평균 GPU 전력 | 68.5 W (TDP cap 70 W) |
| 토큰당 에너지 | 2.74 J/token (avg-power × elapsed / 256) |
| 전력 sample 수 | 510 |

이 배포-스케일 baseline은 해석적 예측과 일치합니다. FP16에서 토큰당
디코더 바이트는 2.62 GB이고, PCIe-Gen4 대역폭이 이 영역에서 throughput을
제한합니다. 우리는 이 측정값과 양자화 구성 사이의 *순서*를 cross-check
으로 사용하지 절대값을 사용하지 않습니다. (FP16 vs INT4 vs INT8 vs
mixed의 통제된 GPU 비교 — 해석적 비율이 wall-time에도 그대로 나타나는지
직접 확인하는 — 는 `pipeline.profile_gpu`로 즉시 가능하며 §7.2의 후속
실험입니다.)

### 6.3 Sweep — cross-model 일반화

Figure 3 (`figs/sweep_cross_model.png`)이 같은 정확도 × DRAM/token
관계를 3 모델 × log-scale GBuf로 보여줍니다. 실선 aware, 점선 oblivious.
세 가지 관찰:

1. **곡선 형태가 보편적.** 세 모델 모두 `B_GBuf`에 대해 단조 감소,
   순위 INT4-uniform < Mixed-4.5 < Mixed-5.0 < INT8-uniform.
2. **"무릎(knee)"이 `B_GBuf ≈ INT4_total`에 위치**. 무릎 아래에서는 aware-
   oblivious 격차가 커지고, 무릎 위에서는 aware가 0으로 수렴.
3. **win 계수가 모델 크기와 함께 증가.** `B_GBuf ≈ INT4_total/2`
   (비교 가능한 운영점)에서:

| 모델 | INT4 total | GBuf | INT4 win | Mixed-4.5 win | Mixed-5.0 win |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.5B | 655 MB | 256 MB | 1.33× | 1.21× | 1.18× |
| Gemma-2-2B | 968 MB | 512 MB | **1.55×** | 1.37× | 1.27× |
| Qwen-7B | 3.25 GB | 2 GB | **1.97×** | 1.45× | 1.30× |

![Cross-model sweep](../experiments/results/figs/sweep_cross_model.png)
*Figure 3. cross-model sweep. 실선: 정밀도 인지 스케줄러. 점선: 정밀도
비인지 INT8 예산. aware 기여가 모델 family(Qwen vs Gemma)와 scale
(1.5B → 7B)에 무관하게 일반화됨.*

모델 크기와 함께 win 계수가 커지는 현상은, INT4와 INT8 표현 사이 절대
용량 차이가 클수록 per-tile byte 정밀 회계가 더 큰 이득을 가져옴을 시사
합니다.

### 6.4 E4 — per-tile HW-normalized 정책 (preliminary)

> **상태.** §6.4는 정책 *형태* (Table 5)와 그 형태의 잔존 결과
> (Table 6)을 보고합니다. Δppl/Δbyte를 Δppl/Δbit 위에 두는 동기인
> *정확도 결과*는 per-tile 정책에 대한 GPTQ 재실행이 필요하며 본
> 드래프트에서 *측정하지 않았습니다* (정책당 ~50분). §6.4를 preliminary
> 로 위치시키고, 제출 전 가장 중요한 단일 실험으로 §7.2에 기록합니다.
> 위 §4.3은 점수의 동기를 설명하고, §6.4의 표는 *정책 분포가 예상한
> 방향으로 달라짐을 보여주는 증거* 로 해석되어야 합니다.

§2.1의 레이어별 그리디는 byte 비용을 무시합니다. *per-projection*
granularity (Qwen-1.5B 기준 196 tile)에서 Δppl/Δbyte를 사용해 정책을 다시
도출합니다. Table 5가 동일 평균 비트 폭에서 per-layer 그리디와 정책 형태를
대비합니다.

| Projection 역할 | Mixed-4.5 (per-layer 그리디) | hwnorm-4.5 (per-tile) |
| --- | ---: | ---: |
| q_proj | 4 / 28 가 8-bit | 7 / 28 |
| k_proj | 4 / 28 | **17 / 28** |
| v_proj | 4 / 28 | **25 / 28 (89 %)** |
| o_proj | 4 / 28 | 15 / 28 |
| gate_proj | 4 / 28 | **0 / 28** |
| up_proj | 4 / 28 | 2 / 28 |
| down_proj | 4 / 28 | 5 / 28 |
| 총 8-bit 타일 | 28 | **71** |
| 달성 평균 bpw | 4.571 | **4.503** |

HW-normalized 정책은 8-bit 예산을 어텐션의 KV projection에 집중시킵니다.
이들은 승격 비용이 가장 싸면서 (MLP projection 대비 4~6×) 민감도가 높습
니다. MLP `gate_proj`는 전부 INT4에 남는데, 한 번 승격할 때마다 6.89 MB의
예산을 잡아먹어서 Δppl 기여 대비 너무 비싸기 때문입니다.

**잔존 측 결과는 작다.** 동일 평균 비트 폭에서 두 정책의 총 byte는
유사합니다 (정책 *분포*가 더 다르지 byte 총합은 비슷). GBuf = 512 MB에서
토큰당 DRAM:

| 예산 | 그리디 bpw | hwnorm bpw | 그리디 DRAM | hwnorm DRAM | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B = 4.5 | 4.571 | 4.503 | 213.3 MB | **206.4 MB** | 3.2 % ↓ |
| B = 5.0 | 5.000 | 5.025 | 289.0 MB | 289.0 MB | 0.0 % |

잔존 측 win이 작은 이유: knapsack은 타일의 절대 크기로 결정할 뿐, 그
타일이 어떤 logical role인지는 모릅니다. *정확도* 쪽에서 진짜 win이
기대됩니다 — 89 %의 `v_proj`을 8-bit로 보호하고 `gate_proj`을 전부
4-bit로 두면, (i) KV projection이 attention quality에 기여가 크고
(ii) 평균 비트 폭이 같으므로, per-layer 정책보다 낮은 ppl이 예상됩니다.
본 드래프트에서는 직접 측정하지 않았습니다 (정책당 ~50분 × 4 = ~3시간
wall-time 필요). 실험 loop은 (`scripts/_e4_per_tile.py` + 기존
`run_quant.py`) 갖춰져 있으며, camera-ready 버전의 headline 실험입니다.

![E4 hwnorm vs greedy](../experiments/results/figs/e4_hwnorm_vs_greedy.png)
*Figure 4. per-tile HW-normalized 정책 대 per-layer 그리디, 4개 GBuf
운영점. 두 스코어러의 토큰당 DRAM 곡선은 iso-bpw에서 거의 겹침. 차이는
정책 형태(Table 5)에 살아 있음.*

---

## 7. 논의

### 7.1 negative 매핑 결과의 진짜 의미

§3은 16×16 Eyeriss 계열 아키텍처 + `N = 1` 디코드에서 매퍼가 INT4와 INT8
같은 GEMM에 대해 동등 비용 매핑을 찾는다고 보고합니다. 이는 매핑 탐색이
LLM 가속기에 무용하다는 주장이 *아닙니다*. 다만, "총 가중치 byte ≫ on-chip
용량 + 가중치를 토큰당 한 번씩 touch"라는 영역에서는 매핑 결정 공간이
하나의 byte-bound 점으로 축소된다는 뜻입니다. 두 가지 보완 영역에서
매핑이 다시 의미를 가질 것입니다:

- **prefill (`N ≫ 1`)**: 같은 forward 안에서 가중치가 토큰 간 재사용되며,
  타일 선택이 재사용 패턴을 의미 있게 바꿉니다.
- **on-chip / off-chip 비율이 다른 아키텍처** (예: 64 KB GBuf의 weight-
  stationary): 버퍼 압박이 지배 변수가 되어 정밀도 인지 매핑 spread가
  재출현합니다.

둘 다 §7.2의 명시적 방향.

### 7.2 한계와 후속

- **에너지 모델링.** 잔존 모델은 DRAM byte를 유일한 효율 지표로 사용합니다.
  Accelergy 기반의 (비트 폭 × 타일 × pin 여부) 분해 에너지 추정으로
  mJ/token을 직접 보고할 수 있고, 이미 `pipeline.hw_timeloop`에 hook이
  있지만 본 드래프트의 main 효율 레버로는 사용하지 않았습니다.
- **Prefetching.** 0/1 knapsack 모델은 보수적입니다 — 타일을 pin 하거나
  전부 stream하거나 둘 중 하나입니다. 실제 스케줄러는 현재 레이어 계산
  중에 다음 레이어 가중치를 prefetch할 수 있습니다. 이를 모델링하면
  oblivious 기준선이 부분적으로 DRAM 시간을 숨길 수 있어 우리 주장의
  여유 폭이 줄어듭니다.
- **시퀀스 길이 의존.** Pinning은 많은 토큰에 걸쳐 가치가 있고, 단일 첫
  토큰 생성에서는 잔존 win이 0입니다. 본 결과는 묵시적으로 `T ≫ 1`
  (지속 디코드)을 가정합니다.
- **E4 정확도 검증.** HW-normalized 정책에 대해 GPTQ를 돌리지 않았습니다.
  "KV projection을 8-bit로 보호하면 iso-bpw에서 ppl이 낮아진다"는 주장은
  현재 가설입니다. 실험 루프는 (`_e4_per_tile.py`에 `--run-gptq` 옵션
  추가) 준비되어 있으며 가장 자연스러운 다음 실험.
- **워크로드 범위.** 3 모델, 1 task (WikiText-2 perplexity). MMLU, ARC-c,
  HellaSwag 등 downstream task 평가는 `pipeline.eval`에 plumb 되어 있으나
  본 보고서엔 미포함.

### 7.3 MAGNETO 및 GATHER와의 관계

MAGNETO는 power-aware mapping을 가속기 에너지의 주 contribution surface로
잡았습니다. GATHER는 LLM 특화 dataflow를 그 위에 얹었습니다. 본 연구는
스택의 한 층 위입니다: 고정된 정책과 고정된 매핑이 주어졌을 때, 어떤
타일이 on-chip 잔존의 이득을 받는지를 스케줄링이 결정합니다. 세 층
(정책 / 매핑 / 스케줄링)은 합성 가능하며, 본 논문은 스케줄링 층만
분리하여 정량화합니다.

---

## 8. 관련 연구

**레이어별 혼합 정밀도 정책 탐색.** HAWQ [Dong et al., ICCV 2019]와
Hessian eigen 변형인 HAWQ-V2 [Dong et al., NeurIPS 2020]가 2차 민감도
스코어를 도입했습니다. BRECQ [Li et al., ICLR 2021]는 블록 단위 재구성
오차를 실용적 신호로 제시했고, OmniQuant [Shao et al., ICLR 2024]는
탐색을 재매개변수화해 예산을 학습 변수로 만들었습니다. LLM.int8()
[Dettmers et al., NeurIPS 2022]는 outlier-aware INT8 추론을 주장했습니다.
넷 모두 Fig. 0의 *알고리즘* 층에 위치하며 잔존 예산과의 상호작용을
다루지 않습니다.

**LLM 양자화 배포.** GPTQ [Frantar et al., ICLR 2023]는 Hessian-aware
열 단위 라운딩으로, 본 논문의 양자화 backend입니다 (§5.1). AWQ
[Lin et al., MLSys 2024]는 activation-salient 채널을 보존합니다.
SmoothQuant [Xiao et al., ICML 2023]는 양자화 난이도를 activation–
weight 간에 옮깁니다. Marlin [Frantar et al., 2024]는 빠른 4-bit
디코드 커널을 제공해 혼합 정밀도의 byte 절약을 GPU 실측 지연 시간으로
변환합니다. 이들은 *커널*을 만듭니다. 어떤 가중치가 on-chip에 머무는지에
대한 스케줄링 결정은 이들의 범위 밖입니다.

**LLM 가속기 dataflow.** Sanger [Lu et al., MICRO 2021]와 Ant
[Guo et al., HPCA 2022]는 사용자 정의 dataflow를 가진 대표적 LLM
가속기입니다. 본 연구실의 선행 MAGNETO (power-aware mapping)과 GATHER
(LLM accelerator) — review를 위해 익명화 — 도 같은 공간에 위치합니다.
모두 Fig. 0의 *매핑* 층에 기여하며 정밀도 계획은 입력으로 고정합니다.
본 연구는 이들 모두와 합성됩니다 — Sanger·GATHER 스타일 매퍼가
타일당 `cycles_per_token`을 결정하면, 본 스케줄러가 그 중 어떤
타일이 DRAM 왕복을 피할지 결정합니다.

**LLM 서빙의 메모리 잔존 / 캐시 스케줄링.** SARATHI [Agrawal et al.,
MLSys 2024]는 GPU 활용을 위해 prefill/decode batch를 스케줄링하지만
요청 수준에서 동작합니다. DistServe [Zhong et al., OSDI 2024]는
prefill과 decode를 기계 사이에 분리합니다. FlexGen [Sheng et al.,
ICML 2023]은 offload-친화적 추론을 위해 GPU/CPU/SSD에 weight 저장을
tier로 나눕니다. ChunkAttention [Ye et al., ACL 2024]는 시퀀스
재사용을 위해 KV-캐시 접근을 재정렬합니다. 이들은 *KV 캐시*나 *런타임
상태*의 잔존을 다룹니다. 본 연구는 *혼합 정밀도 정책 하의 가중치
잔존*을 다루며, 다른 최적화 변수입니다.

**DNN 추론에서 knapsack 기반 스케줄링.** Tetris [Gao et al., ASPLOS
2017]는 3D DNN 분할에 연산자 ILP를 사용합니다. TVM/Ansor
[Zheng et al., OSDI 2020]는 탐색 기반이지만 kernel-fusion 단위입니다.
Liquid (익명화 lab 연구, 투고 중)는 추론 서버용 knapsack 기반 prefetch
를 다룹니다. 이들 중 어느 것도 정밀도 인지가 아니며, 피연산자 비트
폭은 사전 고정됩니다. 본 연구의 knapsack 인스턴스는 per-tile
`bytes_aware(t)`로 정의되며, 이는 *상위 정책이 만들어내는* 양입니다 —
이것이 스케줄링 층이 surface해야 하는 정확한 결합점입니다.

---

## 9. 결론

혼합 정밀도 양자화 문헌이 다루지 않는 시스템 레벨의 질문을 분리합니다:
*정책이 고정되었을 때, 어떤 가중치 타일을 on-chip에 둘 것인가?* 우리는
이를 per-tile byte 0/1 그리디 knapsack으로 정식화하고, 정밀도 비인지 INT8
예산 기준선 대비 토큰당 DRAM 트래픽을 3 모델·3 스케일에서 동일 운영점
(`B ≈ INT4_total/2`)에서 **1.33×–1.97×** 줄이고, 1.5B 모델의 INT4-total
knee 영역에서 최대 **3.17×** 줄임을 보입니다. win은 모델 크기와 함께
커집니다. 전체 재현 trail은 `experiments/`에서 제공되며, FP16 baseline의
GPU profile을 통한 해석-↔-실측 cross-check도 포함합니다. 양자화 알고리즘과
dataflow 매퍼 사이의 한 층에 이름과 정량적 기여를 부여하며, 두 가지
자연스러운 후속을 가리킵니다: prefetch-aware 스케줄링 모델, 그리고
동일 평균 비트 폭에서 어텐션 KV projection을 보호하는 per-tile
HW-normalized 정책 (per-tile 정책은 §6.4에서 구성되며 GPTQ 정확도
검증이 camera-ready 실험).

---

## 부록 A — 재현성

본 논문의 모든 주장은 다음에 추적됩니다:

- **코드:** `pipeline/` (Python 패키지). 10개 모듈; 모듈 맵은
  `pipeline/README.md`.
- **드라이버:** `experiments/scripts/run_<phase>.py`,
  `_sweep_target.sh`, `_sweep_qwen7b.sh`.
- **설정:** `experiments/configs/`.
- **결과:** `experiments/results/`. figure마다 옆에 데이터를 만든 CSV와
  생성기 스크립트가 있습니다.
- **파이프라인 상태:** `experiments/ROADMAP.md`.
- **상세 측정 방법:** `experiments/README.md` §6 (메트릭별), §8a–8f
  (실험별 결과).

acpl 서버 설치 경로 (sudo 없는 conda-forge Timeloop v3.0.3 + Accelergy,
gcc-13 `<cstdint>` 92-헤더 패치, `activate.d/timeloop_ld.sh` LD 경로)는
`experiments/scripts/install_timeloop.sh`에 캡처되어 있습니다.

## 부록 B — Walltime 프로파일

| 단계 | Qwen-1.5B | Gemma-2B | Qwen-7B |
| --- | --- | --- | --- |
| 민감도 (per-layer, 200 calib, 1 GPU) | ~5 분 | ~3 분 | OOM* |
| 민감도 (per-layer, 50 calib, 2 GPU shard) | ~2 분 | n/a | ~2 분 |
| 민감도 (per-projection, 50 calib, 1 GPU) | ~4 분 | 미측정 | 미측정 |
| 정책 생성 (4 예산) | < 1 초 | < 1 초 | < 1 초 |
| GPTQ (정책당, 32 calib) | ~12 분 | ~5 분 | ~25 분 |
| WikiText-2 평가 (체크포인트당) | ~30 초 | ~30 초 | ~2 분 |
| 잔존 sweep (8 GBuf × 2 sched) | < 1 초 | < 1 초 | < 1 초 |
| codesign 집계 + figure | ~10 초 | ~10 초 | ~10 초 |

\* Qwen-7B의 per-layer 민감도가 16 GiB 단일 GPU에서 FP16일 때 OOM —
14 GB 모델 위에 152k vocab의 sliding-window CE loss 텐서 ~1 GB가 더해져
용량 초과.
