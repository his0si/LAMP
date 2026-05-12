"""Experimental pipeline for the LAMP (Layer-wise Mixed-Precision) project.

The pipeline executes six logical phases plus two analysis modules. Each
module is independently importable; see `pipeline/README.md` for the
per-module summary and entry-point scripts under `experiments/scripts/`.

    config        — YAML loader (targets, policy search, arch)
    sensitivity   — Phase 1: per-layer Δppl when only layer l is quantized
    policy        — Phase 2: greedy bit-width assignment under a budget
    hw_timeloop   — Phase 3: Timeloop+Accelergy wrapper (mapper invocation)
    quant_runner  — Phase 4a: build a GPTQ-quantized checkpoint per policy
    eval          — Phase 4b: WikiText-2 perplexity + lm-eval-harness
    profile_gpu   — Phase 5: GPU latency / throughput / NVML energy
    viz           — Phase 6: sensitivity / policy / Pareto plots

    mapping       — E2 (mapping pilot): LayerShape, MappingCache, aggregate_model
    residency     — E2-residency (main): knapsack pinning, aware vs oblivious
    codesign      — E3 aggregator: policy + mapping + eval → codesign points
"""

__all__ = [
    "config",
    "sensitivity",
    "policy",
    "hw_timeloop",
    "quant_runner",
    "eval",
    "profile_gpu",
    "viz",
    "mapping",
    "residency",
    "codesign",
]
