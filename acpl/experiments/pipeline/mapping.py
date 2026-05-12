"""E2/E3 — precision-aware mapping search and policy aggregation.

This is the central module for the paper's main contribution. It does NOT
call Timeloop directly — that lives in `hw_timeloop.py`. Instead, this
module is the data layer + analyzer:

  * `LayerShape`               — canonical (M, K, N, name) per unique GEMM
  * `MappingResult`            — single timeloop run output
  * `MappingCache`             — JSON-backed (shape, bits) → top-K mappings
  * `apply_mapping(layer, bits, mapping)` → analytic energy/cycles when
    we re-use a mapping picked for a different bits
  * `aggregate_model(policy, cache, strategy)` → per-token totals under one
    of {"oracle", "precision_aware", "oblivious_int8", "oblivious_int4"}

Strategies map directly to E3 configurations A–D:
  - "uniform_int4"     == Config A (uniform INT4 + INT4-optimal mapping)
  - "uniform_int8"     == Config B (uniform INT8 + INT8-optimal mapping)
  - "oblivious_int8"   == Config C (mixed bits + INT8-optimal mapping forced)
  - "precision_aware"  == Config D (mixed bits + per-bit best mapping)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Literal

import yaml

Strategy = Literal["uniform_int4", "uniform_int8", "oblivious_int8", "precision_aware"]


@dataclass(frozen=True)
class LayerShape:
    """Canonical GEMM shape. We deduplicate by (M, K, N)."""
    name: str            # logical role: q_proj, k_proj, ..., down_proj
    M: int               # output features
    K: int               # input features
    N: int = 1           # token batch (decode step → 1)

    @property
    def key(self) -> str:
        return f"{self.name}_M{self.M}_K{self.K}_N{self.N}"


@dataclass
class MappingResult:
    """One Timeloop mapper output. cycles/energy are per single GEMM call."""
    shape_key: str
    bits_w: int
    bits_a: int
    cycles: int
    energy_pJ: float
    pe_utilization: float           # 0..1
    dram_bytes: int
    gbuf_hit_rate: float
    tile_loops: dict = field(default_factory=dict)   # e.g. {"M_l1": 32, ...}
    dataflow: str = ""              # e.g. "weight_stationary"
    rank: int = 0                   # rank inside top-K for the (shape, bits) key


class MappingCache:
    """JSON-backed cache of (shape_key, bits) → list[MappingResult].

    Used by E2 (mapping spread) and E3 (aggregation). Keeps top-K mappings
    per (shape, bits) so we can re-evaluate alternatives without re-running
    Timeloop.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict[str, list[dict]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def _k(shape_key: str, bits: int) -> str:
        return f"{shape_key}::w{bits}"

    def get(self, shape: LayerShape, bits: int) -> list[MappingResult]:
        raw = self._data.get(self._k(shape.key, bits), [])
        return [MappingResult(**r) for r in raw]

    def best(self, shape: LayerShape, bits: int, objective: str = "energy_pJ") -> MappingResult:
        cands = self.get(shape, bits)
        if not cands:
            raise KeyError(f"No mapping for {shape.key} @ {bits}b")
        return min(cands, key=lambda r: getattr(r, objective))

    def put(self, shape: LayerShape, bits: int, results: Iterable[MappingResult]) -> None:
        rs = list(results)
        for i, r in enumerate(rs):
            r.rank = i
        self._data[self._k(shape.key, bits)] = [asdict(r) for r in rs]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))


def apply_mapping(shape: LayerShape, target_bits: int, mapping: MappingResult) -> MappingResult:
    """Re-evaluate `mapping` (which was picked at mapping.bits_w) when applied
    to the same shape at `target_bits`.

    Analytic model — used so we can answer "what if we forced the
    INT8-optimal mapping on a 4-bit layer?" without re-running Timeloop.

    Assumptions (documented as part of the methodology):
      * Cycles scale by max(1, target_bits / native_bits) where native = 8.
        i.e. INT4 packed 2× per slot ⇒ INT8-optimal mapping run on INT4
        data takes the SAME cycles (compute-bound) but moves half the
        weight bytes. (This matches Option-A INT8-native + INT4 packing.)
      * DRAM bytes scale linearly with target_bits / mapping.bits_w
        (only weight tensors; activations stay fp16).
      * Energy is split: compute_energy ≈ cycles · pj_per_op(bits),
        memory_energy ≈ dram_bytes · pj_per_byte. We assume the cache
        stored `energy_pJ` already used `mapping.bits_w`; we rescale
        proportionally — a coarse but order-preserving model.
      * PE utilization and GBuf hit-rate unchanged (mapping geometry is
        the same; only operand sizes change).

    A finer model (per-component Accelergy table) is a TODO; cross-app
    ratios computed here should be treated as *ranking* signals first.
    """
    if mapping.bits_w == target_bits:
        return mapping

    byte_ratio = target_bits / mapping.bits_w
    # Cycles unchanged when target_bits <= 8 (packed); doubled if target_bits > 8.
    cycle_ratio = max(1.0, target_bits / 8)
    return MappingResult(
        shape_key=mapping.shape_key,
        bits_w=target_bits,
        bits_a=mapping.bits_a,
        cycles=int(mapping.cycles * cycle_ratio),
        energy_pJ=mapping.energy_pJ * (0.5 * cycle_ratio + 0.5 * byte_ratio),
        pe_utilization=mapping.pe_utilization,
        dram_bytes=int(mapping.dram_bytes * byte_ratio),
        gbuf_hit_rate=mapping.gbuf_hit_rate,
        tile_loops=mapping.tile_loops,
        dataflow=mapping.dataflow,
        rank=-1,           # synthesized, not from the cache
    )


def cross_application_loss(
    shape: LayerShape, cache: MappingCache, bits_pair: tuple[int, int] = (4, 8)
) -> dict:
    """How much do we lose by using the wrong-bit best mapping?

    Returns {energy_loss_4on8, energy_loss_8on4, cycles_loss_4on8, ...}
    where `x_loss_AonB` = (best_A applied to bits B) / (best_B applied to bits B).
    Values ≥ 1.0 mean the same-precision mapping is at least as good.
    A high value ⇒ precision-aware mapping helps for that shape.
    """
    a, b = bits_pair
    best_a = cache.best(shape, a)
    best_b = cache.best(shape, b)
    a_on_b = apply_mapping(shape, b, best_a)
    b_on_a = apply_mapping(shape, a, best_b)
    return {
        "shape": shape.key,
        "energy_loss_AonB": a_on_b.energy_pJ / best_b.energy_pJ,
        "energy_loss_BonA": b_on_a.energy_pJ / best_a.energy_pJ,
        "cycle_loss_AonB":  a_on_b.cycles / best_b.cycles,
        "cycle_loss_BonA":  b_on_a.cycles / best_a.cycles,
        "dram_ratio_AonB":  a_on_b.dram_bytes / best_b.dram_bytes,
        "dram_ratio_BonA":  b_on_a.dram_bytes / best_a.dram_bytes,
    }


def aggregate_model(
    *,
    policy_yaml: Path | str,
    cache: MappingCache,
    shapes_per_layer: list[LayerShape],   # 7 shapes — repeated across all 28 layers
    strategy: Strategy,
) -> dict:
    """Sum per-layer metrics across the whole decoder under a given strategy.

    `shapes_per_layer`: the unique linear shapes inside one decoder block.
    All blocks share the same shapes, so we multiply by L = num_hidden_layers
    from the policy's per_layer_bits length.
    """
    pol = yaml.safe_load(Path(policy_yaml).read_text())
    L = len(pol["per_layer_bits"])

    total_cycles, total_energy, total_dram = 0, 0.0, 0
    util_weighted, util_ops_total = 0.0, 0

    for l in range(L):
        bits = pol["per_layer_bits"][l]

        for shape in shapes_per_layer:
            if strategy == "uniform_int4":
                m = cache.best(shape, 4)
            elif strategy == "uniform_int8":
                m = cache.best(shape, 8)
            elif strategy == "oblivious_int8":
                # Force INT8-optimal mapping, evaluate at actual bits.
                m = apply_mapping(shape, bits, cache.best(shape, 8))
            elif strategy == "precision_aware":
                m = cache.best(shape, bits)
            else:
                raise ValueError(f"unknown strategy: {strategy}")

            ops = shape.M * shape.K * shape.N
            total_cycles += m.cycles
            total_energy += m.energy_pJ
            total_dram   += m.dram_bytes
            util_weighted += m.pe_utilization * ops
            util_ops_total += ops

    return {
        "strategy": strategy,
        "policy": Path(policy_yaml).stem,
        "cycles_per_token": total_cycles,
        "energy_pJ_per_token": total_energy,
        "dram_bytes_per_token": total_dram,
        "mean_pe_utilization": util_weighted / max(1, util_ops_total),
    }


def unique_shapes_qwen(target_cfg: dict) -> list[LayerShape]:
    """Return the 7 unique linear shapes per Qwen2/Llama decoder block."""
    h = target_cfg["hidden_size"]
    inter = target_cfg["intermediate_size"]
    nh = target_cfg["num_attention_heads"]
    kv = target_cfg["num_kv_heads"]
    hd = h // nh
    return [
        LayerShape("q_proj",    M=nh * hd, K=h),
        LayerShape("k_proj",    M=kv * hd, K=h),
        LayerShape("v_proj",    M=kv * hd, K=h),
        LayerShape("o_proj",    M=h,       K=nh * hd),
        LayerShape("gate_proj", M=inter,   K=h),
        LayerShape("up_proj",   M=inter,   K=h),
        LayerShape("down_proj", M=h,       K=inter),
    ]
