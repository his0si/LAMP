"""Memory-residency scheduling for mixed-precision LLM decoders.

Given an on-chip buffer (GBuf) that's small relative to the model's total
decoder weights, we cannot keep everything resident. The scheduler decides
which **(layer_idx, projection)** weight tiles to pin in GBuf so they
don't need re-reading from DRAM every token.

Three schedulers we compare:

  * **precision_aware**: knows the per-layer bit-width policy and uses
    the *actual* tile bytes for packing with the smallest-first knapsack
    rule. Can fit more tiles than oblivious; also uses a better solver
    than random.
  * **precision_aware_random**: also knows the policy and uses actual
    tile bytes for the budget check, but selects tiles uniformly at
    random (averaged over multiple seeds). Isolates the contribution of
    the *solver* on top of precision awareness.
  * **precision_oblivious_int8**: budgets every tile at the worst case
    bit-width (INT8 in our setup). Reflects a deployment-time scheduler
    that ignores the policy. Pins fewer tiles → more DRAM traffic.

The win condition for the paper: at the same GBuf size and the same
mixed-precision policy, `precision_aware` strictly reduces DRAM bytes /
token compared to `precision_oblivious_int8` — and the gap widens as the
GBuf size approaches the *INT4-fit-but-INT8-overflow* regime for the
relevant projections.

This module is intentionally analytic (no Timeloop dependence). E2 pilot
already established that in our 16×16 Eyeriss-like arch, per-shape cycle
count tracks DRAM bytes nearly linearly at decode-N=1, so DRAM bytes/token
is the right efficiency proxy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import yaml

from . import config

# Per-block linear shapes are the seven projections; we enumerate them
# per layer so the scheduler can pin individual (layer, role) tiles.
SchedStrategy = Literal[
    "precision_aware",
    "precision_aware_random",
    "precision_oblivious_int8",
]
RANDOM_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class WeightTile:
    layer_idx: int          # 0 .. L-1
    role: str               # one of PROJECTIONS
    bits_aware: int         # actual bit-width from the policy
    n_weights: int          # number of weight scalars in this tile
    bytes_aware: int        # bits_aware * n_weights // 8
    bytes_int8: int         # what an oblivious scheduler assumes
    bytes_int4: int         # symmetric

    @property
    def key(self) -> str:
        return f"L{self.layer_idx:02d}.{self.role}"


def _proj_param_count(target_cfg: dict, role: str) -> int:
    h = target_cfg["hidden_size"]
    inter = target_cfg["intermediate_size"]
    nh = target_cfg["num_attention_heads"]
    kv = target_cfg["num_kv_heads"]
    hd = h // nh
    return {
        "q_proj":    h * nh * hd,
        "k_proj":    h * kv * hd,
        "v_proj":    h * kv * hd,
        "o_proj":    nh * hd * h,
        "gate_proj": h * inter,
        "up_proj":   h * inter,
        "down_proj": inter * h,
    }[role]


def _tile_key(layer_idx: int, role: str) -> str:
    return f"L{layer_idx:02d}.{role}"


def enumerate_tiles(target_cfg: dict, policy_yaml: Path | str) -> list[WeightTile]:
    """Build the per-(layer, projection) tile list given a policy.

    Two policy formats are supported:
      * per-layer:  `per_layer_bits: [4, 8, 4, ...]` — one bit-width per
        decoder block (covers all 7 projections inside).
      * per-tile:   `per_tile_bits: {"L00.q_proj": 4, "L00.k_proj": 8, ...}`
        — per-projection bit-width. Used by E4 hwnorm policies.

    If both are present, per-tile takes precedence.
    """
    pol = yaml.safe_load(Path(policy_yaml).read_text())
    per_tile = pol.get("per_tile_bits") or {}
    bits_per_layer = pol.get("per_layer_bits") or []
    tiles: list[WeightTile] = []
    n_layers = max(
        len(bits_per_layer),
        1 + max((int(k.split(".")[0][1:]) for k in per_tile), default=-1),
    )
    for l in range(n_layers):
        for role in PROJECTIONS:
            key = _tile_key(l, role)
            if key in per_tile:
                bits = per_tile[key]
            elif l < len(bits_per_layer):
                bits = bits_per_layer[l]
            else:
                raise KeyError(f"policy has neither per-tile nor per-layer entry for {key}")
            n = _proj_param_count(target_cfg, role)
            tiles.append(WeightTile(
                layer_idx=l, role=role,
                bits_aware=bits, n_weights=n,
                bytes_aware=n * bits // 8,
                bytes_int8=n,
                bytes_int4=n // 2,
            ))
    return tiles


# ---- packing -------------------------------------------------------------

def pack_greedy(tiles: Iterable[WeightTile], gbuf_bytes: int,
                size_attr: str) -> set[str]:
    """Greedy 0/1 knapsack: pick tiles by ascending size (biggest reuse benefit
    per byte for autoregressive decode where every tile has the same access
    count). For a fixed total access pattern, smallest-first is optimal."""
    sorted_tiles = sorted(tiles, key=lambda t: getattr(t, size_attr))
    pinned: set[str] = set()
    used = 0
    for t in sorted_tiles:
        sz = getattr(t, size_attr)
        if used + sz <= gbuf_bytes:
            pinned.add(t.key)
            used += sz
    return pinned


def pack_random(tiles: Iterable[WeightTile], gbuf_bytes: int,
                size_attr: str, seed: int) -> set[str]:
    """Random 0/1 knapsack: same budget check as ``pack_greedy`` but visits
    tiles in a uniformly-random order. A weaker solver that still knows the
    actual per-tile cost; comparing this to ``pack_greedy`` isolates the
    contribution of the smallest-first rule on top of precision awareness.
    """
    order = list(tiles)
    rng = random.Random(seed)
    rng.shuffle(order)
    pinned: set[str] = set()
    used = 0
    for t in order:
        sz = getattr(t, size_attr)
        if used + sz <= gbuf_bytes:
            pinned.add(t.key)
            used += sz
    return pinned


def schedule_dram_per_token(
    tiles: list[WeightTile],
    gbuf_bytes: int,
    strategy: SchedStrategy,
) -> dict:
    """Return scheduling outcome under one strategy.

    Output keys:
      pinned_count, pinned_bytes_aware, pinned_bytes_budget
      dram_bytes_per_token, total_decoder_bytes_aware
    """
    if strategy == "precision_aware":
        # Pack using actual bytes; pinned tiles consume exactly bytes_aware.
        pinned = pack_greedy(tiles, gbuf_bytes, "bytes_aware")
    elif strategy == "precision_oblivious_int8":
        # Scheduler budgets every tile at INT8; pinned tiles physically occupy
        # only bytes_aware, but its decision-making uses bytes_int8.
        pinned = pack_greedy(tiles, gbuf_bytes, "bytes_int8")
    elif strategy == "precision_aware_random":
        # Same byte awareness as precision_aware, but a uniformly-random
        # visiting order instead of smallest-first. Return the mean DRAM
        # bytes per token across RANDOM_SEEDS so the result is stable.
        seed_dram: list[int] = []
        seed_pinned_count: list[int] = []
        seed_pinned_aware: list[int] = []
        for s in RANDOM_SEEDS:
            p = pack_random(tiles, gbuf_bytes, "bytes_aware", seed=s)
            seed_pinned_count.append(len(p))
            seed_pinned_aware.append(sum(t.bytes_aware for t in tiles if t.key in p))
            total = sum(t.bytes_aware for t in tiles)
            seed_dram.append(total - seed_pinned_aware[-1])
        return {
            "strategy": strategy,
            "gbuf_bytes": gbuf_bytes,
            "pinned_count": int(round(sum(seed_pinned_count) / len(RANDOM_SEEDS))),
            "pinned_keys": [],  # varies per seed; not meaningful to report
            "pinned_bytes_aware": int(round(sum(seed_pinned_aware) / len(RANDOM_SEEDS))),
            "pinned_bytes_budget": int(round(sum(seed_pinned_aware) / len(RANDOM_SEEDS))),
            "dram_bytes_per_token": int(round(sum(seed_dram) / len(RANDOM_SEEDS))),
            "total_decoder_bytes_aware": sum(t.bytes_aware for t in tiles),
            "n_seeds": len(RANDOM_SEEDS),
        }
    else:
        raise ValueError(strategy)

    total_aware = sum(t.bytes_aware for t in tiles)
    pinned_aware = sum(t.bytes_aware for t in tiles if t.key in pinned)
    pinned_budget = sum(
        (t.bytes_int8 if strategy == "precision_oblivious_int8" else t.bytes_aware)
        for t in tiles if t.key in pinned
    )

    return {
        "strategy": strategy,
        "gbuf_bytes": gbuf_bytes,
        "pinned_count": len(pinned),
        "pinned_keys": sorted(pinned),
        "pinned_bytes_aware": pinned_aware,
        "pinned_bytes_budget": pinned_budget,
        "dram_bytes_per_token": total_aware - pinned_aware,
        "total_decoder_bytes_aware": total_aware,
    }


def compare(
    target_key: str,
    policy_yaml: Path | str,
    gbuf_sizes_bytes: Iterable[int],
) -> list[dict]:
    """Full sweep: (gbuf × strategy) for one policy. Returns rows for table."""
    cfg = config.load_targets()
    target_cfg = cfg["targets"][target_key]
    tiles = enumerate_tiles(target_cfg, policy_yaml)
    rows: list[dict] = []
    for g in gbuf_sizes_bytes:
        for strat in ("precision_aware", "precision_aware_random",
                      "precision_oblivious_int8"):
            r = schedule_dram_per_token(tiles, g, strat)
            r["policy"] = Path(policy_yaml).stem
            rows.append(r)
    return rows
