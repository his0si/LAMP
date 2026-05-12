"""Phase 2 — mixed-precision policy generation.

Inputs:
  sensitivity_path : path to sensitivity JSON produced by Phase 1
  allowed_widths   : e.g. [4, 8]
  target_avg_bits  : average bits-per-weight budget across all layers

Default algorithm (greedy, sensitivity-only):
  Start every layer at the minimum allowed width. At each step, promote the
  layer whose Δppl drop per added bit is largest, until the budget is hit.

E4 extension (`scorer="hwnorm"`): promote the layer whose **Δppl per added
byte** is largest. Small projections (k_proj, v_proj) become cheap to
promote because each added bit costs less *absolute byte*, so they get
8-bit upgrades earlier than the baseline scorer would pick. This couples
the policy decision to the residency budget directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import yaml


@dataclass
class Policy:
    target: str
    allowed_widths: list[int]
    target_avg_bits: float
    achieved_avg_bits: float
    per_layer_bits: list[int]
    notes: str = ""
    scorer: str = "sensitivity"   # "sensitivity" | "hwnorm"

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)


# Per-Qwen2/Llama transformer block weight count (used by hwnorm scorer).
# Caller hands target_cfg in; this avoids importing config from policy module
# to keep policy importable without paths.
def _block_weight_count(target_cfg: dict) -> int:
    h = target_cfg["hidden_size"]
    inter = target_cfg["intermediate_size"]
    nh = target_cfg["num_attention_heads"]
    kv = target_cfg["num_kv_heads"]
    hd = h // nh
    return (h * nh * hd) + 2 * (h * kv * hd) + (nh * hd * h) + 2 * (h * inter) + (inter * h)


def greedy_assign(
    sensitivities: list[dict],
    allowed_widths: list[int],
    target_avg_bits: float,
    weight_metric_prefix: str = "delta",
    cost_fn: Callable[[int, int, int], float] | None = None,
) -> tuple[list[int], float]:
    """Greedy bit promotion under an average-bits budget.

    `cost_fn(layer_idx, current_bits, next_bits) -> float` is the "price"
    of promoting a single layer from current_bits → next_bits. The
    objective at every step is to maximize `(Δppl drop) / cost`.

    Default cost: `next_bits - current_bits` → Δppl-per-bit (the
    sensitivity-only scorer; identical to the original implementation).
    """
    allowed = sorted(allowed_widths)
    n = len(sensitivities)
    bits = [allowed[0]] * n
    if cost_fn is None:
        cost_fn = lambda _i, c, nxt: float(nxt - c)

    def gain(layer_idx: int, current_bits: int) -> float:
        cur = sensitivities[layer_idx].get(f"{weight_metric_prefix}_{current_bits}bit", 0.0)
        nxt_w = next((w for w in allowed if w > current_bits), None)
        if nxt_w is None:
            return -1.0
        nxt = sensitivities[layer_idx].get(f"{weight_metric_prefix}_{nxt_w}bit", 0.0)
        return max(0.0, cur - nxt)

    while sum(bits) / n < target_avg_bits:
        best_idx, best_gain_per_cost = -1, -1.0
        for i, b in enumerate(bits):
            nxt = next((w for w in allowed if w > b), None)
            if nxt is None:
                continue
            c = cost_fn(i, b, nxt)
            if c <= 0:
                continue
            g = gain(i, b) / c
            if g > best_gain_per_cost:
                best_gain_per_cost, best_idx = g, i
        if best_idx < 0:
            break
        bits[best_idx] = next(w for w in allowed if w > bits[best_idx])

    achieved = sum(bits) / n
    return bits, achieved


def _hwnorm_cost(target_cfg: dict) -> Callable[[int, int, int], float]:
    """Cost = added bytes when promoting layer i from current_bits → next_bits.

    Uses the *block* weight count since our policy is per-decoder-block (one
    bit-width per layer covers all seven projections inside).
    """
    block_n = _block_weight_count(target_cfg)
    def cost(_layer_idx: int, current_bits: int, next_bits: int) -> float:
        return block_n * (next_bits - current_bits) / 8.0
    return cost


def make_policy(
    sensitivity_json_path: Path | str,
    policy_cfg: dict,
    *,
    target_cfg: dict | None = None,
) -> Policy:
    """Build a Policy from sensitivity + policy_cfg.

    `policy_cfg["search"]["method"]` controls the scorer:
      * "greedy" (default): sensitivity-only cost (Δppl / Δbits).
      * "hwnorm": Δppl / Δbytes (requires target_cfg).
    """
    data = json.loads(Path(sensitivity_json_path).read_text())
    scorer_name = policy_cfg.get("search", {}).get("method", "greedy")
    cost_fn: Callable[[int, int, int], float] | None
    if scorer_name == "hwnorm":
        if target_cfg is None:
            raise ValueError("hwnorm scorer needs target_cfg with hidden_size/intermediate_size/...")
        cost_fn = _hwnorm_cost(target_cfg)
        scorer_label = "hwnorm"
    else:
        cost_fn = None
        scorer_label = "sensitivity"

    bits, achieved = greedy_assign(
        sensitivities=data["layers"],
        allowed_widths=policy_cfg["allowed_widths"],
        target_avg_bits=policy_cfg["target_avg_bits"],
        weight_metric_prefix=policy_cfg.get("weight_metric", "ppl_delta").replace("ppl_", ""),
        cost_fn=cost_fn,
    )
    return Policy(
        target=data["target"],
        allowed_widths=policy_cfg["allowed_widths"],
        target_avg_bits=policy_cfg["target_avg_bits"],
        achieved_avg_bits=achieved,
        per_layer_bits=bits,
        notes=f"Generated from {sensitivity_json_path} ({scorer_label} scorer)",
        scorer=scorer_label,
    )


def save(policy: Policy, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if policy.scorer == "sensitivity" else f"_{policy.scorer}"
    out_path = out_dir / f"{policy.target}_{policy.target_avg_bits:.1f}bit{suffix}.yaml"
    out_path.write_text(policy.to_yaml())
    return out_path


# ---- E4: per-tile policy (consumes per-projection sensitivity) ----------

PROJECTION_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")


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


def make_per_tile_policy(
    per_projection_json_path: Path | str,
    *,
    target_cfg: dict,
    allowed_widths: list[int],
    target_avg_bits: float,
    scorer: str = "hwnorm",   # "hwnorm" | "sensitivity_per_tile"
) -> dict:
    """Greedy per-tile bit assignment.

    Tiles are the 7 projections × L layers (= 196 for Qwen2.5-1.5B). Each
    tile has its own sensitivity score and its own weight-byte cost when
    promoted. Scorers:
      * hwnorm:               Δppl / Δbytes
      * sensitivity_per_tile: Δppl / Δbits  (per-tile equivalent of greedy)
    """
    data = json.loads(Path(per_projection_json_path).read_text())
    allowed = sorted(allowed_widths)
    tiles = data["tiles"]
    n_tiles = len(tiles)

    role_bytes_step = {r: _proj_param_count(target_cfg, r) for r in PROJECTION_ROLES}

    bits = [allowed[0]] * n_tiles
    # Total *weight count* across all tiles, used to compute the avg bpw of
    # the policy (weighted by tile size so the budget is in real-byte terms).
    weights = [role_bytes_step[t["role"]] for t in tiles]
    total_w = sum(weights)

    def avg_bpw() -> float:
        return sum(bi * wi for bi, wi in zip(bits, weights)) / total_w

    def gain(i: int, cur_b: int) -> tuple[float, int | None]:
        cur = tiles[i].get(f"delta_{cur_b}bit", 0.0)
        nxt_w = next((w for w in allowed if w > cur_b), None)
        if nxt_w is None:
            return -1.0, None
        nxt = tiles[i].get(f"delta_{nxt_w}bit", 0.0)
        return max(0.0, cur - nxt), nxt_w

    while avg_bpw() < target_avg_bits:
        best_idx, best_gpc, best_next = -1, -1.0, None
        for i, b in enumerate(bits):
            g, nxt = gain(i, b)
            if nxt is None or g <= 0.0:
                continue
            if scorer == "hwnorm":
                cost = weights[i] * (nxt - b) / 8.0       # added bytes
            else:
                cost = float(nxt - b)                    # added bits
            gpc = g / cost
            if gpc > best_gpc:
                best_gpc, best_idx, best_next = gpc, i, nxt
        if best_idx < 0:
            break
        bits[best_idx] = best_next

    per_tile = {
        f"L{t['layer_idx']:02d}.{t['role']}": b
        for t, b in zip(tiles, bits)
    }
    return {
        "target": data["target"],
        "allowed_widths": allowed_widths,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": avg_bpw(),
        "per_tile_bits": per_tile,
        "notes": f"per-tile {scorer} scorer from {per_projection_json_path}",
        "scorer": scorer,
    }


def save_per_tile(policy: dict, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_" + policy["scorer"]
    out_path = out_dir / f"{policy['target']}_{policy['target_avg_bits']:.1f}bit{suffix}.yaml"
    out_path.write_text(yaml.safe_dump(policy, sort_keys=False))
    return out_path
