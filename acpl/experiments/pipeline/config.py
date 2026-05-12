"""YAML config loader. Paths are resolved relative to experiments/."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS_ROOT.parent
DEFAULT_RESULTS = EXPERIMENTS_ROOT / "results"


@dataclass(frozen=True)
class TargetSpec:
    key: str
    repo: str
    local_dir: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int

    def weights_path(self, models_root: Path) -> Path:
        return models_root / self.local_dir


def load_targets(path: Path | str = EXPERIMENTS_ROOT / "configs" / "targets.yaml") -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    cfg["defaults"]["models_root"] = (path.parent / cfg["defaults"]["models_root"]).resolve()
    return cfg


def get_target(cfg: dict, key: str) -> TargetSpec:
    raw = cfg["targets"][key]
    return TargetSpec(
        key=key,
        repo=raw["repo"],
        local_dir=raw["local_dir"],
        num_hidden_layers=raw["num_hidden_layers"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_attention_heads=raw["num_attention_heads"],
        num_kv_heads=raw["num_kv_heads"],
    )


def load_yaml(path: Path | str) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)
