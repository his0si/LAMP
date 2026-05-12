"""Phase 5 — GPU latency / throughput / energy.

These numbers are a GPU-side proxy for what the accelerator simulator
(Phase 3) reports analytically. They're not directly comparable in
absolute terms (GPU vs ASIC), but they let us cross-check trends:
if the simulator says policy A is 1.4× faster than policy B, GPU
generation throughput should move in the same direction.

Energy is read from NVML `nvmlDeviceGetPowerUsage` averaged over the
generate() call window. Subtract the idle baseline measured immediately
before generation to remove fan/PSU overhead.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ProfileResult:
    model_path: str
    prompt_tokens: int
    new_tokens: int
    elapsed_s: float
    tokens_per_s: float
    avg_power_W: float
    idle_power_W: float
    energy_J: float
    energy_per_token_mJ: float
    n_power_samples: int


def _power_watts(handle) -> float:
    import pynvml
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


class _PowerSampler(threading.Thread):
    """Background thread that polls NVML power every `interval` seconds."""

    def __init__(self, handle, interval: float = 0.02):
        super().__init__(daemon=True)
        self.handle = handle
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples: list[tuple[float, float]] = []  # (t, W)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.samples.append((time.perf_counter(), _power_watts(self.handle)))
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


def profile_generation(
    model_path: Path | str,
    *,
    prompt: str = "Explain in detail how transformer self-attention works.",
    new_tokens: int = 256,
    warmup: int = 2,
    gpu_index: int = 0,
) -> ProfileResult:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, device_map=f"cuda:{gpu_index}"
    ).eval()
    enc = tok(prompt, return_tensors="pt").to(f"cuda:{gpu_index}")
    ids, attn = enc.input_ids, enc.attention_mask

    for _ in range(warmup):
        model.generate(ids, attention_mask=attn, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    idle = sorted(_power_watts(handle) for _ in range(10))[5]

    sampler = _PowerSampler(handle, interval=0.02)
    sampler.start()
    t0 = time.perf_counter()
    out = model.generate(ids, attention_mask=attn, max_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sampler.stop()

    samples = sampler.samples
    n_new = int(out.shape[1] - ids.shape[1])
    avg_W = sum(w for _, w in samples) / max(1, len(samples))
    energy_J = max(0.0, avg_W - idle) * elapsed
    return ProfileResult(
        model_path=str(model_path),
        prompt_tokens=int(ids.shape[1]),
        new_tokens=n_new,
        elapsed_s=elapsed,
        tokens_per_s=n_new / elapsed,
        avg_power_W=avg_W,
        idle_power_W=idle,
        energy_J=energy_J,
        energy_per_token_mJ=1000.0 * energy_J / max(1, n_new),
        n_power_samples=len(samples),
    )
