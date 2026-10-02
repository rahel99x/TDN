"""Honest steady-state CPU/CUDA timing and allocator measurements."""
from __future__ import annotations

import os
import resource
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def measure(operation: Callable[[], Any], *, device: str | torch.device = "cpu",
            warmup: int = 1, repeats: int = 3) -> tuple[Any, dict]:
    dev = torch.device(device)
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid measurement repeat counts")
    if dev.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA benchmarks require a Slurm allocation")
    def synchronize() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    synchronize()
    if dev.type == "cuda":torch.cuda.reset_peak_memory_stats(dev)
    initialize_start = time.perf_counter()
    output = operation()
    synchronize()
    initialization_seconds = time.perf_counter() - initialize_start
    warm_start = time.perf_counter()
    for _ in range(warmup):
        output = operation()
    synchronize()
    warmup_seconds = time.perf_counter() - warm_start
    first_use_memory={"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(dev)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(dev))} if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    elapsed, cuda_elapsed, device_used = [], [], []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        if dev.type == "cuda":
            before = torch.cuda.Event(enable_timing=True)
            after = torch.cuda.Event(enable_timing=True)
            before.record()
        output = operation()
        if dev.type == "cuda":
            after.record()
        synchronize()
        elapsed.append(time.perf_counter() - start)
        if dev.type == "cuda":
            cuda_elapsed.append(before.elapsed_time(after) / 1000)
            free, total = torch.cuda.mem_get_info(dev)
            device_used.append(total - free)
    report = {
        "device": str(dev), "initialization_first_use_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds, "warmup_repeats": warmup,
        "first_use_and_warmup_memory":first_use_memory,
        "steady_state_repeats": repeats, "wall_seconds_raw": elapsed,
        "wall_seconds_median": float(np.median(elapsed)),
        "cuda_event_seconds_raw": cuda_elapsed if dev.type == "cuda" else None,
        "cuda_event_seconds_median": float(np.median(cuda_elapsed)) if cuda_elapsed else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else None,
        "device_used_bytes_observed_peak": max(device_used) if device_used else None,
        "device_used_sampling": "after each synchronized repetition; not a per-kernel peak" if device_used else None,
        "host_process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "host_memory_scope": "process lifetime peak RSS (Linux), not stage-specific allocation",
        "includes": "full operation including feature construction, layout conversion, solver and inference",
        "data_io_included": False,
    }
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(dev).total_memory
        report.update(device_total_bytes=int(total), soft_budget_bytes=int(min(30 * 1024**3, .8 * total)),
                      soft_budget_passed=report["peak_reserved_bytes"] <= min(30 * 1024**3, .8 * total),
                      hard_device_memory_warning=report["device_used_bytes_observed_peak"] > .9 * total)
    return output, report
