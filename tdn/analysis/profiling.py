"""Honest steady-state CPU/CUDA timing and allocator measurements."""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

try:
    import resource
except ImportError:  # Native Windows has no resource module.
    resource = None


def host_peak_rss() -> tuple[int | None, str]:
    """Return a process-lifetime host peak, with explicit platform semantics."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]

        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize), "process lifetime peak working set (Windows), not stage-specific allocation"
        except (OSError, AttributeError):
            pass
        return None, "Windows peak working set unavailable; host memory was not measured"
    if resource is not None:
        multiplier = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier), f"process lifetime peak RSS ({sys.platform}), not stage-specific allocation"
    return None, "host peak RSS unavailable on this platform"


def measure(operation: Callable[[], Any], *, device: str | torch.device = "cpu",
            warmup: int = 1, repeats: int = 3, memory_policy: dict | None = None) -> tuple[Any, dict]:
    dev = torch.device(device)
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid measurement repeat counts")
    if dev.type == "cuda":
        from tdn.runtime.preflight import execution_mode
        if execution_mode() != "desktop" and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("CUDA benchmarks require a Slurm allocation or explicit desktop mode")
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
    host_memory, host_scope = host_peak_rss()
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
        "host_process_peak_rss_bytes": host_memory,
        "host_memory_scope": host_scope,
        "includes": "full operation including feature construction, layout conversion, solver and inference",
        "data_io_included": False,
    }
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(dev).total_memory
        desktop = os.environ.get("TDN_EXECUTION_MODE") == "desktop"
        policy = memory_policy or {}
        soft = min(policy.get("soft_vram_gib", 18 if desktop else 30) * 1024**3,
                   policy.get("soft_vram_fraction", .75 if desktop else .8) * total)
        report.update(device_total_bytes=int(total), soft_budget_bytes=int(soft),
                      soft_budget_passed=report["peak_reserved_bytes"] <= soft,
                      hard_device_memory_warning=report["device_used_bytes_observed_peak"] > policy.get("hard_memory_fraction", .9) * total)
    return output, report
