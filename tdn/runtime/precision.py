"""Precision controls outside compiled kernels; reference physics stays FP32/64."""
from contextlib import nullcontext
import torch

def reference_precision():
    # Use one family of APIs for the installed version, never mix old and new.
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def encoder_autocast(device: str, enabled: bool):
    if enabled:
        if not str(device).startswith("cuda"): raise ValueError("BF16 optimized candidate requires allocated GPU parity")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
