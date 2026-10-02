"""Training-only feature normalization remains part of the deployed state map."""
from __future__ import annotations

import torch
from torch import nn

from tdn.models import build_model


class NormalizedModel(nn.Module):
    def __init__(self, base: nn.Module, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32))
        self.t_ref = base.t_ref
        self.U_ref = base.U_ref
        object.__setattr__(self, "_forward_impl", base.forward)
        object.__setattr__(self, "_encode_impl", getattr(base, "encode", None))
        object.__setattr__(self, "_decode_impl", getattr(base, "decode", None))

    @property
    def is_h_independent(self):
        return self.base.is_h_independent

    def normalize(self, features):
        normalized = (features - self.feature_mean.to(features.dtype)) / self.feature_std.to(features.dtype)
        return normalized.to(next(self.base.parameters()).dtype)

    def forward(self, features, h):
        return self._forward_impl(self.normalize(features), h)

    def encode(self, features):
        return self._encode_impl(self.normalize(features))

    def decode(self, encoded, h):
        return self._decode_impl(encoded, h)

    def compile_kernels(self, mode: str):
        """Keep original parameters/state keys while compiling pure tensor calls."""
        object.__setattr__(self, "_forward_impl", torch.compile(self.base.forward, mode=mode))
        if self.is_h_independent:
            object.__setattr__(self, "_encode_impl", torch.compile(self.base.encode, mode=mode))
            object.__setattr__(self, "_decode_impl", torch.compile(self.base.decode, mode=mode))


def model_from_checkpoint(payload: dict, device: str | torch.device = "cpu") -> NormalizedModel:
    config = payload["config"]
    normalization = payload["normalization"]
    base = build_model(len(normalization["mean"]), config["model"],
                       t_ref=config["problem"]["t_ref"], U_ref=config["problem"]["U_ref"])
    model = NormalizedModel(base, normalization["mean"], normalization["std"])
    model.load_state_dict(payload["model_state"])
    return model.to(device)
