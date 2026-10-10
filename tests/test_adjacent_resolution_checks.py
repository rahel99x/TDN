"""Meaningful CPU controls for the requested-resolution readiness checks."""
from types import SimpleNamespace
import pytest
import torch

from tdn.analysis.adjacent.resolution_checks import audit_field, run_audit
from tdn.analysis.adjacent.models import make_model
from tdn.analysis.frontier.models import FNOBlock


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_resolution_audit_samples_the_same_continuous_field():
    coarse, fine = audit_field(64), audit_field(128)
    torch.testing.assert_close(coarse, fine[..., ::2, ::2], atol=0, rtol=0)
    assert coarse.min() > 0 and coarse.max() < 1
    spectrum = torch.fft.fft2(coarse, norm="forward")
    assert spectrum[0, 0, 13, 3].abs() > .01
    assert spectrum[0, 0, 14, 3].abs() > .01


def test_resolution_audit_large_grid_mathematical_controls():
    rows = []
    ctx = SimpleNamespace(protocol={"resolution": {"grids": [64, 128]},
            "tracks": ["discrete", "continuum"]}, device="cpu",
        budget=SimpleNamespace(check=lambda: None), record=lambda *a, **kw: rows.append((a, kw)))
    output = run_audit(ctx)
    assert len(output["checks"]) == len(rows) == 8
    for _, row in rows:
        assert all(c["verdict"] == "GOOD" for c in row["checks"] if c["category"] != "gap")
        assert [c["verdict"] for c in row["checks"] if c["category"] == "gap"] == ["NA"]


@pytest.mark.parametrize("grid", [64, 128])
def test_resolution_fno_full_grid_path_survives_cutoff(grid):
    torch.manual_seed(941781)
    model = make_model("fno_standard", "discrete", dict(modes=8, width=16, depth=3)).float()
    block = next(m for m in model.modules() if isinstance(m, FNOBlock))
    x = torch.arange(grid, dtype=torch.float32) / grid
    value = torch.cos(2*torch.pi*13*x)[None, None, :, None].expand(1, 16, grid, grid).clone()
    result = block(value)
    grad = torch.autograd.grad(result.square().mean(), block.local.weight)[0]
    assert torch.isfinite(grad).all() and bool(grad.abs().max() > 0)
