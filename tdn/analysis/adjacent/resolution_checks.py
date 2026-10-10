"""Large-grid algebra checks, independent of fitting or reference accuracy."""
from __future__ import annotations

import torch

from tdn.numerics import Equation, Geometry
from .core import check
from .models import interaction_channels, make_model


def audit_field(grid, *, device="cpu", dtype=torch.float64):
    """One resolved continuous field sampled on either grid, without clipping."""
    x = torch.arange(grid, dtype=dtype, device=device) / grid
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return (.43 + .05 * torch.cos(2 * torch.pi * (xx + yy) + .3)
            + .04 * torch.cos(2 * torch.pi * (13 * xx + 3 * yy) + .7)
            + .03 * torch.cos(2 * torch.pi * (14 * xx + 3 * yy) - .2))[None, None]


def run_audit(ctx):
    """Check both requested sizes; passing this is not solver superiority."""
    rows = []
    for grid in ctx.protocol["resolution"]["grids"]:
        geo, eq = Geometry((grid, grid), (1., 1.)), Equation(.004, 3.)
        u = audit_field(grid, device=ctx.device)
        for track in ctx.protocol["tracks"]:
            for nodes in (2, 4):
                ctx.budget.check()
                shared = interaction_channels(u, .06, eq, geo, track=track,
                    nodes=nodes, split_modes=4, output_modes=8)
                direct = interaction_channels(u, .06, eq, geo, track=track,
                    nodes=nodes, split_modes=4, output_modes=8, implementation="reference")
                config = dict(modes=8, split_modes=4, quad_nodes=nodes, reaction_substeps=4)
                channel = make_model("channel_fixed", track, config).to(device=ctx.device, dtype=u.dtype)
                rule = make_model("quad2_fixed" if nodes == 2 else "quad4_fixed", track, config).to(device=ctx.device, dtype=u.dtype)
                error = float((shared - direct).abs().max())
                endpoint_error = float((channel(u, .06, eq, geo) - rule(u, .06, eq, geo)).abs().max())
                ctx.record(f"resolution-identity/{grid}/{track}/{nodes}", ["C01", "D05"],
                    metrics=dict(grid=grid, track=track, nodes=nodes, polarization_max_error=error,
                                 fixed_recombination_max_error=endpoint_error),
                    checks=[check("large-grid-polarization", error, 2e-12, category="math"),
                            check("large-grid-fixed-recombination", endpoint_error, 2e-12, category="correctness"),
                            check("scientific-advantage", None, None, category="gap",
                                  reason="Algebra alone does not establish improved prediction or cost")])
                rows.append(dict(grid=grid, track=track, nodes=nodes, polarization_max_error=error,
                                 fixed_recombination_max_error=endpoint_error))
    return dict(status="COMPLETED", checks=rows,
                scope="FP64 implementation identities on the requested grids; not trained or GPU evidence unless executed there")
