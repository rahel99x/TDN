from types import SimpleNamespace
import json
import torch

from tdn.analysis.response.diagnostics import run
from tdn.analysis.response.evaluation import classical_methods
from tdn.analysis.response.references import equation_geometry, generate_reference


def test_symmetry_and_independent_parameter_probes_keep_one_parent(tmp_path):
    x=torch.arange(6)/6
    u=(.4+.05*torch.cos(2*torch.pi*x[:,None])+.02*torch.sin(4*torch.pi*x[None,:]))[None,None]
    r=dict(parent_id="parent",query_id="query",target_kind="fd_same_grid",split="development_test",
        grid=[6,6],lengths=[1.,1.],horizon=.02,kappa=.003,reaction_rate=2.)
    settings=dict(initial_steps=4,max_steps=64,tolerance=1e-9,max_seconds_per_query=5,crosscheck=True)
    eq,geom=equation_geometry(r);truth,meta=generate_reference(u,.02,eq,geom,r["target_kind"],settings)
    r["reference"]=meta
    sample=dict(u0=u,target=truth,record=r)
    methods=classical_methods(r,[1])
    # Exactly one numerical control keeps the diagnostic regression cheap.
    methods={k:v for k,v in methods.items() if "df" in k}
    ctx=SimpleNamespace(path=tmp_path,protocol={"reference":settings},budget=SimpleNamespace(check=lambda:None))
    result=run(ctx,[sample],methods)
    symmetry=json.loads((tmp_path/"symmetry_metrics.json").read_text())
    params=json.loads((tmp_path/"parameter_metrics.json").read_text())
    assert result["symmetry_rows"]==2
    assert all(row["equivariance_rms"]<1e-6 for row in symmetry["rows"])
    assert params["independent_parents"]==1
    assert {r["parameter"] for r in params["rows"]}=={"kappa","reaction_rate"}
    assert all(r["reference_accepted"] and r["tangent_error"] is not None for r in params["rows"])
    assert all(r["teacher_tangent_halving_change_rms"]>=0 for r in params["rows"])
