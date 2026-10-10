"""Resolution learning uses prepared data, exact recovery and explicit development scope."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.adjacent import resolution_learning as learning
from tdn.analysis.adjacent.resolution_protocol import build_resolution_protocol
from tdn.analysis.adjacent.models import make_model
from tdn.numerics import Equation, Geometry


class Budget:
    def check(self):
        return None


def context(protocol, unit, path, prerequisites=None, budget=None):
    return SimpleNamespace(protocol=protocol, unit=protocol["units"][unit], stage=unit,
        path=Path(path), prerequisites=prerequisites or {}, device="cpu", budget=budget or Budget())


def fixture_protocol():
    p = build_resolution_protocol("resolution-smoke")
    cfg = p["resolution"]
    cfg.update(tracks=["discrete"], frozen_families=["quad2_fixed", "channel_fixed", "df"],
               endpoint_steps=[1, 2], additional_classical_steps=[], cross_grid_transfer=True)
    cfg["training"].update(updates=2, validation_every=1, checkpoint_every=1, fit_iterations=4)
    units = {}
    for grid in (64, 128):
        name = f"train-{grid}"
        units[name] = dict(id=name, kind="resolution_train", grid=grid, track="discrete",
            families=["channel_global", "channel_neural"], seeds=[871001],
            bank_units=[f"bank-train-{grid}", f"bank-validation-{grid}"])
        name = f"evaluate-{grid}"
        units[name] = dict(id=name, kind="resolution_evaluate", grid=grid, track="discrete", field_indices=[0, 1],
            bank_units=[f"bank-evaluation-{grid}"], freeze_unit="freeze")
    units["freeze"] = dict(id="freeze", kind="resolution_freeze")
    units["aggregate"] = dict(id="aggregate", kind="resolution_aggregate", evaluation_units=["evaluate-64", "evaluate-128"])
    p["units"] = units
    return p


@pytest.fixture
def prepared_banks(monkeypatch, tmp_path):
    """Manufactured targets test data plumbing, not teacher convergence or science."""
    from tdn.analysis.adjacent import resolution_diagnostics as refs
    torch.set_num_threads(1)
    bank_rows, data = {}, {}
    for grid in (64, 128):
        axis = torch.arange(grid, dtype=torch.float64) / grid
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        eq, geom = Equation(.004, 3.), Geometry((grid, grid), (1., 1.))
        base = make_model("df", "discrete", dict(modes=8, split_modes=4, quad_nodes=2))
        corrected = make_model("quad2_fixed", "discrete", dict(modes=8, split_modes=4, quad_nodes=2))
        for split, count in (("train", 1), ("validation", 1), ("evaluation", 2)):
            directory = tmp_path / f"bank-{split}-{grid}"
            directory.mkdir()
            values = []
            for index in range(count):
                parent = f"fixture-{split}-{index}"
                phase = {"train": .2, "validation": .6, "evaluation": 1.1}[split] + .3 * index
                initial = (.43 + .04 * torch.cos(2 * math.pi * (x + y) + phase) +
                           .02 * torch.cos(2 * math.pi * (9 * x + y) - phase))[None, None]
                with torch.no_grad():
                    physical = base(initial, .04, eq, geom)
                    target = physical + 1.2 * (corrected(initial, .04, eq, geom) - physical)
                meta = dict(parent_id=parent, field_cluster=parent, split=split, index=index,
                    grid=grid, track="discrete", horizon=.04, accepted=True, reference_accepted=True,
                    uncertainty_rms=1e-12, uncertainty_max_bound=1e-12,
                    kappa=.004, reaction_rate=3., domain=[1., 1.], regime="manufactured_test_fixture",
                    alpha=.3, generator="test_fixture", parent=dict(parent_id=parent),
                    bank_dir=str(directory))
                values.append(meta); data[(parent, grid, .04)] = (initial, target, meta)
            bank_rows[str(directory)] = values
    monkeypatch.setattr(refs, "resolution_parent", lambda protocol, split, index, grid: dict(parent_id=f"fixture-{split}-{index}"))
    monkeypatch.setattr(refs, "iterate_bank_entries", lambda directory: copy.deepcopy(bank_rows[str(directory)]))
    def load(directories, parent, track, grid, horizon):
        assert track == "discrete"
        return data[(parent, grid, horizon)]
    monkeypatch.setattr(refs, "load_bank_entry", load)
    monkeypatch.setattr(refs, "converged_reference", lambda *args, **kwargs: pytest.fail("Learning must never create a CPU reference"))
    return bank_rows, data


def test_same_physical_cutoffs_and_trial_budget_bound():
    p = build_resolution_protocol("resolution-full")
    cfg, settings = learning._options(p)
    assert cfg["model_config"]["modes"] == 8
    assert cfg["model_config"]["split_modes"] == 4
    assert settings["trial_seconds"] == 20
    assert len(settings["optimizer_controls"]) == 2
    assert learning._config(cfg, "analytic_quad_cubic")["quad_nodes"] == 4
    assert learning._config(cfg, "channel_neural")["quad_nodes"] == 2
    # Largest ours shard: 24 neural optimizer trials and four non-neural fits.
    assert 28 * settings["trial_seconds"] < 900


def test_larger_profiles_require_real_cuda_and_missing_banks_fail(tmp_path):
    p = build_resolution_protocol("resolution-full")
    unit = next(k for k, u in p["units"].items() if u["kind"] == "resolution_train")
    with pytest.raises(ValueError, match="real native CUDA"):
        learning.run_train(context(p, unit, tmp_path))
    p = fixture_protocol()
    with pytest.raises(ValueError, match="Missing prepared"):
        learning.run_train(context(p, "train-64", tmp_path))


def test_prepared_training_freeze_transfer_aggregate_and_recovery(tmp_path, prepared_banks, monkeypatch):
    p = fixture_protocol()
    train_dirs = {}
    for grid in (64, 128):
        unit = f"train-{grid}"
        prerequisites = {name: tmp_path / name for name in p["units"][unit]["bank_units"]}
        path = tmp_path / unit
        ctx = context(p, unit, path, prerequisites)
        result = learning.run_train(ctx)
        assert result["model_count"] == 2 and result["trial_count"] == 2
        assert result["scientific_outcome"] == "DEVELOPMENT_ONLY"
        assert all(r["train_grid"] == grid and r["scientific_verdict"] == "NA" for r in result["rows"])
        assert all(r["initial_checkpoint_sha256"] for r in result["rows"])
        train_dirs[unit] = path
        recovered = learning.run_train(ctx)
        assert recovered["rows"] == result["rows"]
    freeze_dir = tmp_path / "freeze"
    frozen = learning.run_freeze(context(p, "freeze", freeze_dir, train_dirs))
    assert frozen["model_count"] == 4
    (freeze_dir / "execution.json").write_text('{"status":"COMPLETED"}')
    learning.verify_freeze(p, freeze_dir)
    evaluation_dirs = {}
    for grid in (64, 128):
        unit = f"evaluate-{grid}"
        prerequisites = {"freeze": freeze_dir, f"bank-evaluation-{grid}": tmp_path / f"bank-evaluation-{grid}"}
        ctx = context(p, unit, tmp_path / unit, prerequisites)
        result = learning.run_evaluate(ctx)
        assert result["teacher_work_in_gpu_job"] is False
        assert result["endpoint_count"] == (20 if grid == 64 else 28)
        assert any(r["family"] == "channel_fixed" for r in result["rows"])
        assert all(r["input_shape"] == [1, 1, grid, grid] for r in result["rows"])
        if grid == 128:
            transfers = [r for r in result["rows"] if r["transfer"]]
            assert len(transfers) == 8
            assert all(r["train_grid"] == 64 and r["grid"] == 128 for r in transfers)
        with monkeypatch.context() as patches:
            patches.setattr(learning, "measure_paired", lambda *args, **kwargs: pytest.fail("Complete timing groups cannot be retimed"))
            assert learning.run_evaluate(ctx)["rows"] == result["rows"]
        evaluation_dirs[unit] = tmp_path / unit
    aggregate = context(p, "aggregate", tmp_path / "aggregate", {"freeze": freeze_dir, **evaluation_dirs})
    result = learning.run_aggregate(aggregate)
    assert result["endpoint_count"] == 48
    assert result["independent_parent_count"] == 2
    assert all(r["scientific_verdict"] == "NA" for r in result["rows"])
    path = evaluation_dirs["evaluate-64"] / "evaluation-rows.json"
    values = json.loads(path.read_text()); values.pop()
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="missing, extra or duplicate"):
        learning.run_aggregate(aggregate)


def test_frozen_checkpoint_mutation_and_missing_model_inventory_fail(tmp_path, prepared_banks):
    p = fixture_protocol(); p["units"] = {key: value for key, value in p["units"].items() if key in {"train-64", "freeze"}}
    train = tmp_path / "train"
    ctx = context(p, "train-64", train, {name: tmp_path / name for name in p["units"]["train-64"]["bank_units"]})
    learning.run_train(ctx)
    path = train / "catalog.json"; rows = json.loads(path.read_text()); removed = rows.pop(); path.write_text(json.dumps(rows))
    freeze = context(p, "freeze", tmp_path / "frozen", {"train-64": train})
    with pytest.raises(ValueError, match="model inventory"):
        learning.run_freeze(freeze)
    rows.append(removed); path.write_text(json.dumps(rows))
    learning.run_freeze(freeze)
    model = json.loads((freeze.path / "catalog.json").read_text())[0]
    (freeze.path / model["checkpoint"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Frozen resolution artifact changed"):
        learning.verify_freeze(p, freeze.path)


def test_gradient_interruption_resumes_optimizer_and_retains_attempt(tmp_path, prepared_banks):
    p = fixture_protocol(); p["units"]["train-64"]["families"] = ["channel_neural"]
    ctx = context(p, "train-64", tmp_path / "training", {name: tmp_path / name for name in p["units"]["train-64"]["bank_units"]})
    samples, _ = learning._prepared_samples(ctx, torch.device("cpu"))
    class Interrupt:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise InterruptedError("test interruption")
    ctx.budget = Interrupt()
    control = dict(name="clipped", clip_grad_norm=1., loss_scale=1.)
    with pytest.raises(InterruptedError):
        learning._fit(ctx, "channel_neural", 871001, .001, control, samples, torch.device("cpu"))
    assert list((ctx.path / "interrupted-trials").glob("*.json"))
    assert list((ctx.path / "progress").glob("*.pt"))
    ctx.budget = Budget()
    recovered = learning._fit(ctx, "channel_neural", 871001, .001, control, samples, torch.device("cpu"))
    assert recovered["updates_completed"] == 2
    assert [r["update"] for r in recovered["curves"]] == [0, 1, 2]
    uninterrupted_ctx = context(p, "train-64", tmp_path / "uninterrupted", ctx.prerequisites)
    uninterrupted = learning._fit(uninterrupted_ctx, "channel_neural", 871001, .001, control, samples, torch.device("cpu"))
    recovered_weights = torch.load(ctx.path / recovered["checkpoint"], weights_only=True)["state_dict"]
    uninterrupted_weights = torch.load(uninterrupted_ctx.path / uninterrupted["checkpoint"], weights_only=True)["state_dict"]
    assert recovered_weights.keys() == uninterrupted_weights.keys()
    assert all(torch.equal(value, uninterrupted_weights[name]) for name, value in recovered_weights.items())


def test_distinct_field_batch_costs_and_missing_batch_inventory(tmp_path, prepared_banks):
    bank_rows, data = prepared_banks
    p = fixture_protocol(); cfg = p["resolution"]
    cfg["splits"]["evaluation"] = 4
    cfg["timing"].update(batches=[1, 4], repeats=2)
    p["units"] = {key: value for key, value in p["units"].items() if key in {"train-64", "freeze", "evaluate-64", "aggregate"}}
    p["units"]["evaluate-64"]["field_indices"] = [0, 1, 2, 3]
    p["units"]["aggregate"]["evaluation_units"] = ["evaluate-64"]
    directory = str(tmp_path / "bank-evaluation-64")
    for index in (2, 3):
        initial, target, source = data[(f"fixture-evaluation-{index - 2}", 64, .04)]
        meta = {**source, "parent_id": f"fixture-evaluation-{index}", "field_cluster": f"fixture-evaluation-{index}", "index": index}
        bank_rows[directory].append(meta)
        # Manufactured, explicitly non-scientific fixture targets. Each field
        # is distinct, with a different small perturbation to its stored state.
        data[(meta["parent_id"], 64, .04)] = (initial + index * .0001, target + index * .0001, meta)
    training = context(p, "train-64", tmp_path / "train", {name: tmp_path / name for name in p["units"]["train-64"]["bank_units"]})
    learning.run_train(training)
    frozen = context(p, "freeze", tmp_path / "freeze", {"train-64": training.path})
    learning.run_freeze(frozen)
    evaluation = context(p, "evaluate-64", tmp_path / "eval", {"freeze": frozen.path, "bank-evaluation-64": Path(directory)})
    result = learning.run_evaluate(evaluation)
    batch = [r for r in result["rows"] if r["batch_size"] == 4]
    assert len(batch) == 40
    assert all(r["cost_seconds"] == r["batch_cost_seconds"] / 4 for r in batch)
    assert all(r["workload"] == "distinct_field_throughput" for r in batch)
    aggregate = context(p, "aggregate", tmp_path / "aggregate", {"freeze": frozen.path, "evaluate-64": evaluation.path})
    assert learning.run_aggregate(aggregate)["endpoint_count"] == 80
    path = evaluation.path / "evaluation-rows.json"
    path.write_text(json.dumps([r for r in result["rows"] if r["batch_size"] == 1]))
    with pytest.raises(ValueError, match="missing, extra or duplicate"):
        learning.run_aggregate(aggregate)
