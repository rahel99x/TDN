"""Bounded adjacent execution and strict immutable science manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from tdn.analysis.premix.neural import _RunBudget
from tdn.research.protocol import file_digest
from tdn.runtime.metadata import software_metadata, write_json
from .core import Context, clean, validate_row, write_reviews
from .protocol import digest, validate_protocol

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "science_manifest.json"
EXCLUDED = {MANIFEST, "stage.json", "execution.json", "workflow-seal.json", "COMPLETED"}
REQUIRED = {"summary.json", "protocol.json", "rows.jsonl", "rows.json", "review.csv", "review.md", "summary.txt"}


def software_identity(software=None):
    software = software_metadata() if software is None else software
    return {key: software.get(key) for key in ("python", "executable", "venv", "torch", "numpy", "scipy",
            "torch_cuda_runtime", "git_commit", "source_tree_sha256")}


def _inside(value):
    path = Path(value)
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()) or
        any(p.is_symlink() for p in path.absolute().parents if p.is_relative_to(ROOT))):
        raise ValueError("Adjacent artifacts must stay inside this project without symlinks")
    return path.resolve()


def inventory(path):
    result = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("Adjacent science contains a symlink")
        relative = file.relative_to(path).as_posix()
        if file.is_file() and relative not in EXCLUDED:
            result[relative] = file_digest(file)
    return result


def verify_science(protocol, path, *, source_tree_sha256=None):
    validate_protocol(protocol)
    path = _inside(path)
    if (path / MANIFEST).is_symlink() or (path / "COMPLETED").is_symlink():
        raise ValueError("Unsafe adjacent manifest or marker")
    manifest = json.loads((path / MANIFEST).read_text())
    stage = manifest.get("stage")
    if (manifest.get("schema") != "tdn.adjacent-science/v1" or stage not in protocol["units"] or
        manifest.get("protocol_sha256") != digest(protocol)):
        raise ValueError("Adjacent manifest protocol or unit differs")
    unit = protocol["units"][stage]
    if manifest.get("kind") != unit["kind"] or manifest.get("unit_sha256") != digest(unit):
        raise ValueError("Adjacent unit scope differs")
    if source_tree_sha256 is not None and manifest.get("source_tree_sha256") != source_tree_sha256:
        raise ValueError("Adjacent source differs from expected source")
    if manifest.get("software", {}).get("source_tree_sha256") != manifest.get("source_tree_sha256"):
        raise ValueError("Adjacent software identity differs from source")
    if (path / "COMPLETED").read_text().strip() != digest(manifest):
        raise ValueError("Adjacent completion marker differs")
    if not REQUIRED <= set(manifest.get("artifacts", {})):
        raise ValueError("Adjacent prerequisite has incomplete evidence")
    for name in manifest["artifacts"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Unsafe adjacent artifact path")
    if inventory(path) != manifest["artifacts"]:
        raise ValueError("Adjacent scientific artifact inventory or bytes changed")
    if json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Stored adjacent declaration differs")
    summary = json.loads((path / "summary.json").read_text())
    if (summary.get("status") != "COMPLETED" or summary.get("schema") != protocol["schema"] or
        summary.get("stage") != stage or summary.get("protocol_sha256") != digest(protocol) or
        summary.get("source_tree_sha256") != manifest.get("source_tree_sha256") or
        summary.get("device") != manifest.get("device")):
        raise ValueError("Only completed correctly bound science satisfies a prerequisite")
    expected_prior = set(unit["dependencies"])
    actual_prior = set(manifest.get("prerequisites", {}))
    if not (actual_prior <= expected_prior if unit["kind"] == "report" else actual_prior == expected_prior):
        raise ValueError("Adjacent prerequisite scope differs")
    rows = [validate_row(json.loads(line)) for line in (path / "rows.jsonl").read_text().splitlines() if line]
    if json.loads((path / "rows.json").read_text()).get("rows") != rows:
        raise ValueError("Adjacent streamed and compact ledgers differ")
    if len(rows) != summary.get("experiment_count") or len({r["experiment_id"] for r in rows}) != len(rows):
        raise ValueError("Adjacent ledger count or identity differs")
    if any(r["protocol_sha256"] != digest(protocol) or r["stage"] != stage or
           not set(r["mechanism_ids"]) <= set(protocol["mechanisms"]) or r["combination_ids"] for r in rows):
        raise ValueError("Adjacent row belongs to another scope")
    return manifest


def _dispatch(ctx):
    kind = ctx.unit["kind"]
    if kind == 'audit':
        return _audit(ctx)
    if kind == 'roughness':
        return _roughness(ctx)
    if kind == 'diagnostic':
        if ctx.stage == 'D07':
            from .temporal import run
            return run(ctx)
        from .diagnostics import run_diagnostic
        result = run_diagnostic(ctx.stage, ctx.protocol['profile'], ctx.path,
            protocol=ctx.protocol, check_budget=ctx.budget.check)
        _record_result(ctx, ctx.stage, result, 'diagnostic_rows.json')
        return result
    if kind == "explore":
        from .prototypes import run
        return run(ctx)
    if kind == "report":
        from .report import run
        return run(ctx)
    if kind in ('train', 'confirm'):
        import copy
        from .pilots import run_pilot, evaluate_pilot
        execution = copy.deepcopy(ctx.protocol)
        execution['pilot']['tracks'] = ctx.unit['tracks']
        pilot_unit = ctx.stage if kind == 'train' else ctx.unit['pilot_unit']
        execution['pilot']['max_seconds'] = ctx.protocol['units'][pilot_unit]['seconds'] - 15
        evaluation_unit = next(v for v in ctx.protocol['units'].values() if v.get('pilot_unit')==pilot_unit)
        execution['pilot']['evaluation_seconds'] = evaluation_unit['seconds'] - 15
        if kind == 'train':
            result = run_pilot(execution, ctx.path, device=ctx.device, stop=ctx.budget.check)
        else:
            result = evaluate_pilot(execution, ctx.path,
                ctx.prerequisites[ctx.unit['pilot_unit']], device=ctx.device, stop=ctx.budget.check)
        _record_result(ctx, 'C01', result, 'pilot-summary.json' if kind == 'train' else 'evaluation-summary.json')
        return result
    raise ValueError('Unknown adjacent stage kind')


def _record_result(ctx, mechanism, result, source):
    """Keep scientific interpretation separate from successful orchestration."""
    from .core import check
    source_file = ctx.path/source
    evidence = [{'path': source, 'sha256': file_digest(source_file)}] if source_file.is_file() else []
    compact={k:v for k,v in (result or {}).items() if k not in ('rows','timings','curves','catalog')}
    ctx.record(ctx.stage+'/summary', [mechanism], metrics=clean(compact),
        checks=[check('mathematical-generalization', None, None, category='math',
                      reason='See the separately measured identities and raw reference uncertainties; a completed diagnostic is not a proof'),
                check('primary-scientific-claim', None, None, category='gap',
                      reason='Representation, fitting and complete cost require their own reviewed effects and precision gates')],
        evidence=evidence)


def _audit(ctx):
    import torch
    from tdn.numerics import Equation, Geometry
    from tdn.analysis.roadmap.numerics import quadrature
    from .models import interaction_channels, make_model
    from .fields import make_parent, sample_field
    from .core import check
    parent = make_parent(ctx.protocol['seed'], .5, levels=2, rms=.04)
    u = sample_field(parent, 16)
    geo, eq = Geometry((16,16),(1.,1.)), Equation(.004,3.)
    for track in ctx.protocol['tracks']:
        for nodes in (2,4):
            ctx.budget.check()
            reference = interaction_channels(u,.12,eq,geo,track=track,nodes=nodes,implementation='reference')
            shared = interaction_channels(u,.12,eq,geo,track=track,nodes=nodes,implementation='shared')
            rule = make_model('quad2_fixed' if nodes==2 else 'quad4_fixed',track,dict(modes=4)).double()
            model = make_model('channel_fixed',track,dict(modes=4,quad_nodes=nodes)).double()
            error = float((reference-shared).abs().max())
            baseline = float((model(u,.12,eq,geo)-rule(u,.12,eq,geo)).abs().max())
            positions, weights = quadrature(nodes)
            moment = max(abs(sum(w*s**j for s,w in zip(positions,weights))-1/(j+1)) for j in range(2*nodes))
            ctx.record(f'identity/{track}/{nodes}', ['C01','D05'],
                metrics=dict(shared_reference_error=error, fixed_baseline_error=baseline, quadrature_moment_error=moment),
                checks=[check('polarization-shared-identity',error,2e-13,category='math'),
                        check('normalized-fixed-recombination',baseline,2e-13,category='correctness'),
                        check('gauss-moments',moment,2e-14,category='math')])
        for family in ('channel_fixed','channel_global','channel_affine','channel_neural','feature_capacity'):
            ctx.budget.check()
            model = make_model(family,track,dict(modes=4)).double()
            nulls = {'zero_time':float((model(u,0.,eq,geo)-u).abs().max())}
            for key,equation in [('zero_reaction',Equation(.004,0.)),('zero_diffusion',Equation(0.,3.))]:
                nulls[key]=float(model.correction_components(u,.12,equation,geo)['increment'].abs().max())
            nulls['constant']=float(model.correction_components(torch.full_like(u,.43),.12,eq,geo)['increment'].abs().max())
            ctx.record(f'null/{track}/{family}', ['C01'], metrics=nulls, checks=[
                check(key,value,0.,category='math') for key,value in nulls.items()], config=model.architecture_metadata())
    return {'status':'COMPLETED','scope':'FP64 algebra/implementation checks, not a statistical advantage or GPU validation'}


def _roughness(ctx):
    """All ten requested alpha labels; refinement and bandwidth remain separate."""
    import numpy as np
    from .fields import make_parent, sample_field, roughness_metrics, with_bandwidth
    from .core import check
    from tdn.analysis.frontier.numerics import heat_step
    from tdn.numerics import Equation, Geometry
    rows, parents, arrays = [], {}, {}
    seed = ctx.protocol['seed'] + 801
    for alpha in ctx.protocol['roughness']['alpha_levels']:
        for generator in ('ridge','multiscale_2d'):
            for phase in ('structured','random'):
                original = make_parent(seed, alpha, generator=generator, phase_mode=phase,
                    levels=2, rms=.04, split='roughness-development')
                for parent in (original,with_bandwidth(original,3)):
                    parents[parent.parent_id] = parent.to_dict()
                    for n in (16,32):
                        ctx.budget.check()
                        u=sample_field(parent,n); geo=Geometry((n,n),(1.,1.))
                        for horizon in (0.,.02,.12):
                            v=heat_step(u,horizon,Equation(.004,0.),geo,'continuum')
                            key=f'field_{len(rows):04d}'
                            record=dict(parent_id=parent.parent_id,field_cluster=parent.cluster_id,
                                alpha=alpha,generator=generator,phase_mode=phase,grid=n,levels=parent.levels,
                                dimension_label=parent.dimension_label,workload=parent.workload,
                                horizon=horizon,dynamics='analytic heat-only smoothing diagnostic, not full reaction-diffusion',
                                array_key=key if horizon==0 and n==16 else None,metrics=roughness_metrics(v))
                            rows.append(record)
                            if horizon==0 and n==16: arrays[key]=v.numpy()
    write_json(ctx.path/'roughness_rows.json',rows)
    write_json(ctx.path/'roughness_parents.json',parents)
    np.savez_compressed(ctx.path/'roughness_fields.npz',**arrays)
    count=len({r['alpha'] for r in rows})
    ctx.record('roughness/sweep',['R01'],metrics=dict(cases=len(rows),alpha_levels=count,
        independent_clusters=len({r['field_cluster'] for r in rows}),statistical_scope='one random parent family plus deterministic ridge controls; variants are paired'),
        checks=[check('all-ten-labels',count,10,'eq',category='correctness'),
                check('finite-fields-are-smooth',True,True,'eq',category='math'),
                check('predictive-value-of-roughness',None,None,reason='Field validation does not establish a solver advantage')])
    return dict(status='COMPLETED',cases=len(rows),alpha_levels=count)


def run_stage(protocol, stage, path, *, prerequisites=None, device="cpu", stop=None,
              resume=False, stage_failures=None):
    validate_protocol(protocol)
    if stage not in protocol["units"] or device not in ("cpu", "cuda"):
        raise ValueError("Invalid adjacent stage or device")
    unit = protocol["units"][stage]
    if protocol["profile"] == "full":
        if os.environ.get("TDN_EXECUTION_MODE") != "desktop-slurm" or device != unit["device"]:
            raise ValueError("Full adjacent requires its native allocated Fedora device")
        from tdn.runtime.preflight import verify_runtime
        verify_runtime(device, "adjacent-" + stage)
    if resume and unit["kind"] not in ("train", "confirm"):
        raise ValueError("Only journaled train and confirm units support interrupted resumption")
    path = _inside(path); path.mkdir(parents=True, exist_ok=True)
    if (path / "COMPLETED").exists() or (path / MANIFEST).exists():
        raise FileExistsError("Preserve completed adjacent evidence; use a fresh run")
    if not resume and any((path / name).exists() for name in ("summary.json", "rows.jsonl")):
        raise FileExistsError("Preserve the previous attempt; use a fresh run or verified resumption")
    if (path / "protocol.json").exists() and json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Worker and engine protocol differ")
    prior = {key: _inside(value) for key, value in (prerequisites or {}).items()}
    required = set(unit["dependencies"])
    if not (set(prior) <= required if unit["kind"] == "report" else set(prior) == required):
        raise ValueError("Missing or unexpected adjacent prerequisites")
    software = software_metadata(); source = software["source_tree_sha256"]
    stable_software = software_identity(software)
    hashes = {}; lineage = {}
    for key, base in prior.items():
        if path == base or path.is_relative_to(base) or base.is_relative_to(path):
            raise ValueError("Adjacent stages must have separate directories")
        sealed = verify_science(protocol, base, source_tree_sha256=source)
        if sealed["stage"] != key:
            raise ValueError("Adjacent prerequisite unit differs")
        if sealed.get("software") != stable_software:
            raise ValueError("Adjacent prerequisite software differs")
        hashes[key] = file_digest(base / MANIFEST)
    # Cross-check every ancestor identity visible among immediate prerequisites.
    for key, base in prior.items():
        sealed = json.loads((base / MANIFEST).read_text())
        inherited = {**sealed.get("lineage", {}), key: hashes[key]}
        for name, sha in inherited.items():
            if name in lineage and lineage[name] != sha:
                raise ValueError("Adjacent prerequisites mix frozen lineages")
            lineage[name] = sha
    if resume:
        prior_summary = json.loads((path / "summary.json").read_text())
        if (prior_summary.get("stage") != stage or prior_summary.get("protocol_sha256") != digest(protocol) or
            prior_summary.get("source_tree_sha256") != source or prior_summary.get("software") != stable_software or
            prior_summary.get("prerequisites") != hashes or prior_summary.get("device") != device or
            prior_summary.get("status") not in ("FAILED", "INTERRUPTED")):
            raise ValueError("Interrupted scientific attempt has incompatible identity")
    write_json(path / "protocol.json", protocol)
    (path / "rows.jsonl").touch()
    start = time.monotonic(); budget = _RunBudget(unit["seconds"], stop, device)
    ctx = Context(protocol, stage, path, prior, device, budget, resume=resume, stage_failures=stage_failures)
    try:
        extras = _dispatch(ctx)
        correctness = sum(c["required"] and c["category"] in ("correctness", "math") and c["verdict"] == "BAD"
                          for row in ctx.rows for c in row["checks"])
        if unit["kind"] == "audit" and correctness:
            raise RuntimeError(f"{correctness} required adjacent mathematical checks failed")
        budget.check()
        if software_metadata()["source_tree_sha256"] != source:
            raise ValueError("Executable source changed during adjacent execution")
        for key, base in prior.items():
            verify_science(protocol, base, source_tree_sha256=source)
            if file_digest(base / MANIFEST) != hashes[key]:
                raise ValueError("Adjacent prerequisite changed during execution")
        verdicts = write_reviews(path, ctx.rows)
        summary = dict(schema=protocol["schema"], stage=stage, kind=unit["kind"], profile=protocol["profile"],
            status="COMPLETED", device=device, source_tree_sha256=source, protocol_sha256=digest(protocol),
            software=stable_software, prerequisites=hashes,
            elapsed_seconds=time.monotonic()-start, experiment_count=len(ctx.rows), verdict_counts=verdicts,
            correctness_failures=correctness, memory=budget.memory,
            scientific_outcome="BOUNDED_MEASURED_EVIDENCE; SEE GOOD/BAD/NA CHECKS", details=clean(extras or {}))
        write_json(path / "summary.json", summary)
        (path / "summary.txt").write_text(f"TDN adjacent {stage}: COMPLETED; {len(ctx.rows)} rows; {summary['elapsed_seconds']:.2f}s / {unit['seconds']}s\n"
            f"Verdicts: {verdicts}\nScores measure declared check attainment, not proof or a universal ranking.\n")
        manifest = dict(schema="tdn.adjacent-science/v1", stage=stage, kind=unit["kind"], device=device,
            protocol_sha256=digest(protocol), unit_sha256=digest(unit), source_tree_sha256=source,
            prerequisites=hashes, lineage=lineage, software=stable_software, artifacts=inventory(path))
        write_json(path / MANIFEST, manifest)
        (path / "COMPLETED").write_text(digest(manifest)+"\n")
        return summary
    except BaseException as error:
        verdicts = write_reviews(path, ctx.rows)
        status = "INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else "FAILED"
        write_json(path / "summary.json", dict(schema=protocol["schema"], stage=stage, status=status,
            protocol_sha256=digest(protocol), source_tree_sha256=source, device=device,
            software=stable_software, prerequisites=hashes,
            experiment_count=len(ctx.rows), elapsed_seconds=time.monotonic()-start, verdict_counts=verdicts,
            error=f"{type(error).__name__}: {error}", scientific_outcome="PARTIAL_EVIDENCE_RETAINED"))
        (path / "summary.txt").write_text(f"TDN adjacent {stage}: {status}; {error}\nPartial rows: {len(ctx.rows)}\n")
        raise
