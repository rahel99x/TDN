"""Source-traced tangent research tables and portable Raven projects.

Reporting never loads binary teachers/checkpoints or revises a gate. Every
registered unit/method remains visible when its evidence is missing or stopped.
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import SimpleNamespace

from tdn.analysis.response import diagnostic_metrics as shared
from tdn.analysis.response import diagnostic_report as portable
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json

SCHEMA = "tdn.tangent-response-research/v1"
BUNDLE = "tangent-response-raven.zip"
CONFIG = dict(project_id="TDN-tangent-response", experiment_prefix="tangent-response",
    name_prefix="TDN/tangent-response", catalog_schema="tdn.tangent-response-raven-catalog/v1",
    parameter_keys=("family", "arm", "target_kind", "seed", "phase", "basis_rank", "updates"),
    metric_scope="Tangent mechanism, response actions, physical fitting/validation, precision, rollout, complete costs and worker resources",
    readme="# Tangent-response research\n\nRegister this directory with `:project PATH`. The overview preserves every registered method, missing unit and negative gate. Large campaigns use page-000 and later separate detail projects below Raven's discovery cap. Native attempts carry actual worker job IDs; the aggregate overview has none.\n\nOnly research metadata, CSV/JSON, plots and retained resource streams are exported. Binary teachers/checkpoints/journals remain in the native run; the omission inventory retains their hashes. CPU implementation checks do not establish CUDA readiness or scientific promotion.\n")
RESEARCH_NAMES = {"mechanism_results.json", "compression_results.json", "training_results.json", "trial.json",
    "evaluation_results.json", "performance_results.json", "decision.json", "calibration.json",
    "learning_curves.jsonl", "query_plan.json", "query_registry.json", "reference_summary.json",
    "data_manifest.json", "fit_screen.json", "screening.json", "mathematical_bound.json", "work.json", "summary.json"}
METHODS = ("anchor", "G", "U", "fixed_shared_basis", "learned_response", "residual_fno")
LARGE_SCIENCE_KINDS = {"compression_results.json": "compression"}
DEFINITIONS = {
    "RMS / MSE": "Physical root-mean-square / mean-squared field error. FD uses nodal weighting; Galerkin uses its physical interpolant and Nyquist weights. Paired descendants are not independent parents.",
    "maximum": "Largest sampled absolute physical error; not a continuous supremum certificate.",
    "mean / centered": "Spatial-average error and error remaining after mean removal. Their squared contributions and signed cross terms retain their actual source definitions.",
    "G / U": "Matched anchor-mean response / spatially varying tangent response on the same smooth FP64 anchor and physical residual.",
    "action / rank": "Matrix-free response applied to a predeclared source direction / retained response-factor dimension. Probe count is work, not independent population count; vector fit rank is not operator rank.",
    "uncertainty / headroom": "Empirical teacher/path/tangent refinement estimates and measurable signal above those estimates/output rounding. Not certified bounds or confidence intervals. All requested/null/rejected slots remain in raw source coverage.",
    "conditioning": "Singular values, declared cutoff/ridge, factor/Gram scales and sensitivity in actual deployment precision. Conditional least squares does not prove a nonlinear global optimum.",
    "training / validation": "Full-parent physical objectives and actual optimizer commits. Validation selects deployment; initialization/final/best-training fits remain distinct. No old 100-fold rule is applied.",
    "physical / arithmetic bridge": "Teacher64 minus smooth anchor64 / smooth anchor64 minus deployed FP32 anchor. Final FP32 addition and numerical remainder are separately named and costed.",
    "rollout / composition": "Re-encoded autonomous macrosteps and intermediate teachers / a separately measured composition difference. Stable finite values do not imply accuracy.",
    "cost / telemetry": "Offline labels, optimizer/stage/worker/allocation and complete inference scopes remain separate. CPU/RSS snapshots exclude departed/remote processes; GPU samples are allocated-device counters, not per-process VRAM accounting. Null means unavailable; zero is observed. No energy/monetary inference from latency.",
    "CPU utilization / RSS / VRAM": "CPU percent uses a single logical core as 100% and may exceed 100%; process-tree changes leave interval utilization unavailable. RSS is resident host memory and can count shared pages repeatedly. VRAM is whole assigned-device memory, including any display/co-resident processes.",
    "public observations / optimizer journal": "Scheduled learning rows are exported observations, not every private optimizer commit. Resource charts use at most 256 uniform observations per retained worker stream; full raw JSONL remains copied and cited. Whole-history optimizer summaries retain their producer's exact scope.",
    "PASS / STOP / UNRESOLVED / IMPLEMENTATION_ONLY": "The unchanged serialized prospective gate / negative evidence / insufficient valid evidence / reduced software verification with no native promotion.",
}


def start_attempt(ctx, job_id=None, sourcehash=None, *, worker_managed=False):
    return shared.start_attempt(ctx, job_id=job_id, sourcehash=sourcehash or getattr(ctx, "sourcehash", None),
                                worker_managed=worker_managed, config=CONFIG)


def finish_attempt(ctx, summary):
    from . import report_artifacts as artifacts
    ctx._raven_config = CONFIG
    names=RESEARCH_NAMES|{"summary.json", "execution.json", "protocol.json"}
    unit=getattr(ctx,"unit",ctx.protocol["units"][ctx.stage])
    large={name for name,kind in LARGE_SCIENCE_KINDS.items() if unit.get("kind")==kind
           and (Path(ctx.path)/name).is_file() and (Path(ctx.path)/name).stat().st_size>artifacts.PAGE_BYTES}
    # Publish truthful worker/body status before the optional bulk export. In
    # particular, a failed body without a science seal must still end FAILED.
    path = shared.finish_attempt(ctx, summary, metadata_files=names-large)
    large_index=[]
    if path is not None and large:
        try:large_index=_finish_large_science(ctx,large)
        except (OSError,ValueError,KeyError,TypeError) as error:
            if summary.get("status") not in ("FAILED","CANCELLED","TIMEOUT"):raise
            large_index=_finish_unsealed_science(ctx,large,str(error))
    if path is not None and large_index:
        source_index,_,_=shared.read_metadata(path/"outputs/source_index.json",root=path)
        source_index["sources"].extend(large_index);source_index["scope"]="Research metadata and explicit bulk-source descriptors; sealed bulk bytes remain native-only here and are retained once by the final research overview; unsealed failure evidence is separately named; binary replay is not included"
        write_json(path/"outputs/source_index.json",source_index)
        contract,_,_=shared.read_metadata(path/"outputs/contract.json",root=path)
        entries={row["path"]:row for row in contract["outputs"]}
        for source in large_index:
            entries[source["path"]]=dict(path=source["path"],required=True,min_bytes=1,sha256=source["sha256"],format="json")
        write_json(path/"outputs/contract.json",dict(version=1,outputs=list(entries.values())))
    if path is not None and (summary.get("native_worker_finalized") or not ctx.protocol.get("execution_scope") == "NATIVE_RESEARCH"):
        # The terminal observer has stopped. Never hash a still-recording stream
        # in a science CLI's deferred native completion.
        try:_finish_telemetry(ctx, path, summary)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Optional collector publication cannot reverse a scientific result.
            print("TDN worker telemetry publication warning: "+str(error)[:512], file=sys.stderr)
    return path


def _finish_large_science(ctx,names):
    """Verify bulk science without duplicating its bytes in a worker export."""
    from . import report_artifacts as artifacts
    path=shared.attempt_path(ctx)
    if not (path/"run.json").is_file():return []
    manifest,manifest_hash,_=shared.read_metadata(Path(ctx.path)/"science_manifest.json",root=ctx.path)
    marker=shared._inside(ctx.path,Path(ctx.path)/"COMPLETED")
    if (manifest.get("schema")!="tdn.tangent-response-science/v1" or manifest.get("stage")!=ctx.stage
            or manifest.get("profile")!=ctx.protocol["profile"] or manifest.get("protocol_sha256")!=digest(ctx.protocol)
            or not stat.S_ISREG(marker.stat().st_mode) or marker.stat().st_size>128 or marker.read_text().strip()!=manifest_hash
            or manifest.get("source_tree_sha256")!=(getattr(ctx,"_raven_sourcehash",None) or getattr(ctx,"sourcehash",None))):
        raise ValueError("Large science export requires its exact completed science seal")
    directory=path/"outputs/large-science";directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in sorted(names):
        payload=artifacts.ScientificPayload(Path(ctx.path)/name,ctx.path,manifest["artifacts"][name])
        descriptor=dict(schema="tdn.tangent-response-native-source/v1",source_name=name,source_sha256=payload.sha256,
            source_bytes=len(payload),science_manifest_sha256=manifest_hash,science_seal_status="VERIFIED",
            source_tree_sha256=manifest["source_tree_sha256"],protocol_sha256=manifest["protocol_sha256"],
            native_stage=ctx.stage,native_source_token=name,availability="ORIGINAL_NATIVE_PROJECT_ONLY",
            scope="Provenance token, not a local artifact pointer. Original sealed bytes remain in the native science stage. A successful final research overview retains the complete source once in verified lossless segments; this descriptor does not assert that a report has completed.")
        target=directory/(name+".descriptor.json")
        if target.exists():
            prior,_,_=shared.read_metadata(target,root=path)
            if prior!=descriptor:raise ValueError("Large scientific descriptor changed after publication")
        else:write_json(target,descriptor)
        _,checksum,_=shared.read_metadata(target,root=path)
        records.append(dict(path=target.relative_to(path).as_posix(),source_name=name,
            sha256=checksum,source_sha256=payload.sha256,bytes=len(payload),representation="native-source-descriptor",
            availability="ORIGINAL_NATIVE_PROJECT_ONLY",
            science_manifest_sha256=manifest_hash,science_seal_status="VERIFIED"))
    return records


def _finish_unsealed_science(ctx,names,reason):
    """Retain partial bytes as failure evidence, never as sealed science."""
    from . import report_artifacts as artifacts
    path=shared.attempt_path(ctx);directory=path/"outputs/unsealed-science"
    directory.mkdir(parents=True,exist_ok=True)
    records=[];payloads={}
    for name in sorted(names):
        file=shared._inside(ctx.path,Path(ctx.path)/name)
        with file.open("rb") as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>artifacts.SCIENCE_PAYLOAD_BYTES:
                raise ValueError("Unsealed scientific evidence exceeds its bounded regular-file scope")
            checksum=hashlib.sha256()
            for block in iter(lambda:stream.read(artifacts.CHUNK_BYTES),b""):
                checksum.update(block)
                if stream.tell()>artifacts.SCIENCE_PAYLOAD_BYTES:raise ValueError("Unsealed evidence grew beyond its bounded scope")
        payloads[name]=artifacts.ScientificPayload(file,ctx.path,checksum.hexdigest())
    catalog=directory/"sources.json"
    if catalog.exists():
        prior,_,_=shared.read_metadata(catalog,root=path)
        if {r["source_name"]:r["source_sha256"] for r in prior["sources"]}!={name:payload.sha256 for name,payload in payloads.items()}:
            raise ValueError("Unsealed failure evidence changed after publication")
        for source in prior["sources"]:
            for _ in artifacts.source_bytes(directory,dict(path=source["local_path"],representation="exact-segments",
                    source_descriptor_sha256=source["sha256"],sha256=source["source_sha256"],bytes=source["bytes"])):pass
        return prior["sources"]
    pages=artifacts.Pages(directory,shared.stable_id(path.name+"-unsealed"),CONFIG,
        dict(source_tree_sha256=getattr(ctx,"sourcehash",None),protocol_sha256=digest(ctx.protocol)))
    for name,payload in payloads.items():
        index,_=artifacts.source_segments(pages,name,payload)
        records.append(dict(path="outputs/unsealed-science/"+index["path"],local_path=index["path"],source_name=name,
            sha256=index["sha256"],source_sha256=payload.sha256,bytes=len(payload),representation="exact-segments",
            science_manifest_sha256=None,science_seal_status="UNSEALED_FAILURE_EVIDENCE",reason=reason[:2048]))
    projects=pages.finish()
    write_json(catalog,dict(schema="tdn.tangent-response-unsealed-science/v1",sources=records,artifact_projects=projects,
        scope="Exact observed failure bytes only; no valid science seal or successful worker outcome is inferred"))
    return records


def _finish_telemetry(ctx, path, summary):
    index = path/"telemetry/index.json"
    if not index.is_file():return
    payload, _, _ = shared.read_metadata(index, root=path)
    contract, _, _ = shared.read_metadata(path/"outputs/contract.json", root=path)
    files = [index, path/"telemetry/observer.log"]
    for source in payload["sources"]:
        files.extend([path/shared.relative_path(source["path"]), path/"telemetry"/(source["family"]+".csv")])
    entries = {row["path"]:row for row in contract["outputs"]}
    for file in files:
        if not file.is_file():continue
        shared._inside(path, file);relative = file.relative_to(path).as_posix()
        entries[relative] = dict(path=relative, required=True, min_bytes=min(1, file.stat().st_size),
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            format="csv" if file.suffix == ".csv" else "json" if file.suffix == ".json" else "text")
    write_json(path/"outputs/contract.json", dict(version=1, outputs=list(entries.values())))
    native, _, _ = shared.read_metadata(path/"summary.json", root=path)
    native["results"]["worker_telemetry"] = summary.get("worker_telemetry", dict(index="telemetry/index.json",
        requested_families=payload.get("requested_families"), sources=payload["sources"],
        observer_wall_seconds=payload.get("observer_wall_seconds"), observer_cpu_seconds=payload.get("observer_cpu_seconds")))
    write_json(path/"summary.json", native)
    observer = path/"telemetry/observer.log"
    if observer.is_file():shared.register_log(ctx, observer, label="Worker telemetry coverage and warnings", group="Resources")


def attempt_path(ctx):
    ctx._raven_config = CONFIG
    return shared.attempt_path(ctx)


def emit_live(metrics, **kwargs):
    """Publish actual numeric observations in the worker's opt-in context."""
    return shared.emit_live(metrics, **kwargs)


def _flat(value, prefix="", *, depth=0):
    result = {}
    for name, child in value.items():
        key = prefix+name
        if isinstance(child, dict) and depth < 8:result.update(_flat(child, key+".", depth=depth+1))
        elif child is None or isinstance(child, (bool, int, float)) or isinstance(child, str) and len(child) <= 1024:
            result[key] = child
        elif isinstance(child, list):
            result[key+"_count"] = len(child)
            if name in ("singular_values", "column_scales", "coefficients", "gram", "grid", "lengths") and len(child) <= 256:
                result[key] = child
    return result


def _tables(payload, base, pointer="", *, depth=0):
    if depth > 12:return
    if isinstance(payload, dict):
        for name, value in payload.items():
            location = pointer+"/"+name
            if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
                for index, row in enumerate(value):
                    citation = dict(base, source_record=location+"/"+str(index))
                    yield name, dict(_flat(row), **citation)
                    controls = row.get("controls", {})
                    if isinstance(controls, dict):
                        for method, metrics in controls.items():
                            if isinstance(metrics, dict):
                                identity = {key:row.get(key) for key in ("query_id", "parent_id", "split", "horizon", "target_kind", "status")}
                                yield "endpoint_controls", dict(identity, method=method, **_flat(metrics), **citation,
                                    metric_record=location+"/"+str(index)+"/controls/"+method)
                    yield from _tables(row, base, location+"/"+str(index), depth=depth+1)
            elif isinstance(value, dict):yield from _tables(value, base, location, depth=depth+1)


def _checksum(path, root, expected):
    """Verify sealed bytes without deserializing binary scientific artifacts."""
    file = shared._inside(root, path)
    if not isinstance(expected, str) or not shared._HASH.fullmatch(expected):
        raise ValueError("Tangent artifact identity must be an actual SHA256")
    with file.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError("Sealed artifact must be a regular file")
        checksum = hashlib.sha256()
        for block in iter(lambda:stream.read(1 << 20), b""):checksum.update(block)
        after = os.fstat(stream.fileno())
    signature = lambda s:(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after) or signature(after) != signature(file.stat(follow_symlinks=False)) or checksum.hexdigest() != expected:
        raise ValueError("Tangent sealed artifact changed: "+file.name)
    return file


def _external_protocol(ctx, phase, stage, path, binding):
    """Follow the prospective ancestor binding, never the child's digest."""
    if str(Path(binding["stage_paths"][stage]).absolute()) != str(path.absolute()):
        raise ValueError("Inherited stage path differs from the frozen prerequisite binding")
    workflow_path = Path(binding["workflow_path"])
    workflow, _, _ = shared.read_metadata(workflow_path, expected=binding["workflow_sha256"])
    root = Path(workflow["root"])
    shared._inside(root, workflow_path);shared._inside(root, path)
    for name in ("run_id", "phase", "profile", "protocol_sha256", "source_sha256", "source_tree_sha256", "slurm_profile_sha256"):
        if workflow.get(name) != binding.get(name):raise ValueError("Inherited workflow binding changed: "+name)
    if getattr(ctx, "sourcehash", None) and workflow["source_tree_sha256"] != ctx.sourcehash:
        raise ValueError("Inherited workflow changed executable source identity")
    protocol, _, _ = shared.read_metadata(path/"protocol.json", root=path)
    if (digest(protocol) != binding["protocol_sha256"] or protocol != workflow["scientific_protocol"]
            or stage not in protocol["units"] or stage not in protocol["stages"]):
        raise ValueError("Inherited stage differs from its own frozen scientific protocol")
    seal, seal_hash, _ = shared.read_metadata(path/"workflow-seal.json", root=path, expected=binding["seals"][stage])
    if (seal.get("schema") != "tdn.tangent-response-workflow-seal/v1" or seal.get("stage") != stage
            or seal.get("status") != "COMPLETED" or seal.get("protocol_sha256") != binding["protocol_sha256"]
            or seal.get("source_tree_sha256") != binding["source_tree_sha256"]
            or seal.get("slurm_profile_sha256") != binding["slurm_profile_sha256"]):
        raise ValueError("Inherited native terminal workflow seal changed")
    _checksum(path/"science_manifest.json", path, seal["science_manifest_sha256"])
    software_path = Path(workflow["run_dir"])/"state/submission-software.json"
    _checksum(software_path, root, binding["software_sha256"])
    worker_path = shared._inside(root, root/shared.relative_path(seal["worker_state_path"]))
    worker, _, _ = shared.read_metadata(worker_path, root=root, expected=seal["artifacts"].get(seal["worker_state_path"]))
    if (worker.get("status") != "COMPLETED" or worker.get("source_sha256") != binding["source_sha256"]
            or worker.get("protocol_sha256") != binding["protocol_sha256"] or worker.get("slurm_job_id") != seal.get("slurm_job_id")):
        raise ValueError("Inherited terminal worker state changed")
    for relative, checksum in seal["artifacts"].items():_checksum(root/shared.relative_path(relative), root, checksum)
    return protocol, dict(inherited_phase=phase, workflow_sha256=binding["workflow_sha256"],
        workflow_seal_sha256=seal_hash, native_job_id=seal.get("slurm_job_id"))


def _declared_sources(ctx, prerequisites):
    for stage in ctx.protocol["stages"]:
        yield stage, stage, ctx.protocol["units"][stage], ctx.protocol, prerequisites.get(stage), {}, None
    for phase, binding in ctx.protocol.get("prerequisites", {}).items():
        shared.relative_path(phase)
        for stage, declared in binding.get("stage_paths", {}).items():
            shared.relative_path(stage);key = phase+"/"+stage
            path = prerequisites.get(key)
            try:
                if path is None:raise ValueError("Inherited requested stage was not supplied to the reporting context")
                protocol, provenance = _external_protocol(ctx, phase, stage, Path(path), binding)
                yield key, stage, protocol["units"][stage], protocol, path, provenance, None
            except (OSError, ValueError, KeyError, TypeError) as error:
                yield key, stage, {"kind":"inherited-evidence"}, None, path, {"inherited_phase":phase}, str(error)


def _workflow_lineage(ctx):
    """Only explicitly captured workflows and recursively hash-bound origins."""
    queue = []
    active = os.environ.get("TDN_TANGENT_WORKFLOW")
    if active:queue.append((active, None, None))
    for binding in ctx.protocol.get("prerequisites", {}).values():
        queue.append((binding["workflow_path"], binding["workflow_sha256"], binding))
    seen, workflows, errors = {}, [], []
    while queue:
        if len(seen) >= 64:raise ValueError("Tangent workflow lineage exceeds its 64-workflow bound")
        name, expected, parent = queue.pop(0);path = Path(name).absolute()
        if path in seen:
            if expected is not None and seen[path] != expected:errors.append(dict(path_token=path.name,error="Conflicting workflow lineage hash"))
            continue
        try:
            if path.stat().st_size > 16 << 20:raise ValueError("Native workflow exceeds bounded research metadata size")
            workflow, checksum, raw = shared.read_metadata(path, expected=expected)
            root, run = Path(workflow["root"]), Path(workflow["run_dir"])
            identity = shared.stable_id(workflow["run_id"])
            if (workflow.get("schema_version") != 1 or workflow.get("kind") != "desktop-slurm-tangent-response"
                    or workflow.get("execution_mode") != "desktop-slurm" or identity != workflow["run_id"]
                    or run != root/"runs"/identity or path != run/"tangent-response-workflow.json"):
                raise ValueError("Exact native workflow root/program identity differs")
            shared._inside(root, path);shared._inside(root, ctx.path)
            if any(not isinstance(workflow.get(key),str) or not shared._HASH.fullmatch(workflow[key])
                   for key in ("protocol_sha256","source_sha256","source_tree_sha256","slurm_profile_sha256")):
                raise ValueError("Native workflow must retain exact SHA256 identities")
            protocol = workflow["scientific_protocol"]
            if (digest(protocol) != workflow["protocol_sha256"] or protocol.get("profile") != workflow["profile"]
                    or set(protocol["stages"]) != set(protocol["units"]) or len(set(protocol["stages"])) != len(protocol["stages"])
                    or workflow["source_tree_sha256"] != getattr(ctx,"sourcehash",None)):
                raise ValueError("Lineage scientific roster or executable identity differs")
            frozen, _, _ = shared.read_metadata(run/"protocol.json",root=root)
            profile, _, _ = shared.read_metadata(run/"slurm-profile.json",root=root)
            software, software_hash, _ = shared.read_metadata(run/"state/submission-software.json",root=root)
            if frozen != protocol or digest(profile) != workflow["slurm_profile_sha256"] or profile != workflow["slurm_profile"]:
                raise ValueError("Lineage frozen protocol/native profile changed")
            if parent is not None:
                for key in ("profile","protocol_sha256","source_sha256","source_tree_sha256","slurm_profile_sha256"):
                    if workflow[key] != parent[key]:raise ValueError("Recovery origin changed "+key)
                if parent.get("software_sha256",software_hash) != software_hash:raise ValueError("Recovery software identity differs")
            elif path == Path(active or "").absolute() and protocol != ctx.protocol:
                raise ValueError("Active reporting workflow differs from the current science protocol")
            workflow = dict(workflow, software_sha256=software_hash, workflow_sha256=checksum)
            workflows.append(workflow);seen[path]=checksum
            recovery = workflow.get("recovery", {})
            if recovery:
                if recovery.get("origin_run") == identity:raise ValueError("Recovery lineage contains a cycle")
                origin = recovery["origin_workflow"]
                if Path(origin) != root/"runs"/shared.stable_id(recovery["origin_run"])/"tangent-response-workflow.json":
                    raise ValueError("Recovery origin path differs from its exact run identity")
                queue.append((origin,recovery["origin_workflow_sha256"],workflow))
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(path_token=path.name,error=str(error)))
    return workflows, errors


def _attempt_metadata(ctx, worker, stage, tables, copies, seen):
    if worker in seen or not (worker/"run.json").is_file():return
    seen.add(worker)
    for name, group in (("summary.json","worker_costs"),("telemetry/index.json","worker_telemetry")):
        file = worker/name
        if not file.exists():continue
        payload, checksum, raw = shared.read_metadata(file, root=worker)
        pointer = "sources/"+checksum+".json";copies[pointer]=raw
        base = dict(stage=stage,attempt=worker.name,source_path=pointer,source_sha256=checksum,
            scope="Exact operational worker observation; not added to body/trial-cumulative or allocation costs")
        if group == "worker_costs":tables[group].append(dict(_flat(payload),**base))
        else:
            for source in payload.get("sources",[]):
                stream=worker/shared.relative_path(source["path"]);stream_pointer=None
                if stream.is_file():
                    samples,checksum,raw=shared.read_metadata(stream,root=worker)
                    stream_pointer="sources/"+checksum+".jsonl";copies[stream_pointer]=raw
                    indices=sorted({round(i*(len(samples)-1)/255) for i in range(256)}) if len(samples)>256 else range(len(samples))
                    for index in indices:
                        tables["worker_resource_samples"].append(dict(base,**_flat(samples[index]),source_path=stream_pointer,
                            source_sha256=checksum,source_record="/lines/"+str(index+1),
                            public_sampling="All samples" if len(samples)<=256 else "At most 256 uniform observations; full retained JSONL is copied"))
                tables[group].append(dict(_flat(source),**base,stream_path=stream_pointer))


def _incomplete_attempts(ctx,project):
    """Preserve failed/unfinalized evidence without presenting a broken run."""
    records=[];copies={}
    for stage in ctx.protocol["stages"]:
        if stage==ctx.stage:continue
        probe=SimpleNamespace(path=Path(ctx.path).parent/stage,stage=stage,protocol=ctx.protocol)
        worker=shared.attempt_path(probe);inventory=worker/"run.json"
        if not inventory.is_file():continue
        run,_,_=shared.read_metadata(inventory,root=project)
        metadata=run.get("metadata",{})
        if (run.get("schema")!="raven.run/v1" or run.get("project_id")!=CONFIG["project_id"]
                or run.get("run_id")!=worker.name or metadata.get("stage")!=stage
                or metadata.get("protocol_sha256")!=digest(ctx.protocol)
                or getattr(ctx,"sourcehash",None) and metadata.get("source_tree_sha256")!=ctx.sourcehash):
            raise ValueError("Incomplete worker inventory differs from its declared source/stage")
        missing=[]
        for key,value in run.get("paths",{}).items():
            target=shared._inside(worker,worker/shared.relative_path(value))
            exists=target.is_dir() and any(target.iterdir()) if key in ("outputs","logs","passports") else target.is_file()
            if not exists:missing.append(value)
        log_index=run.get("paths",{}).get("log_index")
        if log_index and (worker/log_index).is_file():
            logs,_,_=shared.read_metadata(worker/log_index,root=worker)
            if logs.get("schema")!="raven.logs/v1" or logs.get("run_id")!=worker.name or len(logs.get("logs",[]))>256:
                raise ValueError("Incomplete worker log index differs from its exact identity")
            for log in logs.get("logs",[]):
                target=shared._inside(worker,worker/shared.relative_path(log["path"]))
                if not target.is_file():missing.append(log["path"])
        unfinished=run.get("state") in ("RUNNING","PENDING","CREATED","SUBMITTED")
        if not missing and not unfinished:continue
        prefix="incomplete-evidence/"+worker.name;files=[]
        owned=list(worker.rglob("*"))
        if len(owned)>4096:raise ValueError("Incomplete native attempt exceeds its bounded evidence inventory")
        for file in owned:
            shared._inside(worker,file)
            if not file.is_file():continue
            relative=file.relative_to(worker).as_posix()
            renamed=relative.rsplit("/",1)
            basename={"run.json":"inventory.raw.json","logs.json":"log-index.raw.json"}.get(renamed[-1],renamed[-1])
            relative=(renamed[0]+"/" if len(renamed)==2 else "")+basename
            pointer=prefix+"/"+relative
            if file.stat().st_size>shared.MAX_METADATA_BYTES:
                from .report_artifacts import ScientificPayload
                checksum=hashlib.sha256()
                with file.open("rb") as stream:
                    for block in iter(lambda:stream.read(1<<20),b""):checksum.update(block)
                raw=ScientificPayload(file,worker,checksum.hexdigest())
            else:raw=file.read_bytes()
            checksum=raw.sha256 if hasattr(raw,"sha256") else hashlib.sha256(raw).hexdigest()
            copies[pointer]=raw;files.append(dict(original_name=file.relative_to(worker).as_posix(),path=pointer,sha256=checksum,bytes=len(raw)))
        worker_state=Path(ctx.path).parent/"state"/(stage+".json");actual_state="UNAVAILABLE"
        if worker_state.is_file():
            state,checksum,raw=shared.read_metadata(worker_state,root=Path(ctx.path).parent)
            if ("slurm_job_id" in state and state["slurm_job_id"]!=run.get("job_id")
                    or "protocol_sha256" in state and state["protocol_sha256"]!=digest(ctx.protocol)):
                raise ValueError("Incomplete worker state differs from its declared allocation/protocol")
            actual_state=state.get("status","UNKNOWN");pointer=prefix+"/native-worker-state.raw.json";copies[pointer]=raw
            files.append(dict(original_name="native-worker-state",path=pointer,sha256=checksum,bytes=len(raw)))
        records.append(dict(stage=stage,attempt=worker.name,job_id=run.get("job_id"),original_raven_state=run.get("state"),
            actual_native_worker_state=actual_state,missing_declared_paths=missing,files=files,
            excluded_inventory_path=worker.relative_to(project).as_posix(),
            scope="Original failed/unfinished metadata retained unchanged as evidence; not a completed portable attempt"))
    return records,copies


def _lineage_attempts(ctx, tables, copies, seen):
    workflows, errors = _workflow_lineage(ctx);projects=[];total_jobs=0
    for workflow in workflows:
        root,run=Path(workflow["root"]),Path(workflow["run_dir"]);protocol=workflow["scientific_protocol"]
        try:
            jobs,_,_=shared.read_metadata(run/"jobs.json",root=root)
            if (not isinstance(jobs,list) or len(jobs)>20000 or len({r["stage"] for r in jobs})!=len(jobs)
                    or len({r["job_id"] for r in jobs})!=len(jobs)):raise ValueError("Native allocation receipt has ambiguous jobs")
            total_jobs+=len(jobs)
            if total_jobs>20000:raise ValueError("Research lineage exceeds its 20000-job bound")
            prefix="lineage/"+workflow["run_id"];imported=[]
            for job in jobs:
                stage=job["stage"];job_id=job["job_id"]
                if stage not in protocol["units"] or not isinstance(job_id,str) or not re.fullmatch(r"[1-9][0-9]{0,63}",job_id):
                    raise ValueError("Lineage allocation receipt differs from its declared stage/job")
                if run == Path(ctx.path).parent and stage == ctx.stage:continue
                statefile=run/"state"/(stage+".json");state=None
                if statefile.is_file():
                    state,checksum,raw=shared.read_metadata(statefile,root=root)
                    if (state.get("source_sha256")!=workflow["source_sha256"] or state.get("protocol_sha256")!=workflow["protocol_sha256"]
                            or state.get("slurm_job_id")!=job_id):raise ValueError("Native terminal worker differs from its exact submission receipt")
                    pointer="sources/"+checksum+".json";copies[pointer]=raw
                    tables["native_attempts"].append(dict(run_id=workflow["run_id"],stage=stage,job_id=job_id,
                        worker_status=state.get("status"),source_path=pointer,source_sha256=checksum,
                        workflow_sha256=workflow["workflow_sha256"],software_sha256=workflow["software_sha256"],
                        scope="Registered actual allocation attempt; missing runtime remains unavailable"))
                else:
                    tables["native_attempts"].append(dict(run_id=workflow["run_id"],stage=stage,job_id=job_id,
                        worker_status="MISSING",runtime_seconds=None,reason="No retained worker metadata; no counter inferred"))
                probe=SimpleNamespace(path=run/stage,stage=stage,protocol=protocol)
                worker=shared.attempt_path(probe)
                if not (worker/"run.json").is_file():continue
                inventory,_,_=shared.read_metadata(worker/"run.json",root=root)
                if (inventory.get("schema")!="raven.run/v1" or inventory.get("run_id")!=worker.name or inventory.get("job_id")!=job_id
                        or inventory.get("project_id")!=CONFIG["project_id"] or inventory.get("metadata",{}).get("stage")!=stage
                        or inventory["metadata"].get("source_tree_sha256")!=workflow["source_tree_sha256"]
                        or inventory["metadata"].get("protocol_sha256")!=workflow["protocol_sha256"]):
                    raise ValueError("Actual Raven attempt differs from the frozen native job/source")
                if inventory.get("state") not in {"COMPLETED","FAILED","INTERRUPTED","TIMEOUT","CANCELLED","OUT_OF_MEMORY"}:
                    tables["native_attempts"][-1]["portable_status"]="UNFINALIZED_EXCLUDED";continue
                terminal,_,_=shared.read_metadata(worker/"summary.json",root=root)
                if terminal.get("id")!=worker.name or terminal.get("job_id")!=job_id or terminal.get("state")!=inventory["state"]:
                    raise ValueError("Actual Raven terminal summary changed attempt identity")
                tables["native_attempts"][-1].update(raven_terminal_state=inventory["state"],
                    lifecycle_consistency="MATCHED" if state and state.get("status")==inventory["state"] else "MISSING_OR_MISMATCHED_WORKER_STATE",
                    runtime_seconds=terminal.get("runtime_seconds"))
                _attempt_metadata(ctx,worker,workflow["run_id"]+"/"+stage,tables,copies,seen)
                if worker.is_relative_to(shared.raven_root(ctx)):continue
                relative=worker.relative_to(run/"raven").as_posix();target=prefix+"/"+relative
                files=list(worker.rglob("*"))
                if len(files)>2048:raise ValueError("Single native attempt exceeds bounded portable file inventory")
                for file in files:
                    shared._inside(worker,file)
                    if not file.is_file():continue
                    if file.suffix not in {".json",".jsonl",".csv",".log",".txt",".out",".err",".md",".html",".png",".svg"}:
                        raise ValueError("Native research attempt contains an undeclared binary export")
                    if file.stat().st_size>shared.MAX_METADATA_BYTES:raise ValueError("Native attempt metadata exceeds bounded export size")
                    copies[target+"/"+file.relative_to(worker).as_posix()]=file.read_bytes()
                project=target.rsplit("/runs/",1)[0]
                source_project=worker.parent.parent
                for filename in (".raven/config.json",".raven/contracts/outputs.v1.json","README.md"):
                    file=shared._inside(source_project,source_project/filename)
                    if file.is_file():copies[project+"/"+filename]=file.read_bytes()
                imported.append(dict(stage=stage,job_id=job_id,attempt=worker.name,project=project,inventory_path=target+"/run.json"))
            if imported:projects.append(dict(run_id=workflow["run_id"],attempts=imported,
                path_base="Research overview outputs; every imported page is an independent standard Raven project"))
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(run_id=workflow["run_id"],error=str(error)))
    tables["lineage_errors"]=errors
    measured=[row["runtime_seconds"] for row in tables["native_attempts"] if shared.finite(row.get("runtime_seconds"))]
    return projects, dict(workflows=len(workflows),registered_native_attempts=len(tables["native_attempts"]),
        measured_native_worker_runtimes=len(measured),missing_native_worker_runtimes=len(tables["native_attempts"])-len(measured),
        observed_native_worker_seconds=sum(measured),
        runtime_scope="Sum of actual retained elapsed worker attempts, including failed/recovered work; not campaign calendar time, pending time, or sacct allocation time",
        lineage_errors=len(errors),cost_status="INCOMPLETE" if errors else "RETAINED_WORKER_OBSERVATIONS_ONLY",
        allocation_accounting="UNAVAILABLE_UNLESS_SEPARATELY_RETAINED; NO ENERGY OR MONETARY COST INFERRED")


def collect(ctx):
    prerequisites = getattr(ctx, "prereqs", getattr(ctx, "prerequisites", {}))
    tables = collections.defaultdict(list)
    sources, omissions, units, decisions = [], [], [], []
    raw_copies = {}
    source_hashes = set()
    declared = list(_declared_sources(ctx, prerequisites))
    seen_workers=set()
    for stage, science_stage, unit, protocol, base_path, provenance, error in declared:
        row = dict(stage=stage, kind=unit["kind"], device=unit.get("device"), status="NOT_RUN_OR_MISSING",
                   reason="No sealed requested unit; negative/unresolved gates do not imply execution", **provenance)
        if stage == ctx.stage:
            row.update(status="CURRENT_REPORT_OBSERVER", reason="Observer work and its still-running allocation are excluded")
        elif error:
            row.update(status="UNRESOLVED_SOURCE", reason=error)
        elif base_path is not None:
            path = Path(base_path)
            before = dict(sources=len(sources), omissions=len(omissions), decisions=len(decisions),
                          tables={key:len(value) for key,value in tables.items()}, copies=set(raw_copies))
            try:
                manifest, manifest_hash, _ = shared.read_metadata(path/"science_manifest.json", root=path)
                marker = shared._inside(path, path/"COMPLETED")
                if (manifest.get("schema") != "tdn.tangent-response-science/v1" or manifest.get("stage") != science_stage
                        or manifest.get("profile") != protocol["profile"]
                        or manifest.get("protocol_sha256") != digest(protocol)
                        or not stat.S_ISREG(marker.stat().st_mode) or marker.stat().st_size > 128 or marker.read_text().strip() != manifest_hash):
                    raise ValueError("Tangent research source does not match its exact science seal")
                if (not isinstance(manifest.get("artifacts"), dict) or "summary.json" not in manifest["artifacts"]
                        or not isinstance(manifest.get("source_tree_sha256"), str)
                        or not shared._HASH.fullmatch(manifest["source_tree_sha256"])):
                    raise ValueError("Tangent science seal lacks its complete source identity")
                if getattr(ctx, "sourcehash", None) and manifest.get("source_tree_sha256") != ctx.sourcehash:
                    raise ValueError("Tangent scientific sources changed executable identity")
                source_hashes.add(manifest["source_tree_sha256"])
                summary, _, _ = shared.read_metadata(path/"summary.json", root=path,
                    expected=manifest["artifacts"].get("summary.json"))
                row.update(status=summary.get("status", "UNKNOWN"), elapsed_seconds=summary.get("elapsed_seconds"),
                           reason=None, science_manifest_sha256=manifest_hash)
                for name, expected in manifest["artifacts"].items():
                    shared.relative_path(name)
                    if not isinstance(expected, str) or not shared._HASH.fullmatch(expected):
                        raise ValueError("Tangent artifact identity must be an actual SHA256")
                    file = shared._inside(path, path/name)
                    metadata = Path(name).name == name and Path(name).suffix in (".json", ".jsonl")
                    if not metadata or name in ("protocol.json", "execution.json"):
                        omissions.append(dict(stage=stage, path=name, sha256=expected, bytes=file.stat().st_size,
                            reason="Retained in native science run; compact review omits binary replay and repeated infrastructure metadata"))
                        continue
                    from .report_artifacts import PAGE_BYTES,ScientificPayload
                    if LARGE_SCIENCE_KINDS.get(name)==unit["kind"] and file.stat().st_size>PAGE_BYTES:
                        raw=ScientificPayload(file,path,expected);payload=raw.json();checksum=raw.sha256
                    else:payload, checksum, raw = shared.read_metadata(file, root=path, expected=expected)
                    copy = "sources/"+checksum+file.suffix
                    raw_copies[copy] = raw
                    source = dict(stage=stage, name=name, path=copy, sha256=checksum, bytes=len(raw),
                        science_manifest_sha256=manifest_hash, source_tree_sha256=manifest["source_tree_sha256"], **provenance)
                    sources.append(source)
                    base = dict(stage=stage, source_path=copy, source_sha256=checksum, source_file=name, **provenance)
                    if name == "decision.json":decisions.append(dict(payload, **base))
                    if isinstance(payload, list):
                        for index, value in enumerate(payload):
                            tables["learning_curves"].append(dict(_flat(value), **base, source_record="/lines/"+str(index+1)))
                    else:
                        tables["stage_metadata"].append(dict(_flat(payload), **base, source_record="/"))
                        for table, values in _tables(payload, base):tables[table].append(values)
            except (OSError, ValueError, KeyError, TypeError) as error:
                # A later corrupt metadata artifact invalidates the source as
                # a whole; never leave an earlier PASS visible as valid data.
                del sources[before["sources"]:];del omissions[before["omissions"]:];del decisions[before["decisions"]:]
                for key in tables:del tables[key][before["tables"].get(key, 0):]
                for key in set(raw_copies)-before["copies"]:del raw_copies[key]
                row.update(status="UNRESOLVED_SOURCE", reason=str(error))
        units.append(row)
    if len(source_hashes) > 1:raise ValueError("Tangent research mixes executable source identities")
    for stage, science_stage, unit, protocol, base_path, provenance, error in declared:
        if stage == ctx.stage:continue
        if error or protocol is None:continue
        probe = SimpleNamespace(path=Path(base_path) if base_path else Path(ctx.path).parent/stage,
                                stage=science_stage, protocol=protocol)
        worker = shared.attempt_path(probe)
        _attempt_metadata(ctx,worker,stage,tables,raw_copies,seen_workers)
    lineage_projects,lineage_costs=_lineage_attempts(ctx,tables,raw_copies,seen_workers)
    tables["units"] = units
    methods = set(METHODS)
    for table in tables.values():
        for row in table:
            for key in ("method", "arm", "control"):
                if isinstance(row.get(key), str):methods.add(row[key])
    tables["method_coverage"] = [dict(method=method,
        recorded_rows=sum(row.get("method", row.get("arm", row.get("control"))) == method for table in tables.values() for row in table),
        status="RECORDED" if any(row.get("method", row.get("arm", row.get("control"))) == method for table in tables.values() for row in table) else "NOT_RUN_OR_MISSING",
        scope="Requested/observed methods; raw rows retain paired query denominators, not independent replicate counts") for method in sorted(methods)]
    incomplete,extra_copies=_incomplete_attempts(ctx,shared.raven_root(ctx));raw_copies.update(extra_copies)
    return dict(schema=SCHEMA, profile=ctx.protocol["profile"], protocol_sha256=digest(ctx.protocol),
        source_tree_sha256=next(iter(source_hashes), getattr(ctx, "sourcehash", None)),
        independent_unit="Parent field; phases/horizons/grids/seeds/probes are paired descendants",
        native_gpu_verification="NOT_INFERRED_FROM_REPORTING", decisions=decisions, sources=sources,
        lineage_projects=lineage_projects,lineage_costs=lineage_costs,
        incomplete_attempts=incomplete,
        unsupported_targets=ctx.protocol.get("unsupported_targets", {}),
        omissions=omissions, tables=dict(tables),
        summary=dict(registered_units=len(units), completed_science_units=sum(r["status"] == "COMPLETED" for r in units),
            missing_units=sum(r["status"] == "NOT_RUN_OR_MISSING" for r in units), unresolved_source_units=sum(r["status"] == "UNRESOLVED_SOURCE" for r in units),
            declared_methods=len(methods), recorded_source_files=len(sources), incomplete_portable_attempts=len(incomplete))), raw_copies


def _plots(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    groups = (
        ("physical-rms", "Physical RMS (source-declared field units)", lambda k:(k == "rms" or k.endswith((".rms", "_rms"))) and not any(x in k for x in ("relative", "ratio", "gradient", "parameter", "movement"))),
        ("physical-maximum", "Sampled maximum absolute error (field units)", lambda k:k == "maximum" or k.endswith((".maximum", "_maximum"))),
        ("normalized-objective", "Producer-defined normalized objective (not cross-arm comparable)", lambda k:k in ("train_loss", "mean_loss", "centered_loss", "validation_loss")),
        ("gradient", "Gradient norm (producer parameter/optimizer scope)", lambda k:k.endswith("gradient_norm")),
        ("relative-error", "Dimensionless producer-defined ratio/error", lambda k:("relative" in k and "error" in k) or k.endswith("ratio")),
    )
    for name, rows in tables.items():
        if name in ("stage_metadata", "worker_resource_samples"):continue
        for group, label, selector in groups:
            candidates = {key for row in rows for key in row if selector(key)}
            keys = sorted(key for key in candidates if any(shared.finite(r.get(key)) for r in rows))
            if not keys:continue
            figure, axis = plt.subplots(figsize=(8, 4))
            for key in keys[:12]:
                points = [(row.get("update", index), row[key]) for index, row in enumerate(rows) if shared.finite(row.get(key))]
                if points:axis.scatter(*zip(*points), s=7, label=key)
            axis.set(xlabel="Committed update" if name == "learning_curves" else "Recorded observation ordinal (paired descendants)", ylabel=label,
                title="Observed "+name.replace("_", " ")+" — "+group.replace("-", " "))
            axis.legend(fontsize=7);axis.grid(alpha=.2)
            filename = "observed-"+shared.stable_id(name+"-"+group)+".png"
            figure.tight_layout();figure.savefig(output/filename, dpi=140);figures.append(filename);plt.close(figure)
    resources = tables.get("worker_resource_samples", [])
    resource_metrics = (("cpu_utilization_percent", "Process-tree CPU (%; 100% per logical core)"),
        ("rss_bytes", "Process-tree RSS (bytes; shared pages can repeat)"), ("utilization_percent", "Assigned-device GPU utilization (%)"),
        ("memory_used_mib", "Assigned-device VRAM used (MiB)"), ("power_watts", "Assigned-device power (W; sampled, not energy)"),
        ("temperature_celsius", "Assigned-device GPU temperature (°C)"))
    for key, label in resource_metrics:
        series = collections.defaultdict(list)
        for row in resources:
            if shared.finite(row.get(key)):
                identity = row["stage"]+(" / "+str(row["device_uuid"]) if row.get("device_uuid") else "")
                series[identity].append((row["elapsed_seconds"], row[key]))
        if not series:continue
        figure, axis = plt.subplots(figsize=(8, 4))
        for identity, points in sorted(series.items()):
            points.sort();axis.plot(*zip(*points), linewidth=.8, marker=".", markersize=3, alpha=.7, label=identity)
        axis.set(xlabel="Elapsed worker seconds (sampled observations)", ylabel=label, title="Worker resource observations")
        if len(series) <= 12:axis.legend(fontsize=7)
        axis.grid(alpha=.2);figure.tight_layout()
        filename = "resource-"+key+".png";figure.savefig(output/filename, dpi=140);figures.append(filename);plt.close(figure)
    return figures


def run(ctx):
    from . import report_artifacts
    ctx._raven_config = CONFIG
    data, copies = collect(ctx)
    project = shared.raven_root(ctx)
    shared._project_files(project, CONFIG);shared._project_catalog(ctx)
    attempt = project/"runs"/shared.stable_id(Path(ctx.path).parent.name+"-analysis")
    if attempt.exists():raise FileExistsError("Preserve completed tangent research overviews")
    output = attempt/"outputs";output.mkdir(parents=True)
    (attempt/"logs").mkdir()
    index,data=report_artifacts.publish(output,data,copies,CONFIG,Path(ctx.path).parent.name)
    write_json(output/"decisions.json", dict(schema="tdn.tangent-response-decision-catalog/v1", decisions=index["decisions"],
        complete_decisions=index["sections"]["decisions"], native_gpu_verification=data["native_gpu_verification"],
        scope="This is a catalog; complete serialized source decisions are unchanged in the linked pages"))
    write_json(output/"lineage_projects.json", dict(schema="tdn.tangent-response-lineage-projects/v1",
        projects=index["sections"]["lineage_projects"],costs=data["lineage_costs"],
        path_base="This overview outputs directory; imported exact attempts preserve original run/job identities"))
    write_json(output/"omission_inventory.json", dict(schema="tdn.tangent-response-omission-catalog/v1", files=index["sections"]["omissions"],
        replay_scope="Compact research review; exact binary replay/checkpoints/journals remain in source native runs"))
    write_json(output/"metric_definitions.json", dict(definitions=DEFINITIONS, independent_unit=data["independent_unit"],
        uncertainty="Empirical refinement estimates; no confidence intervals or mathematical certification inferred",
        missing="Missing/unrun/failed/precision-null are retained, never converted to successful zeros",
        sampling="Declared public learning observations; private journals retain all optimizer commits"))
    (output/"metric_legend.md").write_text("# Tangent research metric legend\n\n"+"\n\n".join("**"+k+"**: "+v for k,v in DEFINITIONS.items())+"\n")
    figures = _plots(output, data["tables"])
    write_json(output/"artifact_projects.json",dict(schema="tdn.tangent-response-artifact-projects/v1",
        projects=index["artifact_projects"],path_base="This overview outputs directory; register each named page with :project PATH",
        scope="Every table row/raw source is retained; each standard page contract fits Raven's default 8 MiB and 256-entry budgets"))
    links = ["<li><a href='"+html.escape(file.name)+"'>"+html.escape(file.name)+"</a></li>" for file in sorted(output.iterdir()) if file.is_file()]
    links += ["<li><a href='"+html.escape(chunk["csv"]["path"])+"'>"+html.escape(name)+" CSV rows "+str(chunk["start_row"])+"–"+str(chunk["start_row"]+chunk["rows"])+"</a></li>"
        for name,table in index["tables"].items() for chunk in table["chunks"]]
    (output/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>Tangent research</title><h1>Tangent-response research</h1>"
        +"<p>Complete negative, missing and unresolved outcomes are retained. CPU implementation verification is separate from native readiness and scientific benefit.</p>"
        +"<pre>"+html.escape(json.dumps(data["summary"], indent=2))+"</pre><p>"+html.escape(data["independent_unit"])+"</p><ul>"
        +"".join(links)+"</ul>"+"".join("<img style='max-width:100%' src='"+name+"'>" for name in figures))
    files = [f for f in output.iterdir() if f.is_file() and f.suffix in (".json", ".csv", ".html", ".md")]
    contract = dict(version=1, outputs=[dict(path="outputs/"+f.name, required=True, min_bytes=1,
        sha256=hashlib.sha256(f.read_bytes()).hexdigest(), format="csv" if f.suffix == ".csv" else "json" if f.suffix == ".json" else "text") for f in files])
    write_json(output/"contract.json", contract)
    metadata = dict(source_tree_sha256=data["source_tree_sha256"], protocol_sha256=data["protocol_sha256"],
        native_gpu_verification="NOT_INFERRED_FROM_REPORTING", runtime_scope="Aggregate research overview; no independent scheduler allocation",
        metric_definitions="outputs/metric_definitions.json", source_citations="outputs/tangent_research.json",
        replay_scope="Metadata/plots/resource streams only; see omission inventory")
    write_json(attempt/"run.json", dict(schema="raven.run/v1", project_id=CONFIG["project_id"], run_id=attempt.name,
        experiment_id="tangent-response-analysis", attempt=1, name="TDN/tangent-response/research", state="COMPLETED",
        paths=dict(metrics="metrics.jsonl", summary="summary.json", log_index="logs.json", outputs="outputs", logs="logs", contract="outputs/contract.json"), metadata=metadata))
    write_json(attempt/"summary.json", dict(schema="raven.summary/v1", id=attempt.name, name="TDN/tangent-response/research",
        state="COMPLETED", project_id=CONFIG["project_id"], experiment_id="tangent-response-analysis", attempt=1,
        exit_code=0, results=data["summary"], metadata=metadata))
    write_json(attempt/"logs.json", dict(schema="raven.logs/v1", run_id=attempt.name,
        logs=[dict(id="research", path="logs/research.log", label="Research overview", group="Research")]))
    (attempt/"logs/research.log").write_text(json.dumps(data["summary"])+"\n")
    (attempt/"metrics.jsonl").write_text(json.dumps(shared.metric_record(data["summary"], phase="research_overview", t=time.time()))+"\n")
    current = shared.attempt_path(ctx).relative_to(project).as_posix()
    exclusions=[current]+[entry["excluded_inventory_path"] for entry in data["incomplete_attempts"]]
    receipt = portable.portable_export(project, Path(ctx.path)/BUNDLE, exclude_attempts=exclusions)
    write_json(Path(ctx.path)/"research_summary.json", dict(data["summary"], portable=receipt,
        decisions=data["decisions"], exclusions=dict(reporter="Current reporter worker has not yet finalized",
            incomplete_attempts=index["sections"]["incomplete_attempts"],
            scope="Unfinished or structurally incomplete live inventories are excluded; unchanged original metadata, logs and actual failure states are retained as explicit evidence")))
    return dict(status="COMPLETED_RESEARCH_REPORT", scientific_decisions=data["decisions"], research=data["summary"],
                portable_research_bundle=BUNDLE, native_gpu_verification="NOT_INFERRED_FROM_REPORTING")
