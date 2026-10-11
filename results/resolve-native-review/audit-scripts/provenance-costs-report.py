from pathlib import Path
import json
base=Path(__file__).resolve().parent
p=json.loads((base/'provenance-costs.json').read_text());t=p['totals'];pr=p['provenance']
def clock(seconds):
 seconds=round(seconds);return f'{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}'
rows=[]
for kind,r in p['by_kind'].items():
 rows.append(f"| {kind} | {r['jobs']} | {clock(r['allocation_elapsed_seconds'])} | {r['scientific_elapsed_seconds']:.2f} | {r['total_cpu_seconds']/r['allocation_elapsed_seconds']:.2f} | {100*r['cpu_allocation_utilization_proxy']:.1f}% | {r['max_reported_step_rss_bytes']/2**30:.3f} |")
text=f'''# Native Resolve provenance, operational cost and reporting audit

Run: `{p['run_id']}`. All results here are a **development campaign**, not untouched confirmation. Analysis reads exact retained bytes from `review_io.py`; it does not deserialize a checkpoint or re-run the numerical experiments.

## Provenance and transport

**No provenance, protocol, dependency, allocation, software, required-test, or canonical-table inconsistency was found** in the checks implemented in `provenance-costs.py`. One distinct figure-discovery defect is described below; it does not alter the numerical results.

- Independently rechecked all four uploaded parts, their concatenated **86,547,084-byte** archive, and the separately uploaded optional manifest. Its measured SHA-256 `{p['transport']['optional_uploaded_manifest_sha256']}` matches the index's `manifest_sha256` exactly.
- Archive SHA-256: `{p['transport']['archive_sha256']}`. The prior complete transport verification covered **3,716 retained files**. This audit independently rehashed **{p['transport']['canonical_exact_files_rehashed']} canonical files** after decompression and matched their sizes/digests to that verified original export inventory.
- Native source commit: `{pr['native_commit']}`; native scientific source SHA-256: `{pr['native_source_tree_sha256']}`; controller fingerprint: `{pr['native_controller_source_sha256']}`. The current checkout matched all three **before the separate reporting-only fix**.
- **28/28** science completion markers, unit declarations, immediate and transitive science lineage, execution wrapper seals, exact Slurm IDs/steps, and frozen source/software/profile identities checked. Native software was clean Python **3.13.13**, project `.venv`, PyTorch **2.10.0+cu126**, NumPy **2.2.6**, SciPy **1.15.3**, CUDA runtime **12.6**.
- All **578 retained scientific-artifact references** match independently verified transport hashes. Another **626 artifacts (157,680,548 bytes)** have consistent declared hashes but are intentionally omitted: arrays, checkpoints and rendered figures. **Their actual bytes, loaded weights, predictions from a reloaded checkpoint and rendered appearance are not independently verified by this review.** This is not a complete replay bundle.
- The full profile explicitly declares `DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS`; all stage and Tower metadata retain that boundary.

The export reduced 801,391,004 original source bytes to 86,547,084 transport bytes by exact gzip compression plus declared replay/figure omissions, with 643,710,456 retained bytes and 4,594,853 bytes deduplicated. Native export and verification took **3.864 seconds**. This is useful transport efficiency, not a solver result.

## Native checks and scheduler behavior

The exact submitted allocations were jobs **847–874**: **28 completed, zero failed**, each reporting exit `0:0`. The final accounting snapshot at 07:07:55 UTC contains all 28 allocation rows and 549 additional step rows. Only allocation rows enter elapsed/CPU reservation totals; adding allocation and step rows would double count work.

The CPU readiness suite passed **252 cases**. Every one of the **10 GPU allocations** passed **all 20 distinct mandatory CUDA cases**, with no missing, duplicate, skipped, failed or errored case: **200 native GPU case executions**, not 200 distinct scientific tests. Preflights identify the actual RTX 4090, visible device `0`, real job IDs and numeric step `0`. The dedicated VRAM cap remains 18 GiB, the soft fraction 75% of available memory, and the hard sampled device-use fraction 90%.

All recorded allocations match the frozen resources: at most eight CPU cores and 48 GiB host RAM; CUDA stages use four CPUs and one GPU; maximum allocation is **35 minutes** and numerical budget **30 minutes**. The node exposes 110000 MiB host memory. These limits are ceilings, not measured consumption. Total requested time ceilings were **13 h 50 min** across all jobs, including **4 h 50 min** GPU allocation ceilings; the measured run finished far below those ceilings.

The submitted DAG matches every scientific dependency. Separate resource-only `afterany` edges serialize this desktop workflow without propagating failure into unrelated scientific branches; the report waits on `afterany` for every scientific allocation. The protected memory branch depends only on the audit. No pending-job cap, automatic expansion, recovery rerun or resubmission appears in this run.

There are **2,048 experiment-log rows and 11,058 check rows**, with **zero required mathematical/correctness failures** reported by the scientific stage summaries. These row counts include repeated methods, parameters and checks; they are not independent fields and not a count of successful scientific hypotheses.

## Complete measured campaign cost

Total allocation time: **{clock(t['allocation_elapsed_seconds'])}** (**{t['allocation_elapsed_seconds']/3600:.4f} allocation-hours**). GPU reservation time: **{clock(t['gpu_allocation_seconds'])}** (**{t['gpu_allocation_seconds']/3600:.4f} GPU-hours**). Allocated host CPU time: **{t['allocated_cpu_seconds']/3600:.4f} core-hours**; reported consumed CPU time: **{t['total_cpu_seconds']/3600:.4f} core-hours**.

| Stage family | Jobs | Allocation elapsed | Science-stage timer, seconds | Equivalent busy CPU cores | Consumed/allocated CPU proxy | Largest reported step RSS, GiB |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Fairness/training accounts for **{100*p['by_kind']['fairness']['allocation_elapsed_seconds']/t['allocation_elapsed_seconds']:.1f}%** of complete allocation time, interactions **{100*p['by_kind']['interactions']['allocation_elapsed_seconds']/t['allocation_elapsed_seconds']:.1f}%**, and GPU kernel experiments **{100*p['by_kind']['kernels']['allocation_elapsed_seconds']/t['allocation_elapsed_seconds']:.1f}%**. The three continuum fairness jobs take 719–729 seconds each, compared with 267–272 seconds for their discrete counterparts. This is an operational cost observation; it does not by itself assign the difference to a particular mathematical operation.

Science-stage timers sum to **{clock(t['science_elapsed_seconds'])}**. Their difference from allocation time is **{t['allocation_elapsed_seconds']-t['science_elapsed_seconds']:.2f} seconds**, covering readiness, CLI startup/prerequisite checking, reporting, launch and other process overhead. These timers include the scientific stage's own mixed numerical/I/O work and are not pure kernel timers. They must not be added to allocation time.

Tower worker timestamps span **05:23:18.677 to 06:42:09.900 UTC**, or **{t['observed_campaign_span_seconds']:.3f} seconds**. Successive worker gaps total **{t['launch_and_between_worker_gaps_seconds']:.3f} seconds**, ranging from 1.864 to 4.121 seconds. This gives no evidence of a substantial idle gap between this campaign's workers. Exact scheduler `Start`, `End`, `Eligible` and queue-reason history were not exported. Submission-to-worker-start includes the planned dependency chain and must not be called external queue delay.

## Memory, utilization and opportunities

The largest sampled Slurm step RSS is **{t['max_reported_step_rss_bytes']/2**30:.3f} GiB**, in preparation. Peak recorded PyTorch allocations/reservations are **{t['peak_torch_allocated_bytes']/2**20:.2f}/{t['peak_torch_reserved_bytes']/2**20:.2f} MiB**. The largest sampled whole-device use is **{t['peak_sampled_device_used_bytes']/2**30:.3f} GiB**, which includes the display and other GPU processes and is not attributable entirely to TDN. This run gives no reason to expand VRAM budgets.

Preparation uses about **6.38 equivalent CPU cores**, whereas fitting and interaction workers use approximately one core while reserving eight. GPU workers also consume approximately one host core while reserving four. **CPU utilization is not GPU utilization**: neither these ratios nor low allocator usage establish low GPU occupancy, launch-bounded execution or a particular optimization benefit. Those questions require the separately measured profiling results and, for device utilization/energy, appropriate telemetry.

A defensible operational improvement is to assess smaller host CPU/memory reservations for the serial fitting/interactions stages, while retaining preparation's parallel capacity and measurement headroom. Such changes can free resources for the user's other projects; they are not established solver speedups. Avoid reducing memory solely to this single observed maximum because Slurm RSS is sampled and subprocess peaks can be missed.

No electricity meter, GPU utilization stream, electricity tariff or monetary resource price is available. Energy and dollar costs remain **NA**. This campaign cost includes preparation, diagnostics, fitting, optimization trials, readiness and reporting, including any internally unsuccessful optimizer trials. It excludes earlier campaigns and unrelated scheduler jobs.

## Tower reporting: complete tables, missing graphical index

All **28** native Tower table projections report `reporting_complete=true`, zero omissions, the correct job/workflow/source binding and `VERIFIED_CANONICAL` source rows. Their experiment denominators match the stage summaries. The publisher preserves **72,622 learning observations** with exact source pointers and optimizer-trial identity.

There is a narrow **presentation defect**: the scientific report declares **22 figures**, with **44 PNG/PDF artifacts**, but its Tower projection has **`figure_count=0`**. Resolve writes these files directly into `report/`; the publisher's graphical discovery recognized only a directory named `figures`. The source atlas/browser index and artifact inventory remain present, but the dedicated Tower figure index is empty. Thus complete table projection must not be described as complete graphical integration. The scientific data, scores and native evidence must remain unchanged while this project-owned adapter is fixed. The original rendered files were omitted from the upload, so their appearance cannot be inspected here.

A separate source patch and regression test will repair discovery for the actual Resolve report-root layout, with declared file hashes checked before registration. It does not require any numerical rerun or modification of Tower itself. This review records the original defect and the native f07f9d5 evidence independently of that future patch.

## Interpretation

Operationally this campaign is clean, bounded and traceable: no timeout, allocation failure, readiness failure, source mismatch or required numerical-correctness failure was found. That establishes successful execution and retained-artifact integrity, **not scientific superiority**. The independent fit, interaction, competitor-training, kernel and memory analyses must determine which development hypotheses improved and which failed. All such evidence remains development data.
'''
(base/'provenance-costs.md').write_text(text)
