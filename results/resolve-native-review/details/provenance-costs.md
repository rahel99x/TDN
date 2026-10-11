# Native Resolve provenance, operational cost and reporting audit

Run: `fedora-resolve-20261011T052313346218Z`. All results here are a **development campaign**, not untouched confirmation. Analysis reads exact retained bytes from `review_io.py`; it does not deserialize a checkpoint or re-run the numerical experiments.

## Provenance and transport

**No provenance, protocol, dependency, allocation, software, required-test, or canonical-table inconsistency was found** in the checks implemented in `provenance-costs.py`. One distinct figure-discovery defect is described below; it does not alter the numerical results.

- Independently rechecked all four uploaded parts, their concatenated **86,547,084-byte** archive, and the separately uploaded optional manifest. Its measured SHA-256 `a83ab6aeab135d2f1efa664dfb880c1b2f08e452634f8b43605056791e736ae7` matches the index's `manifest_sha256` exactly.
- Archive SHA-256: `a0db3b400425d6f0ca02072e024927e192fc863bf88ff7709f2deccbbaa9bbb2`. The prior complete transport verification covered **3,716 retained files**. This audit independently rehashed **681 canonical files** after decompression and matched their sizes/digests to that verified original export inventory.
- Native source commit: `f07f9d56ccaee85b73b8d090aec3bc22ac067cf9`; native scientific source SHA-256: `5d73ce536ff1bc8a354d68e641b83f4f5afa46481dadb6b853fb86b26480be2e`; controller fingerprint: `5fbe49bb33af65c7ff31a3bb576f4fd351290fb605aab1953f0ddb2a3b2aa5bc`. The current checkout matched all three **before the separate reporting-only fix**.
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

Total allocation time: **1:18:51** (**1.3142 allocation-hours**). GPU reservation time: **0:53:23** (**0.8897 GPU-hours**). Allocated host CPU time: **6.8567 core-hours**; reported consumed CPU time: **1.6224 core-hours**.

| Stage family | Jobs | Allocation elapsed | Science-stage timer, seconds | Equivalent busy CPU cores | Consumed/allocated CPU proxy | Largest reported step RSS, GiB |
|---|---:|---:|---:|---:|---:|---:|
| audit | 1 | 0:00:20 | 0.35 | 0.96 | 24.1% | 0.823 |
| fairness | 6 | 0:49:44 | 2898.87 | 1.00 | 25.1% | 1.443 |
| fit | 2 | 0:00:32 | 20.63 | 0.95 | 11.8% | 0.643 |
| interactions | 12 | 0:20:00 | 1127.54 | 0.99 | 12.4% | 0.578 |
| kernels | 4 | 0:03:39 | 167.22 | 0.99 | 24.8% | 1.264 |
| memory | 1 | 0:00:14 | 8.38 | 0.96 | 24.1% | 0.573 |
| prepare | 1 | 0:03:28 | 201.64 | 6.38 | 79.7% | 2.791 |
| report | 1 | 0:00:54 | 34.63 | 0.98 | 24.4% | 1.347 |

Fairness/training accounts for **63.1%** of complete allocation time, interactions **25.4%**, and GPU kernel experiments **4.6%**. The three continuum fairness jobs take 719–729 seconds each, compared with 267–272 seconds for their discrete counterparts. This is an operational cost observation; it does not by itself assign the difference to a particular mathematical operation.

Science-stage timers sum to **1:14:19**. Their difference from allocation time is **271.75 seconds**, covering readiness, CLI startup/prerequisite checking, reporting, launch and other process overhead. These timers include the scientific stage's own mixed numerical/I/O work and are not pure kernel timers. They must not be added to allocation time.

Tower worker timestamps span **05:23:18.677 to 06:42:09.900 UTC**, or **4731.223 seconds**. Successive worker gaps total **63.245 seconds**, ranging from 1.864 to 4.121 seconds. This gives no evidence of a substantial idle gap between this campaign's workers. Exact scheduler `Start`, `End`, `Eligible` and queue-reason history were not exported. Submission-to-worker-start includes the planned dependency chain and must not be called external queue delay.

## Memory, utilization and opportunities

The largest sampled Slurm step RSS is **2.791 GiB**, in preparation. Peak recorded PyTorch allocations/reservations are **68.10/70.00 MiB**. The largest sampled whole-device use is **5.020 GiB**, which includes the display and other GPU processes and is not attributable entirely to TDN. This run gives no reason to expand VRAM budgets.

Preparation uses about **6.38 equivalent CPU cores**, whereas fitting and interaction workers use approximately one core while reserving eight. GPU workers also consume approximately one host core while reserving four. **CPU utilization is not GPU utilization**: neither these ratios nor low allocator usage establish low GPU occupancy, launch-bounded execution or a particular optimization benefit. Those questions require the separately measured profiling results and, for device utilization/energy, appropriate telemetry.

A defensible operational improvement is to assess smaller host CPU/memory reservations for the serial fitting/interactions stages, while retaining preparation's parallel capacity and measurement headroom. Such changes can free resources for the user's other projects; they are not established solver speedups. Avoid reducing memory solely to this single observed maximum because Slurm RSS is sampled and subprocess peaks can be missed.

No electricity meter, GPU utilization stream, electricity tariff or monetary resource price is available. Energy and dollar costs remain **NA**. This campaign cost includes preparation, diagnostics, fitting, optimization trials, readiness and reporting, including any internally unsuccessful optimizer trials. It excludes earlier campaigns and unrelated scheduler jobs.

## Tower reporting: complete tables, missing graphical index

All **28** native Tower table projections report `reporting_complete=true`, zero omissions, the correct job/workflow/source binding and `VERIFIED_CANONICAL` source rows. Their experiment denominators match the stage summaries. The publisher preserves **72,622 learning observations** with exact source pointers and optimizer-trial identity.

There is a narrow **presentation defect**: the scientific report declares **22 figures**, with **44 PNG/PDF artifacts**, but its Tower projection has **`figure_count=0`**. Resolve writes these files directly into `report/`; the publisher's graphical discovery recognized only a directory named `figures`. The source atlas/browser index and artifact inventory remain present, but the dedicated Tower figure index is empty. Thus complete table projection must not be described as complete graphical integration. The scientific data, scores and native evidence must remain unchanged while this project-owned adapter is fixed. The original rendered files were omitted from the upload, so their appearance cannot be inspected here.

A separate source patch and regression test will repair discovery for the actual Resolve report-root layout, with declared file hashes checked before registration. It does not require any numerical rerun or modification of Tower itself. This review records the original defect and the native f07f9d5 evidence independently of that future patch.

## Interpretation

Operationally this campaign is clean, bounded and traceable: no timeout, allocation failure, readiness failure, source mismatch or required numerical-correctness failure was found. That establishes successful execution and retained-artifact integrity, **not scientific superiority**. The independent fit, interaction, competitor-training, kernel and memory analyses must determine which development hypotheses improved and which failed. All such evidence remains development data.
