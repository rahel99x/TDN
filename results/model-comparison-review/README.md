# Complete model-comparison evidence annex

For the expert interpretation, model inventory, mathematical distinctions and limitations, read [TDN model and comparison review](../../docs/MODEL_COMPARISON_REVIEW.md). Review date: 2026-10-08, America/Los_Angeles. This annex extracts existing trusted report JSON/CSV only; no training, solver runs, checkpoint deserialization or uploaded-code execution occurred.

These tables preserve actual recorded pairings and missing results. They do not manufacture an all-versus-all tournament. Different studies have different cohorts, tolerances, targets, training budgets and hardware, so do not concatenate them into a common leaderboard. Many rows summarize overlapping observations; record counts are not independent sample counts. Models with the same identifier in different programs may have different architectures/configuration.

| File | Contents and scope |
|---|---|
| [initial-controls.json](initial-controls.json) | All six original 12-update CPU-smoke family frontiers; only two diagnostic parents, no general ranking claim. |
| [early-neural-comparisons.json](early-neural-comparisons.json) | Native A100 benchmark: all 56 eight-by-seven family aggregates and 504 parent-level comparisons, plus endpoint/training/metric definitions and original CPU-smoke controls. Direct neural controls with no feasible rollout retain unavailable speed comparisons. |
| [replication-comparisons.json](replication-comparisons.json) | Native CARC CPU three-seed replication: 15 family/seed endpoint summaries, 12 clock/neural pair summaries, three clock/classical summaries. All seeds share 27 diagnostic parents. |
| [premix-agenda-comparisons.json](premix-agenda-comparisons.json) | Exact native premix/consistency/agenda collections: 90 premix paired rows; 240 consistency paired rows; 16 constraint-overhead entries; six agenda compression-development dictionary entries; 112 agenda neural frontiers; 24 bare-DF comparisons; 120 canonical criteria (96 classical and 24 joint gates); 18 policy summaries. Supporting training/coverage/kernel/control information and compact-spatial review included. Counts describe different record types, not additive experiments. |
| [roadmap-comparisons.csv](roadmap-comparisons.csv) | All 420 existing same-schedule directional family-pair/track rows: 370 measured and 50 zero-row unavailable entries. Source pointers, denominators and exact values retained. There are 165 unordered pairs with evidence in at least one direction; some reverse-direction entries are unavailable because original seed matching has no aligned rows. |
| [frontier-comparisons.csv](frontier-comparisons.csv) | All 684 available selected frontier summary objects across nine existing review/native collections. Scalar columns aid filtering; `source_record` preserves nested original records. |
| [frontier-comparisons.json.gz](frontier-comparisons.json.gz) | Equivalent nested frontier record export, including source context and model IDs. |
| [frontier-comparisons-metadata.json](frontier-comparisons-metadata.json) | Exact collection counts, source hashes and metric/ratio definitions. |
| [frontier-matchup-coverage.csv](frontier-matchup-coverage.csv) | All 45 unordered pairs among ten frontier families: nine rank1-centered pairs have dedicated same-schedule summaries; 36 do not. Common-condition raw observations exist for other pairs, so absent dedicated summaries are not evidence that no raw comparison is possible. |
| [frontier-model-catalog.json.gz](frontier-model-catalog.json.gz) | All 100 original catalogue entries: 80 final records and 20 tuning records, retaining actual dimensions, parameters, selection states and effective configuration. |
| [provenance.json](provenance.json) | File hashes, extraction sources and review validation. |

The frontier's 684 summaries comprise 36 pooled attribution, 288 regime attribution, 72 grid attribution, 72 endpoint-frontier, ten fixed-policy-cell, 12 data-efficiency, two confirmation-gate, 48 scaling and 144 native field-bootstrap records. These overlap. The independent confirmation cohort has 24 fields.

## Read ratio definitions before ranking

For roadmap `median_rms_other_over_candidate` and `median_latency_other_over_candidate`, **greater than one favors the candidate**. Frontier competitor/candidate error and time ratios likewise favor the candidate above one. Initial and intermediate reports have several conventions; premix/agenda annex collections carry their own `ratio_convention` and metric notes. Do not invert only selected unfavorable results.

Same-schedule timing and accuracy-qualified endpoint-frontier timing answer different questions. Frontier classical controls received extra endpoint schedules; that asymmetry is retained in metadata. Conditional speed uses mutually feasible rows and must be read with candidate/control coverage. Reference-informed fastest-method choices are retrospective frontiers, not deployed policies.

Historical roadmap frontiers that mixed horizons are not reproduced here as valid cost comparisons. Use the separate [44,064 corrected fixed-horizon comparisons](../roadmap-full-review/fixed-horizon-comparisons.json.gz) and [corrected summaries](../roadmap-full-review/neural-fixed-horizon-review.json). The accompanying [expert roadmap analysis](../roadmap-full-review/neural-review.md) identifies the original defect and its impact.

## Additional complete recorded numerical comparisons

The authoritative reports for training-free studies remain in place:

- [Mechanism audit](../carc-mechanism-audit-review.json) and [interpretation](../../docs/MECHANISM_REVIEW.md).
- [Interaction screen](../carc-interaction-screen-review.json) and [interpretation](../../docs/INTERACTION_REVIEW.md).
- [Prepared work–precision](../carc-work-precision-review.json) and [interpretation](../../docs/WORK_PRECISION_REVIEW.md).
- [Compact spatial](../carc-compact-spatial-review.json) and [interpretation](../../docs/COMPACT_SPATIAL_REVIEW.md).
- [Roadmap numerical combinations](../roadmap-full-review/numerical-review.json), [policy](../roadmap-full-review/policy-review.json) and [all M00–M23/C0–C4 recorded summaries](../figures/roadmap-full-20261008/mechanism-recorded-summary.csv).

Source paths under `.runtime/` describe already-extracted native archive evidence available during this review; they are provenance, not a promise that a fresh checkout contains those archives. This annex retains the selected JSON values themselves, and the original archive/run identities and source hashes remain recorded. Historical reports and numerical artifacts have not been modified.
