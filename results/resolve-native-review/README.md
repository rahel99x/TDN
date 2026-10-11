# Full native Resolve review

[Read the comprehensive assessment](../../docs/reviews/resolve-full-native-20261011.md).

This package reviews `fedora-resolve-20261011T052313346218Z`, native commit
`f07f9d56ccaee85b73b8d090aec3bc22ac067cf9`, on Fedora / RTX 4090.
It preserves development results, negative findings and uncertainty. The review
does not introduce a new confirmation claim or change any native score.

## Findings and source-traced detail

| Detail | Principal result | Machine-readable table |
|---|---|---|
| [Coefficient fitting](details/fit-review.md) | Optimization nearly exhausted; modest parent gain, substantial GL4 deficit | `data/fit-review.json.gz` |
| [Finite-step interactions](details/interactions-review.md) | Leading cubic often opposes the missing correction; nested improves most cases | `data/interactions-review.json.gz` |
| [FNO fairness and conditioning](details/fairness-review.md) | All settings trained; scaling remains a serious unresolved lead | `data/fairness-review.json.gz` |
| [GL3 and shared-transform GL4](details/kernels-review.md) | Same-rule GL4 implementation 1.5–2.3× faster in measured conditions | `data/kernels-review.json.gz` |
| [Memory and noise](details/memory-review.md) | Clean benefit fails under noise; validation selects zero gain | `data/memory-review.json.gz` |
| [Provenance, cost and Tower](details/provenance-costs.md) | 28 completed jobs, 53m23s GPU allocation; missing figure index identified | `data/provenance-costs.json.gz` |

The high-resolution [overview PNG](figures/resolve-native-overview.png) and
[vector PDF](figures/resolve-native-overview.pdf) summarize separate objectives;
they are not a universal leaderboard. Plotted values and their sources accompany
the figure. These are new review visualizations, not the omitted native images.

## Verification boundary

All four parts and all 3,716 retained files passed transport verification.
The compressed archive is 86,547,084 bytes, SHA-256
`a0db3b400425d6f0ca02072e024927e192fc863bf88ff7709f2deccbbaa9bbb2`.
The optional uploaded manifest SHA-256 is
`a83ab6aeab135d2f1efa664dfb880c1b2f08e452634f8b43605056791e736ae7`.
The native scientific source fingerprint is
`5d73ce536ff1bc8a354d68e641b83f4f5afa46481dadb6b853fb86b26480be2e`.

The compact upload omits checkpoints, arrays and rendered figures. Their declared
hashes are not equivalent to verifying their bytes. The audits reconstruct
results from retained measurements and metadata; they do not replay native
predictions from omitted weights or independently certify reference accuracy.

`provenance/` retains the original upload index/manifest, exact canonical-file
index, original protocol/stage metadata, transport log and the separate Tower
fix validation. JSON is losslessly gzip-compressed. `review-manifest.json`
hashes this derivative review package. Original multi-part uploads remain
outside Git; this package cannot substitute for them in a full raw-data audit.

## Reproducing the analysis

The scripts are the review's actual analysis programs. They expect the exact
canonical store in `.runtime/resolve-native-analysis/` and the four original
parts plus original `index.json`/`manifest.json` in
`.runtime/resolve-native-upload/`, all inside the checkout.

Use the original native source commit in a separate checkout for provenance
reproduction. `provenance-costs.py` intentionally rejects a different current
commit/source fingerprint. In particular, the reporting fix published with this
review changes the source identity; it must not be misrepresented as the source
that produced the native results.

With the original uploads in place and a project venv containing the declared
dependencies, from that checkout:

```bash
mkdir -p .runtime/resolve-native-analysis
# Copy this package's audit-scripts/*.py into that directory.
.venv/bin/python scripts/compact_review.py verify .runtime/resolve-native-upload/index.json
.venv/bin/python .runtime/resolve-native-analysis/extract.py
.venv/bin/python .runtime/resolve-native-analysis/provenance-costs.py
.venv/bin/python .runtime/resolve-native-analysis/fit-analysis.py
.venv/bin/python .runtime/resolve-native-analysis/interactions-review.py
.venv/bin/python .runtime/resolve-native-analysis/fairness-analysis.py
.venv/bin/python .runtime/resolve-native-analysis/kernels-review.py
.venv/bin/python .runtime/resolve-native-analysis/memory-review.py
```

The extractor must start with an absent `objects/` directory: it refuses to
overwrite an existing canonical store. It selects and hashes original bytes;
it changes no measurement. Its packaged root lookup is portable rather than
hard-coded to the original cloud path. Audit scripts write derived outputs in
their working analysis directory. The field bootstrap is post-hoc and conditional
on the frozen fit, not retraining, multiplicity adjustment or confirmation.

Read the small derived tables without extracting large arrays:

```python
import gzip, json
with gzip.open("results/resolve-native-review/data/fit-review.json.gz", "rt") as f:
    review = json.load(f)
```

## Refreshing the completed run's Tower figures

The native report had complete tables but an empty Tower figure index. The
project-side patch recognizes Resolve's declared report-root PNG/PDF layout,
verifies content hashes and exposes omissions. Tower itself is unchanged.
Validation: 44 publisher tests passed; an actual existing cloud atlas import
registered 32 PNG/PDF files while preserving all 43 science-artifact hashes.
This is CPU integration verification, not a new GPU scientific run.

On the desktop, after this review's commit has been pulled:

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/tower.sh import runs/fedora-resolve-20261011T052313346218Z/report
```

The command prints a **new** `report-tower/tdn-import-*` directory. Pass that exact
directory to `bash scripts/tower.sh validate` and `bash scripts/tower.sh show`.
The old science and old Tower report remain intact. No numerical jobs or
completed experiments need to be rerun.
