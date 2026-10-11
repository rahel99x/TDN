# Native Advance full: compact independent review

Read the [main assessment](../../docs/reviews/advance-full-native-20261011.md). This bundle reviews `fedora-advance-full-20261011T001715`, source `a7646702ff1061d78a722153ef2548d9d831de0d`. No training, inference, source modification or new confirmation was performed for this review.

## Contents

- `advance-full-primary-results.png`: 5400×3900 pixels, 300 dpi. A vector PDF is alongside it. All 20 primary accuracy comparisons and the corresponding accuracy-qualified field coverage are shown.
- `primary-results-data.json`: self-contained figure data with exact claim records, source hashes and absent-schedule explanations. `claims.json` is the original, byte-identical primary claim file.
- `all-model-metrics.csv` and its Markdown note: absolute metrics for all 13 families, separated by track, final time and training count; 60 rows. Same-schedule latency is not accuracy-qualified solver cost.
- `statistics.md`, `training.md`, `mathematics.md`, `references.md`, `costs.md`: five detailed independent audits, including negative findings and limitations.
- `provenance-summary.json`: stage/source/protocol/prerequisite consistency checks for all 49 stages. All retained upload bytes were verified before analysis.
- `data/*.json.gz`: lossless compressed numerical review outputs, original protocol/catalog/accounting, source-file index and original upload index/manifest. Gzip is storage compression; no rounding or lossy numeric transformation was applied.
- `audit-scripts/`: exact scripts used for the independent review. They require the canonical evidence store described below. They are analysis artifacts, not replacements for the experiment workflow.
- `review-manifest.json`: SHA-256 and byte length for each published artifact and the main assessment, plus provenance and replay limitations.

The primary outcome is **14 GOOD / 6 BAD accuracy claims, but 0 GOOD / 14 BAD / 6 NA cost claims**. Labels are frozen scoped criteria; they are not probabilities of mathematical correctness or a universal model ranking. GL4 beats the fitted candidate in all four primary accuracy cells. Local FNO controls retain substantial optimization limitations.

## Reproduce the figure

The plot is self-contained and uses the packaged raw figure data. From the repository root, with NumPy and Matplotlib installed in the project venv:

```bash
.venv/bin/python results/advance-full-review/plot_primary_results.py
```

This writes PNG/PDF alongside the script. It does not retrain models. The packaged provenance describes the supplied rendering; it is not automatically regenerated. Rendering metadata and PDF bytes may vary with software versions; numerical claim records must remain identical. The figure uses Ours triangles, analytic squares and Theirs circles, thin lines, and explicit direction guidance. Intervals are unadjusted field/seed bootstrap intervals, not observed-range bands.

## Reproduce the deeper audit

The original 12 upload parts, `index.json` and `manifest.json` were preserved under `.runtime/advance-full-native-upload/`. Full transport verification was:

```bash
bash scripts/compact_review.sh verify .runtime/advance-full-native-upload/index.json
```

The review extracted a canonical exact-byte subset into `.runtime/advance-full-native-analysis/objects/<sha256>.gz`, indexed by `evidence-index.json`. Each object decompresses to the original hashed bytes. `review_io.py` reads this store without expanding the 2.4 GB archive. The object store remains local; it is **not included in this small report bundle**.

The copies in `audit-scripts/` preserve the executed analysis code. To rerun them, place the desired script and `review_io.py` beside the original `evidence-index.json` and `objects/`, and execute from the repository root. Statistics/reference/cost scripts use only this canonical evidence. Provenance checking additionally requires the sibling original upload index and manifest. `math-summary.py` also compares against `.runtime/advance-exploration-full-development/`; that historical development store is needed to reproduce its prior-result identity check. `statistics-publish-table.py` writes the table under this repository's results directory. `extract.py` records the original `/workspace/TDN` review location; change its `ROOT` explicitly before rebuilding the store elsewhere, and use a new empty output directory.

The small bundle contains derived summaries and their scripts, not the entire raw archive. It cannot independently replay the 1,282 intentionally omitted original arrays/checkpoints/figures. Retained artifact consistency, successful native execution, numerical correctness checks and scientific success remain separate claims.

## Scope of inference

There are 64 independent main-confirmation fields. Repeated seeds, grids, schedules and timing samples do not multiply that sample size. Primary confirmation uses 32²/64²; 128² belongs to the separate eight-field scaling study. Additional subgroup, sensitivity and all-control contrasts are explicitly exploratory. Prototypes reuse previously inspected development parents. New methods motivated by this review require new confirmation fields.

All files are new review outputs. Historical artifacts and scientific source code remain unchanged.
