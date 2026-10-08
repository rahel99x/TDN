TDN COMPLETE EXPERIMENT ATLAS

Open TDN-complete-experiment-atlas.pdf and zoom. Page 1 maps EVERY archived experiment, check and software-test occurrence. Pages 2–4 show corrected horizon-matched comparisons at all three tolerances. The lossless PNG is 11520 x 14720 pixels. The preview is for navigation; use the full PNG/PDF for cells.

COVERAGE
74,150 experiment rows = 74,149 scientific records + 1 report-inventory record.
619,697 logged individual checks, including required / inapplicable / NA checks.
642 native software-test occurrences = 641 passed + 1 skipped.
Each color cell is a logged occurrence, NOT a new independent sample.

FIND A CELL
Each stage is separately indexed from zero. Read left to right then downward.
Experiment verdict and score maps: 256 columns.
Individual check maps: 768 columns.
JUnit software-test strips: 150 columns.
Index = row * number_of_columns + column.
White cells are padding and do not refer to a test.

Unzip the CSV .gz files (gzip-compatible, e.g. 7-Zip), then filter by stage and ID or coordinates in a spreadsheet. The experiment lookup links to source ledger line and SHA256. The check lookup includes identity, measured value, target, comparison, required/applicable flags and reason. JUnit CSV includes exact parameterized test identities.

HOW TO READ SCORES
GOOD: declared checks met; BAD: at least one required failure; NA: incomplete evidence. The 1–100 score is required-check weight met, with no credit for NA; not model quality, probability or mathematical proof. Mechanism rows overlap and cannot be summed as independent evidence.

KNOWN SCIENTIFIC LIMITS
The original frontier rows sometimes mix final times. Page 1 preserves their recorded scores; pages 2–4 reconstruct comparisons with T=0.27 and T=0.81 kept separate. It uses existing measurements and post-hoc feasible schedule selection, not a new experiment or deployable policy. Formal fallback amortization is not displayed as a win because identical fallback work has a first-call timing confound. Internal FNO baselines are not an official paper reproduction.

PROVENANCE
Source is the uploaded full Fedora/RTX4090 run, with immutable scientific ledgers preserved. atlas-manifest.json records source hashes and an independent one-to-one CSV coordinate coverage audit. Source archive Python / pickle code was not executed. build_atlas.py is the review-side plotting script, requiring numpy, matplotlib and Pillow. No observations were dropped or resampled.

REGENERATE
python build_atlas.py --run-root /path/to/extracted/fedora-roadmap-20261008T022526329292Z --review-root /path/to/review-json-directory --output-dir /path/to/new-figures
The review directory must contain aggregate-review.json, neural-review.json, neural-fixed-horizon-review.json, policy-review.json and the uploaded archive index. Its independently derived summaries are additional inputs to the plot, not newly generated science.

ALLOCATIONS
Slurm allocation elapsed seconds (jobs 197–206): 28,118,114,536,148,1757,1066,390,382,140. Total 4,679 s; GPU allocations 3,741 s. Allocation bars include stage worker/export overhead but exclude queue time. No invented monetary cost.
The skipped software test requires an optional unchanged Tower checkout. Separate read-only native Tower review validated all 45,277 exported pages.
