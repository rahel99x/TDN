# Isolated adjacent interaction study

This branch is independent of the running portfolio. Do not merge it or pull it
into a checkout used by live jobs. The [handoff](ADJACENT_INTERACTION_STUDY_PROMPT.md),
[evidence audit](ADJACENT_EVIDENCE.md), and [research register](ADJACENT_RESEARCH.md)
describe its scope. No result is automatically promoted into TDN.

## Separate Fedora checkout and venv

Fetching a branch does not change the active checkout's source. Run once:

```bash
git -C /home/rahel/TDN fetch origin research/adjacent-interactions
git -C /home/rahel/TDN worktree add --track -b research/adjacent-interactions \
  /home/rahel/TDN-adjacent origin/research/adjacent-interactions
cd /home/rahel/TDN-adjacent
bash scripts/fedora_adjacent.sh doctor
bash scripts/fedora_adjacent.sh configure --cpu-partition local --gpu-partition local \
  --gpu-gres gpu:1 --expected-gpu-name 'RTX 4090' --gpu-vram-gib 24
bash scripts/fedora_adjacent.sh setup
```

The setup checks the driver and creates this checkout's own Python venv. Use the
Fedora Python 3.13.13 interpreter; no Conda, copied CUDA venv or driver changes.
All run files, cache and temporary files stay inside this isolated checkout.
The host scheduler limit remains 110000 MiB; host RAM is not GPU VRAM.

## Plan and allocated smoke

```bash
cd /home/rahel/TDN-adjacent
bash scripts/fedora_adjacent.sh plan --smoke
bash scripts/fedora_adjacent.sh run --smoke
bash scripts/fedora_adjacent.sh status latest
bash scripts/fedora_adjacent.sh logs latest --lines 200
bash scripts/fedora_adjacent.sh validate latest
bash scripts/fedora_adjacent.sh paths latest
bash scripts/fedora_adjacent.sh collect latest
```

Submission discovers your active `tdn-portfolio-*` jobs and places the adjacent
chain after their completion. It does not cancel, alter or limit the main queue.
For an additional known campaign boundary, repeat `--after-job JOB_ID` on `run`.
This captures jobs present at submission; it cannot predict jobs you submit later.
The plan command performs no scheduler calls or numerical work. There is no
pending-job count cap. Within this study, jobs use a serial resource dependency;
scientific dependencies separately use `afterok`. The report uses `afterany` and
retains failed, missing and inconclusive work.

After smoke verifies, the larger pilot and fresh evaluation are available:

```bash
bash scripts/fedora_adjacent.sh plan --full
bash scripts/fedora_adjacent.sh run --full
```

This is a bounded local-method comparison, not a reproduction of a published
FNO benchmark. Three primary contrasts are corrected across both equation
tracks. The pilot freezes checkpoints and a sample-size calculation before
generating fresh evaluation fields. If reference uncertainty, optimization,
coverage or attainable precision is inadequate, the relevant claim remains NA.
Smoke/development results are exploratory regardless of the number of repeated
schedule rows. Subgroup analyses do not acquire confirmatory status afterward.

## Frozen finite ceilings

| Profile | Jobs | Scientific timer ceiling | Summed allocation ceiling | Longest job |
|---|---:|---:|---:|---:|
| smoke | 15 | 50 min 40 s | 2 h 17 min | 18 min |
| development | 15 | 2 h 12 min | 3 h 35 min | 28 min |
| full | 17 | 3 h 28 min | 5 h 5 min | 38 min |

These sums are safety ceilings, not runtime predictions or bills. Each job uses
four CPUs and 32 GiB host memory; GPU stages request one RTX 4090. GPU soft caps
remain 18 GiB and 75% of available memory, with the 90% hard limit. No allocation
exceeds the 45-minute maximum. Full training is split by equation track.

Discretionary architecture/exploration budgets are respectively 320/80 seconds
for smoke, 960/240 for development, and 2400/600 for full. Thus the two alternatives
retain **20%** of this study's discretionary budget. Diagnostic teachers,
evaluation, checks and reporting have separately enumerated ceilings. Nothing
is transferred from the main portfolio. Actual failed attempts and recovery
allocations remain in the accounting ledger; monetary and energy costs are NA
without measured values or rates.

## Recovery and artifacts

```bash
bash scripts/fedora_adjacent.sh recover latest --plan
bash scripts/fedora_adjacent.sh recover latest
```

Recovery requires terminal origin jobs and identical source, software, protocol
and hardware profile. It creates a new coordinator, references verified completed
units, and copies compatible interrupted journals into new attempt directories.
Original artifacts remain intact. Do not update code, the venv or the Slurm
profile during a campaign. Changed scientific code requires a fresh campaign,
not disabling a fingerprint guard. No automatic repeated resubmission occurs.

`paths latest` prints the atlas, machine-readable measurements and per-job Tower
metrics paths. `collect` includes science, models, field identities, checks,
failed logs, resource accounting and original recovery attempts. Large bundles
are split into 28 MiB parts with a hash index. Tower itself is unchanged.

## Cloud CPU checks

```bash
bash scripts/adjacent_local.sh --smoke --run-dir runs/adjacent-local-unique
```

`--development` is available; local `--full` and local CUDA execution are
rejected. CPU checks and successful collection of the 92 mandatory CUDA case
identities do not establish GPU readiness. Each allocated GPU stage executes
all 21 new channel cases plus the 71 existing portfolio cases without skips.

The atlas retains Ours/Theirs/analytic roles, triangles for ours, thin lines,
better/worse guidance, translucent continuous observed-range bands and visible
missing measurements. An observed range is not a confidence interval. Read
scientific effects, reference and sampling uncertainty, failure denominators
and complete costs separately from computational completion.

For high-resolution illustrations of a frozen checkpoint while predicting,
see the [prediction-image guide](ADJACENT_PREDICTION_IMAGES.md). It includes actual
parameter values, signed spatial responses, raw plotted arrays and replay commands.
