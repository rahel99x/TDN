# Selected CARC smoke configuration

`configs/carc-smoke.yaml` is the selected starting manifest for the bounded
12-step development run. Its settings were compared on the cloud CPU using the
same immutable dataset and validation parents. Actual A100 throughput tuning and
CUDA validation of this manifest still need the allocated CARC run.

The selected settings are an 8×8 grid, width 32, learning rate `0.003`, one parent
per optimizer step, 64 cells per microbatch/decoder chunk, and 12 optimizer steps.
Keep eager FP32, TF32 off, FP64 teachers, synchronous data loading, the 30 GiB
soft VRAM cap, and the proposed development tolerance `0.002`. A 64-cell chunk
covers this complete grid; the old 2048 limit also executed one chunk, so this
change does not imply a speedup or a smaller effective batch.

## Measured comparison

Four candidates were declared before training: widths 16 and 32 crossed with
learning rates `0.001` and `0.003`. Every candidate used the same seed and parent
design: four training, two validation, two diagnostic, and zero confirmatory
parents. Each completed 12 steps three times, with rotating timing order.
Selection first required feasible full-rollout validation at physical time
`0.64`, then minimized mean validation error; median training time would break
an exact tie. No diagnostic parent contributed to that selection.

| Width | Learning rate | Best validation mean error | Worst validation error | Best step | Median CPU training seconds |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.001 | 0.000375189 | 0.000377038 | 10 | 0.163 |
| 16 | 0.001 | 0.000302498 | 0.000317082 | 10 | 0.125 |
| **32** | **0.003** | **0.000275985** | **0.000308585** | **12** | **0.140** |
| 16 | 0.003 | 0.000661215 | 0.000661829 | 12 | 0.147 |

All four candidates met the unchanged `0.002` tolerance. Identical validation
values across repetitions establish deterministic replay of this small example;
they do not add independent validation parents. Measurements used a Xeon Platinum
8370C, Python 3.12.14, PyTorch 2.10.0+cpu, and one Torch intra/inter-operation
thread. Times include validation and checkpoint writes but exclude interpreter
startup, Torch import, and dataset generation. The first baseline repetition
paid a 1.448-second cold optimizer/import cost; raw repeats retain that value.
These timings do not establish which model is fastest on the A100.

After selection was fixed, CPU audit, dataset reuse, training, evaluation and
benchmark commands all completed. The learned method had zero failed diagnostic
parents and worst diagnostic error `0.000549054` across the required constant,
mixed and unseen-horizon sequences. Five CPU stages are implementation evidence;
CARC GPU tests and calibration remain separate requirements.

[The machine-readable report](../results/carc-config-selection.json) records the
complete candidate configurations, exact commands, software/source fingerprints,
immutable dataset and split hashes, raw repetitions, checkpoint/result checksums,
and post-selection CPU checks. Local generated artifacts are preserved under
`runs/carc-config-selection-20261003` and are not committed as source.

## Repeat the bounded comparison

The optional `scripts/select_carc_config.py` reproduces the four-candidate,
three-repeat comparison and writes its plan, complete raw measurements, summary,
and selected manifest into a fresh project-local run directory. It verifies the
project venv and actual runtime policy before creating run outputs, refuses
changes to the fixed smoke design, and checks scientific source/config/selector
fingerprints before and after every trial. An existing run directory is refused.
No diagnostic or confirmatory parent is used for selection.

After the workflow has prepared the venv and Python module, submit this optional
CPU task from the project checkout; `srun` supplies the real allocation and job
step. Do not run the Python comparison directly on a CARC login node.

```bash
cd /home1/aadaniel/projects/TDN
tdn_selection_run="$PWD/runs/config-selection-$(date -u +%Y%m%dT%H%M%SZ)"
srun --account=anakano_81 --partition=main --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:10:00 \
  .venv/bin/python scripts/select_carc_config.py --run-dir "$tdn_selection_run"
```

This is a proposed 10-minute CPU request for 144 total optimizer steps; it does
not request a GPU or edit the checked-in configuration automatically. Inspect
`summary.json` and `selected-config.yaml` in that new run. The prepared reusable
helper adds runtime/storage/scope guards to the historical measurement script;
the report preserves the exact historical command and script checksum separately.
A cloud checkout can also execute this bounded helper through its standalone
project `.venv` with explicit CPU scope; that result remains cloud CPU evidence.

## Scope

This is the best validation error among four tiny development candidates. Two
validation parents and twelve steps cannot establish a global optimum, robust
hardware tuning, or a TDN efficiency advantage. The actual CARC software is
Python 3.11.9 / PyTorch 2.10.0+cu126 and must validate the chosen settings there.
The historical G2/G4 failures remain blockers to larger efficiency experiments.
`configs/pilot.yaml` keeps `require_headroom: true`; the smoke's permissive setting
only permits this bounded diagnostic, and no confirmatory parents are opened.
