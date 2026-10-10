# Watching a frozen adjacent model predict

These illustrations replay a saved checkpoint and held-out input from the
isolated adjacent study. They show **this adjacent interaction prototype**, not
a replacement for the historical rank1 model or a confirmed best architecture.
No training or model selection is performed by the renderer.

The three high-resolution views are available in
[`results/adjacent-prediction-view`](../results/adjacent-prediction-view/):

| View | PNG dimensions at 400 DPI | What to inspect |
|---|---:|---|
| `prediction-architecture` | 6400 × 6800 | Input, low/high fluctuations, signed LL/LH/HH interactions, actual gains, weighted contributions, physical backbone, prediction and reference error |
| `prediction-parameters` | 6000 × 5200 | All 59 saved neural weights/biases, hidden activations, output gains, and the resulting change from unit gains |
| `prediction-sensitivity` | 6400 × 4800 | Each effective gain's signed spatial response and explicitly hypothetical bounded gain perturbations |

Every view includes a PNG and a PDF. The PDF retains vector labels and parameter
text for zooming. Spatial fields retain their actual 8 × 8 cells: high export
resolution improves readability, not the underlying simulation resolution.
No smoothing or invented subgrid detail is added.

## What the overlay means

The physical branch receives the entire field. It splits fluctuations about the
mean into low and high modes, computes signed quadratic interactions from
`vL²`, `2 vL vH`, and `vH²` under the shared physical transport, then projects
the correction to the declared output cutoff. The learned branch receives three
global features: mean state, `log1p(h r)`, and `log1p(h active diffusion rate)`.
A 3 → 8 → 3 SiLU network maps those features to three gains,
`g = 1 + 0.75 tanh(logit)`. The final prediction adds the gain-weighted correction
to the analytic backbone.

The neural weights are **not spatial pixel weights**. Their relationship to
space is displayed through the physical channel they modulate. The architecture
view therefore places each numerical gain beside its signed spatial channel and
its weighted contribution. The parameter view separately exposes the full
feature-to-gain calculation. Color bars distinguish state values, signed
corrections and errors; lower absolute error is better, while signed values and
larger gains have no universal better direction.

The sensitivity is the derivative of the analytic linear combination with
respect to an independently varied effective gain, holding the field and physics
fixed. It is not an input saliency map, a neural-weight derivative, or a derivative
of floating-point rounding. Hypothetical probes use FP64 arithmetic around the
actual FP32 endpoint and are clipped to the admissible gain interval. They are
not newly trained models or newly selected predictions.

## What this example establishes

The displayed checkpoint was selected after two smoke-training updates. Its
gains are approximately LL **0.996078**, LH **0.996085**, HH **1.003922**. The
standalone gain-mixing change is only about **3.15e−10** in maximum magnitude.
For this field the complete selected and unit-gain endpoints are **identical
after FP32 rounding**; both have RMS error about **6.54e−8**. A rescaled channel
picture must not be mistaken for an accuracy gain. This example explains the
mechanism, not scientific superiority or training convergence.

The renderer verifies the sealed pilot and evaluation lineage, checkpoint,
reference-bank hashes, continuous input identity and protected implementation
against the experiment's source commit. It replays the original one-step error
metrics and verifies source files remain unchanged after plotting. Historical
full prediction arrays were not stored, so this is endpoint-metric replay, not
a byte-for-byte comparison with a historical prediction array. Replaying a GPU
run on CPU is explicitly labeled an alternate-device illustration, not GPU parity.

`prediction-data.json` records features, gains, all parameters, physical settings,
reference uncertainty, original metrics and source identities.
`prediction-arrays.npz` retains plotted arrays, and `manifest.json` hashes every
output. The original experiment remains immutable.

## Reproduce or visualize a fresh run

Run from the isolated checkout with its own venv. For the included cloud run:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/adjacent_prediction_view.py \
  --run-dir runs/adjacent-local-review-v2 \
  --output-dir results/prediction-replay-new \
  --family channel_neural --track discrete --parent-index 0 --dpi 400
```

A clone includes that run's complete sealed archive at
`results/adjacent-development/archives/smoke-v2.tar.gz`; the `runs/` directory
itself is not version controlled. For a new native Fedora campaign, replace
`--run-dir` with the completed coordinator path printed by
`bash scripts/fedora_adjacent.sh status latest`, and choose a new output directory.
The public renderer accepts 400–600 DPI. Keep scientific source and environment
consistent with the frozen experiment; do not bypass an identity failure.

`--family` also supports `channel_global`, `channel_affine` and `channel_fixed`.
Use `--seed` and `--horizon` when more than one frozen choice is available.
Parent and method are explicitly chosen, never selected by the nicest error plot.
These images do not add deployment timing or energy measurements.
