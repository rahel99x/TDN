# Next actions

The next bounded neural experiment is the [three-seed replication](docs/NEURAL_REPLICATION.md).
In the existing CARC checkout and verified `.venv`, pull the new source, then run
`bash scripts/carc_neural_replication.sh --submit`. It uses one 30-minute CPU
job for five fixed families, three paired training seeds and 27 fresh diagnostic
parents. Review final-rollout and transient endpoints separately; optional
frozen A100 timing is a separate explicit 30-minute request. The
[cloud CPU validation](results/neural-replication-validation.json) passed the
implementation checks and retained the clock's transient accuracy weaknesses.
This development study does not authorize a pilot or confirmatory campaign.

1. Deploy this repository at `/home1/aadaniel/projects/TDN` on CARC as `aadaniel`.
2. Follow [docs/CARC.md](docs/CARC.md): read-only discovery, explicit five-minute allocated driver audit, then CPU-allocated venv setup using an official compatible CUDA wheel. Every submission charges `anakano_81` and verifies live association/limits.
3. Review `bash scripts/pipeline.sh configs/smoke.yaml --pipeline-id smoke-review` (dry run). The corresponding explicitly submitted smoke validates actual CPU/GPU capabilities without a scientific performance claim.
4. Run the headroom-required `configs/pilot.yaml` audit before any long pilot. If classical controls remove the gap or temporal fits add no benefit, record the result and stop this avenue or revise the workload through a new scientific design.
5. Only after measured G0–G6 and a separately reviewed tolerance/observable/parameter/untouched-parent manifest, design confirmation, stiffness and resolution-transfer experiments. No full campaign, KAN branch or distributed experiment is launched by default.

See `results/gates.json` and `results/validation.json` for the prepared-machine evidence and limitations. This environment has no live CARC account or GPU allocation, and no job was submitted.
