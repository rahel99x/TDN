# Work log

The source baseline is commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; implementation and validation were completed locally before publication. Each executed CLI stage records its exact command, configuration hash, commit and source fingerprint in its `stage.json`. Historical development runs are retained and do not overwrite the final reports.

- M0 completed: audited the README-only checkout and entire uploaded protocol; preserved the original specification, source ledger and repository audit. The user requirements replace its Conda/scratch templates. CARC mount/account/Slurm access are unverified external prerequisites.
- M1 completed: mathematical values, jets, quadrature, state/time gradgrad, exact matrix anchor and split convergence tested in CPU FP64. Independently measured Psi maximum relative errors: FP64 `2.066338914451926e-15`, FP32 `7.410736190010583e-7`; evidence `results/math_validation.json`.
- M2/M3 completed for small development screening: teacher refinement, classical controls, hidden-input probes and temporal oracles executed. Logistic local/global slopes approximately 3.00985/2.00120. G2/G4 failed: numerical correctness does not establish useful method headroom. Evidence `results/audit.json` and `results/gates.json`.
- M4 completed as bounded CPU implementation diagnostics: immutable 4/2/2 train/validation/diagnostic parents, four paired horizons, full-domain 1/2/4-window teachers, 12-step pilot and six matched 12-step learned controls. Exact CPU committed resume, fresh-process serialization and actual USR1 exit75 tested. Evidence `results/validation.json` and `results/learned_controls.json`.
- M5 prepared, unexecuted on A100: strict allocated GPU preflight, parity, actual GPU signal/resume test, CUBLAS deterministic workspace, BF16/compiled forward-gradient audits and measured-memory/phase profiling scripts. No GPU/Slurm result is claimed.
- M6–M8 remain gated: no optional KAN/oscillatory/distributed branch or confirmatory test set was opened. CPU matched-tolerance benchmarking found no learned efficiency advantage; stop default efficiency expansion.
- Environment reproduction actually ran: `TDN_LOCAL_TEST_ROOT=/workspace/TDN TORCH_VERSION=2.10.0 TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_venv.sh`; exit0, compatible existing venv preserved, pip consistency passed. Tested CPU lock in `requirements/cpu.lock`; CARC creates its own exact wheel/version freeze after driver selection.
- Integrated command actually ran: `TDN_LOCAL_TEST_ROOT=/workspace/TDN bash scripts/smoke.sh configs/smoke.yaml /workspace/TDN/runs/final-smoke`; 91 CPU tests passed, four GPU tests deselected, all five functional stages completed. One new CLI regression plus overlapping controls passed in the subsequent eight-test run; total93 unique CPU tests including the final strict-memory-policy regression.
- Matched controls actually ran: `TDN_LOCAL_TEST_ROOT=/workspace/TDN bash scripts/compare_local.sh configs/smoke.yaml /workspace/TDN/runs/final-smoke/dataset /workspace/TDN/runs/final-controls`; exit0, all six trained/evaluated with feasible development policies.
- Pilot stopping rule actually ran: `.venv/bin/python -m tdn.cli audit --config configs/pilot.yaml --run-dir runs/pilot-screen`; expected exit1 at failed G2, output retained, no COMPLETED marker or downstream campaign.
- Shell syntax, mocked scheduling and complete serial dry-run passed. No actual allocation or submission occurred. Cloud CPU install/start instructions saved in configuration draft; publication and new-task restoration remain untested.

## Executed development stage records

- Executed `validate-config`; config `b83b9ea1bc2f3bcfcc049b633a7f4dfe020bb439eba746613ca2887267dfde15`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `results/generated/schema/stage.json`; status COMPLETED, 0.848 seconds.

- Executed `audit`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `results/generated/smoke-audit/stage.json`; status COMPLETED, 4.688 seconds.

- Executed `generate`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `results/generated/smoke-generate/stage.json`; status COMPLETED, 4.005 seconds.

- Executed `train`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `results/generated/smoke-train/stage.json`; status COMPLETED, 1.620 seconds.

- Executed `audit`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-smoke/audit/stage.json`; status COMPLETED, 6.011 seconds.

- Executed `generate`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-smoke/generate/stage.json`; status COMPLETED, 3.983 seconds.

- Executed `train`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-smoke/train/stage.json`; status COMPLETED, 1.516 seconds.

- Executed `evaluate`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-smoke/evaluate/stage.json`; status COMPLETED, 3.836 seconds.

- Executed `benchmark`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-smoke/benchmark/stage.json`; status COMPLETED, 5.388 seconds.

- Executed `compare`; config `8a43808d5c73dd8bdd97c679698ce5e33d7edac0a8216b9c77717fd839601667`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `results/test-work/3cb69f9743cc4b88b0a7abcd87eb900e/cli-comparison/stage.json`; status COMPLETED, 5.591 seconds.

- Executed `compare`; config `75f81362a940a1a66a2a77494097ee376dbefa7b7f11ea5411cd9a90289b8c7d`; commit `6ee03cd3298f6de94da546ffa3f5ee2f8da23a55`; report `runs/final-controls/stage.json`; status COMPLETED, 12.496 seconds.

- Final stricter initial-free memory policy reviewed and covered by a new CPU mock regression; 10 configuration/runtime tests passed. The complete evidence comprises 93 unique CPU tests and four unrun GPU tests.

- Pre-publication packaging review corrected an unanchored generated-data ignore rule that excluded `tdn/data`. Clean staged export imported all37 TDN modules and completed dataset generation, two optimizer steps and evaluation without ignored working files. The archive was rebuilt with the required data package.
