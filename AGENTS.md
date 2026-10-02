# Working in this repository

Read `WINDOWS_START_HERE.md` and `docs/WINDOWS.md` for native Windows use. Read `docs/CARC.md` before changing or running the separate cluster workflow. `docs/IMPLEMENTATION_SPEC.md` is the original uploaded research specification; distinguish its environment examples and requested scientific extensions from implemented and measured behavior in `docs/IMPLEMENTATION_MAP.md` and the source.

## Desktop execution

- The user authorized a native Windows adaptation for an RTX 4090, 24 GB dedicated VRAM and 128 GB host RAM. Shared/host RAM is not additional dedicated VRAM.
- Use standalone 64-bit Python 3.11 or 3.12 in the repository's `.venv`, never Conda. Invoke `.venv\Scripts\python.exe` on Windows; activation is optional.
- Use `scripts/windows/Doctor.ps1`, `Setup.ps1` and `Run.ps1`. They set `TDN_EXECUTION_MODE=desktop` explicitly and root all generated files in this extracted checkout. Do not invent Slurm environment variables to pass a runtime check.
- Run doctor before choosing a CUDA wheel. Select the installed-driver-compatible official PyTorch distribution; do not infer wheel compatibility from CUDA toolkit availability or the GPU name. Do not install or alter GPU drivers automatically.
- Keep temporary files, caches, dependency caches, datasets, logs, checkpoints and reports within the repository. Do not redirect them to system temp, `/tmp`, `/scratch1` or external drives. Do not bring in a Linux venv from an archive.
- Start with `configs/desktop-smoke.yaml`, a fresh run identifier and a 12-step maximum. Keep eager FP32, TF32 off and FP64 teachers. Native Windows does not require Triton or `torch.compile`.
- Use CPU mode explicitly if CUDA is unavailable. Do not report CPU results as GPU results or silently downgrade a requested CUDA run.
- Preserve completed runs. Resume interrupted runs using the same configuration, dataset, source and software/device environment; use the launcher resume checks. Changing code/configuration invalidates prerequisite fingerprints and may invalidate exact resume.
- GPU VRAM policy in the desktop smoke uses an 18 GiB absolute soft cap, a 0.75 fraction of available memory and a 0.90 hard fraction. Respect lower available memory caused by the display or other programs. Do not increase those limits to hide an out-of-memory failure.

## Scientific scope and evidence

The implemented starting problem is periodic logistic reaction–diffusion with an exact discrete diffusion split, FP64 coupled teachers, six learned controls and classical accuracy/cost controls. Retain parent-disjoint splits, validation-only checkpoint selection, independent diagnostics and matched-tolerance comparisons.

Historical CPU screening in `results/gates.json` failed G2 and G4. The tiny smoke is a bounded development diagnostic with its declared permissive headroom setting. It does not authorize disabling `require_headroom` in a pilot or expanding into a confirmatory campaign. Optional Gray–Scott, advection, distributed HALO, KAN and oscillatory branches require separate implementation and validation.

Write factual reports: record actual commands, execution mode, device, versions and failures. Keep historical cloud CPU, actual desktop CPU/GPU and actual CARC A100 evidence distinguishable. A successful pipeline or unit test suite does not demonstrate a TDN efficiency advantage. GPU results must come from actual GPU execution; no mock or skipped test counts as GPU validation.

## CARC execution remains separate

The cluster workflow requires user `aadaniel`, charging account `anakano_81`, root `/home1/aadaniel/projects/TDN`, a project `.venv` and real Slurm allocations for substantial compute. Preserve those checks. Desktop mode must not bypass them on CARC or masquerade as an A100 allocation. All CARC project/cache/temp/run files remain under its fixed root, with no `/tmp` or `/scratch1` use.

When editing, run tests appropriate to the change. Windows launcher syntax and actual NVIDIA execution need native-machine validation; Linux checks cannot substantiate those claims. Never delete results or overwrite a completed run to make a check pass.
