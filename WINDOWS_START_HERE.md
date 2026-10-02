# TDN on your Windows desktop

This edition runs locally in native Windows PowerShell, using your RTX 4090 with **24 GB dedicated VRAM** and **128 GB host RAM**. It uses a project-local Python `.venv` and sequential processes; no Slurm allocation or lab account is involved. The 128 GB of host/shared memory does not increase the GPU's dedicated VRAM budget.

Extract the ZIP to a writable local folder, for example `C:\Users\YourName\projects\TDN`. Open PowerShell in the extracted repository folder containing this file and `pyproject.toml`. Install standalone 64-bit Python 3.11 or 3.12 if needed; leave Conda deactivated.

First inspect the machine without installing anything:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Doctor.ps1
```

Then follow [the Windows runbook](docs/WINDOWS.md) to select the wheel for your installed NVIDIA driver, create `.venv`, and run the 12-step smoke. CUDA compatibility depends on the installed driver, so the GPU model alone cannot select the wheel. A CPU installation is also supported.

The default desktop configuration is `configs/desktop-smoke.yaml`: an 8×8 problem, 12 optimizer steps, eager FP32, TF32 disabled, FP64 reference computations, and conservative VRAM limits. All temporary files, dependency caches, logs, data and checkpoints are kept under the extracted project folder.

For a new local Codex chat, open this project folder and paste:

> Continue this TDN research project on my Windows desktop: RTX 4090 with 24 GB dedicated VRAM and 128 GB host RAM. Read AGENTS.md, WINDOWS_START_HERE.md, docs/WINDOWS.md, docs/IMPLEMENTATION_SPEC.md and docs/IMPLEMENTATION_MAP.md. Follow the implemented scope and existing scientific gates. Use native PowerShell and standalone Python 3.11/3.12 with this project's .venv; keep all temporary/cache/data/run files inside this project. First run Doctor.ps1 and inspect my actual NVIDIA driver, then select a compatible official PyTorch wheel and run Setup.ps1. Preview and execute Run.ps1 with configs/desktop-smoke.yaml, Device cuda, MaxSteps 12, and a fresh RunId. Run the desktop tests and report the measured GPU results. If CUDA is unavailable, explain the blocker and use an explicitly selected CPU wheel and CPU mode for the bounded smoke. Preserve existing reports, datasets and checkpoints, and use the documented resume procedure for interrupted runs. Do not bypass require_headroom, expand to a full campaign, or claim an efficiency improvement from a successful smoke. G2 and G4 failed the historical CPU scientific screen.

The original implementation specification is included for provenance. CARC-specific user, account and `/home1` path requirements continue to apply to the cluster workflow; the user has authorized a separate Windows desktop implementation in this extracted directory.

Historical files in `results/` describe earlier cloud CPU checks. New measurements are written in `runs/`. They do not establish RTX 4090 or A100 performance until those measurements actually run on the named hardware.
