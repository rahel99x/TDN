# Native Windows runbook

This workflow is for a standalone Windows desktop with an RTX 4090 (24 GB dedicated VRAM) and 128 GB host RAM. It runs sequentially without the CARC queue. CARC account `anakano_81` and user `aadaniel` apply only to cluster jobs; desktop processes are local.

## 1. Extract and open PowerShell

Extract the complete ZIP to a writable project directory, such as `C:\Users\YourName\projects\TDN`. Do not execute inside the ZIP viewer. Open Windows PowerShell 5.1 or PowerShell 7 in the extracted directory containing `pyproject.toml`:

```powershell
Set-Location 'C:\Users\YourName\projects\TDN'
Get-ChildItem .\pyproject.toml
py -0p
```

Replace the example path with yours. Use standalone **64-bit Python 3.11 or 3.12**, with Conda deactivated. The Python launcher is optional when you provide an explicit interpreter path. No Git checkout is required to run this source archive.

Commands below use Windows' `powershell.exe`. `-ExecutionPolicy Bypass` applies to that child PowerShell process; it makes no persistent execution-policy change. You may instead invoke the scripts from an already permitted PowerShell session. PowerShell 7 can use `pwsh -NoProfile -File` with the same script arguments.

## 2. Inspect Python and the actual NVIDIA driver

Doctor performs discovery before a venv or PyTorch installation is present:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Doctor.ps1 -CudaDevice 0
```

If automatic Python discovery cannot find your standalone installation, give its full path:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Doctor.ps1 -PythonExecutable 'C:\Path\To\Python312\python.exe' -CudaDevice 0
```

Read the detected Python version/path, NVIDIA driver and GPU memory. The report is saved as `.runtime/desktop-doctor.json`. CUDA toolkit or `nvcc` availability alone does not determine whether the driver can run a PyTorch wheel. The NVIDIA driver supplies GPU access; a compatible PyTorch wheel supplies the CUDA runtime used by this project.

Your 4090 has 24 GB of dedicated VRAM. Windows may also display shared GPU memory drawn from the 128 GB host RAM. Shared memory is slower host memory and does not raise the project's 24 GB dedicated-memory budget.

## 3. Create the project venv

Choose one installation path. The setup script creates `.venv`, installs pinned dependencies, installs this source project and checks the environment. It invokes the venv Python directly, so activation is optional.

### RTX 4090 CUDA installation

An official Windows PyTorch **2.10.0 / CUDA 12.6** wheel is available for the supported Python versions. Use this example after confirming a compatible installed driver. Windows NVIDIA driver **560.76 or newer** meets the conservative CUDA 12.6 GA driver requirement; the subsequent desktop GPU preflight and tests must also verify that PyTorch can actually use your GPU.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Setup.ps1 -TorchVersion 2.10.0 -TorchIndex 'https://download.pytorch.org/whl/cu126'
```

If Doctor reports an older driver, consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and [NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) before choosing another official wheel or updating your driver. Driver installation is a separate user-controlled action. Use the CPU path below to get the bounded pipeline running while resolving CUDA compatibility.

### Explicit CPU installation

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Setup.ps1 -TorchVersion 2.10.0 -TorchIndex 'https://download.pytorch.org/whl/cpu'
```

Add `-PythonExecutable 'C:\Path\To\Python312\python.exe'` to either setup command if needed. An existing venv must match the selected base interpreter. A `.venv` created on Linux cannot be used on Windows.

Setup records `.runtime/windows-environment.json` and `.runtime/windows-environment-freeze.txt`. Keep those files with your run evidence.

Do not request CUDA mode with a CPU-only PyTorch installation. CUDA failure is reported explicitly; the launcher does not silently substitute CPU execution.

## 4. Preview and run the tiny sequential smoke

The desktop smoke uses an 8×8 grid, four training parents, two validation parents and two diagnostic parents. Its configured training limit is 12 optimizer steps. It keeps eager FP32 state/network computations, FP64 teachers and TF32 disabled. Triton and `torch.compile` are unnecessary for this native Windows workflow.

Pick a fresh run identifier and inspect the plan before execution:

```powershell
$tdnRunId = 'desktop-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Run.ps1 -Device cuda -CudaDevice 0 -Config configs/desktop-smoke.yaml -MaxSteps 12 -RunId $tdnRunId -DryRun
```

With the CUDA-enabled installation, execute the same plan:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Run.ps1 -Device cuda -CudaDevice 0 -Config configs/desktop-smoke.yaml -MaxSteps 12 -RunId $tdnRunId
```

For the CPU installation, replace `-Device cuda` with `-Device cpu` in both commands. CPU mode is the launcher's default, but selecting it explicitly makes the measured device clear. Use a new run ID if you already completed that identifier.

The launcher executes stages serially and stops at the first failure:

```text
CPU-compatible tests
→ numerical/scientific audit
→ immutable dataset generation
→ desktop GPU tests, when CUDA is selected
→ eager desktop calibration
→ 12-step training
→ independent diagnostic evaluation
→ matched-tolerance benchmark
```

Numerical audits and immutable teacher-dataset generation run on CPU in both modes. With CUDA selected, desktop GPU tests, calibration, training, evaluation and benchmarking use the selected GPU.

The native Windows test selection excludes Linux/Slurm launcher tests and the A100-allocation GPU tests. Desktop GPU tests are a separate device-specific validation. A skipped A100 test does not validate the 4090 or a CARC allocation.

The smoke's permissive `require_headroom: false` is an existing bounded development setting. It allows implementation diagnostics even when scientific headroom is absent. Keep `require_headroom: true` for a headroom-required pilot.

## 5. Find reports and checkpoints

The source directory is the project root. All project caches and temporary files are rooted there, including dependency installation caches. New runs are kept under `runs/<RunId>`; source and historical `results/` records remain available for provenance.

```powershell
$tdnRun = Join-Path (Get-Location) ('runs\' + $tdnRunId)
Get-ChildItem $tdnRun
Get-Content (Join-Path $tdnRun 'audit\audit.json')
Get-Content (Join-Path $tdnRun 'calibrate\calibration.json')
Get-Content (Join-Path $tdnRun 'train\training_result.json')
Get-Content (Join-Path $tdnRun 'evaluate\evaluation.json')
Get-Content (Join-Path $tdnRun 'benchmark\benchmark.json')
```

The dataset is `runs/<RunId>/dataset`. Training checkpoints are under `runs/<RunId>/train/checkpoints`. Evaluation and benchmarking use the validation-selected `best.pt`. They fail if no feasible validation checkpoint exists. `last.pt` is the training-resume checkpoint, not a replacement for validation-based model selection.

Each executed numerical stage writes `stage.json` with status and software/device metadata. Inspect that file and the retained logs if the launcher stops. A `COMPLETED` marker belongs to a successfully completed stage; do not remove it to overwrite measurements.

The pipeline summary is `runs/<RunId>/pipeline.json`, and per-stage terminal transcripts are `runs/<RunId>/logs/<stage>.log`.

## 6. Resume an interrupted run

Keep the same source, configuration, dataset, software and device. Reuse the original run identifier only with `-Resume`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Run.ps1 -Device cuda -CudaDevice 0 -Config configs/desktop-smoke.yaml -MaxSteps 12 -RunId $tdnRunId -Resume
```

Use `-Device cpu` if the original run was CPU. The launcher checks prerequisite fingerprints, preserves completed stages and resumes unfinished training from its saved checkpoint. Do not change memory limits or precision and then reuse the same run as if the resume were identical. For a revised configuration, record the reason and start a new run.

An exit code of 75 with `PAUSED_NEEDS_RESUME` indicates a saved checkpoint and deliberate stop. Closing the terminal or killing a process can lose work since its last checkpoint. Exact bitwise continuation is established for the deterministic CPU reference path on the same software/device/thread environment; a 4090 resume requires its own measured verification.

## 7. Resource and scientific limits

The desktop configuration uses an **18 GiB absolute soft VRAM cap**, a **0.75 fraction of available device memory**, and a **0.90 hard memory fraction**. Free memory can be lower because Windows, displays and other applications use the same GPU. The limits bound PyTorch-managed work; they cannot reserve memory against unrelated processes. Leave the conservative limits intact for the first smoke and avoid concurrent heavy GPU jobs.

FP64 reference work is intentionally retained. Audits and dataset generation are CPU work; GPU calibration and diagnostics also exercise numerical/reference operations. The RTX 4090's FP64 throughput differs substantially from an A100's, and a tiny workload may be dominated by launch overhead. Measure timing on the actual device; do not transfer cloud CPU or CARC A100 timings to the desktop.

Historical CPU screens in `results/gates.json` passed G1, empirically screened G3, and failed **G2 and G4**. No TDN efficiency advantage has been established. Desktop calibration and a successful 12-step smoke measure implementation behavior on your machine; they do not settle these scientific gates. Larger or confirmatory runs need a valid scientific design and passed required gates.

The original objective and protocol are preserved in [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md); [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md) describes implemented scope. Start a local chat using the prompt in [WINDOWS_START_HERE.md](../WINDOWS_START_HERE.md). The [CARC runbook](CARC.md) remains the separate procedure for charged cluster allocations.
