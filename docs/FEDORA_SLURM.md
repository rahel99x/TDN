# Fedora desktop with local Slurm

This workflow runs the complete premix experiment on the desktop's own Slurm
scheduler. It keeps the numerical protocols, reference tolerances, five model
families, paired seeds, and predefined favorable/typical/adverse regimes from
the CARC experiment. Hardware and scheduler evidence identify these results as
`desktop-slurm`; they do not become CARC A100 measurements.

The intended machine is `/home/rahel/TDN`, standalone **Python 3.13.13**, an
AMD Ryzen 7 7800X3D (8 cores/16 threads), 128 GB host RAM, and one NVIDIA RTX
4090 with 24 GB dedicated VRAM. Python 3.11 and 3.12 are also accepted by the
bootstrap. Host/shared memory does not increase the GPU budget.

The desktop's reported Slurm partition is **`local`** (default, UP), with one
node, 16 scheduled CPUs, 110000 MB scheduled memory, `gpu:1`, and a seven-day
partition limit. The workflow requests only 4 CPUs, 16 GiB and 30 minutes per
job. Use `local` for both CPU and GPU stages; leave the account unset unless
your local scheduler explicitly requires one.

Slurm and the NVIDIA driver must already work. These scripts do not install
drivers, change Slurm daemons, alter operating-system packages, or modify the
Tower application. There is no Conda requirement or environment.

## First setup

Use the normal desktop shell outside an allocation. A Python venv transferred
from CARC or Windows is not portable: preserve it separately and create the
desktop's own `.venv` with this setup. The script refuses an existing venv tied
to a different base interpreter instead of overwriting it.

```bash
cd /home/rahel/TDN
git pull --ff-only origin main

# Read-only inventory: Python, Slurm partitions/GRES, account associations,
# NVIDIA GPU, dedicated memory and installed driver.
bash scripts/fedora_slurm.sh doctor

# Configure the observed local partition and its one GPU.
bash scripts/fedora_slurm.sh configure --python python3.13 \
  --cpu-partition local --gpu-partition local --gpu-gres gpu:1

# Install exact dependencies into this checkout's .venv.
bash scripts/fedora_slurm.sh setup
```

`configure` uses the current checkout path and logged-in user. If partition
flags are omitted, it selects a
partition only when the observed default or single eligible partition makes
the choice unambiguous; the GPU partition must advertise GPU GRES. If your
desktop has several possible partitions, pass their actual names explicitly:

```bash
bash scripts/fedora_slurm.sh configure --python python3.13 \
  --cpu-partition YOUR_CPU_PARTITION --gpu-partition YOUR_GPU_PARTITION \
  --gpu-gres gpu:1
```

Replace the uppercase partition placeholders with names printed by `doctor`.
Typed resources such as `--gpu-gres gpu:rtx4090:1` are supported when that exact
GPU type is configured in your Slurm GRES. Exactly one GPU is requested. If
Slurm enforces an account, append `--account YOUR_LOCAL_ACCOUNT`; otherwise
leave it unset. **No CARC account is inherited or charged by this launcher.**

The profile lives in `.tdn/fedora-slurm.json`. Configuration also creates
`.tower/fedora-slurm.json` for Tower's `desktop-slurm` profile. Both files are
local and ignored by Git. Inspect them before using `configure --replace` to
change an existing selection. Configuration and installation refuse to modify
the checkout's settings or venv while an allocated worker holds the shared
venv lock.

Setup defaults to `torch==2.10.0+cu126` from the official
`https://download.pytorch.org/whl/cu126` index. It reads the installed driver
using `nvidia-smi`, then requires a conservative Linux driver baseline of
**560.28.03** for CUDA 12.6. This is the CUDA 12.6 release baseline, rather than
assuming every driver satisfying CUDA minor-version compatibility supports
all features. See NVIDIA's [CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)
and [minor-version compatibility restrictions](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
No local CUDA toolkit is needed for the official PyTorch wheel. Driver changes
are not performed automatically.

An explicit CUDA selection can be supplied to `configure` with both
`--torch-version` (including its exact `+cuNNN` suffix) and
`--torch-wheel-index`. The bootstrap also recognizes conservative CUDA 12.8
and 13.0 Linux driver baselines of 570.26 and 580.65.06 respectively; this is
not a claim that every PyTorch release publishes every wheel. Package
installation must succeed and the imported PyTorch version/runtime must
match the selected wheel. CUDA execution is then tested inside the allocated
GPU job.

All caches and temporary files stay under `.runtime/` in this checkout. The
setup records installed versions in `.tdn/fedora-environment.json` and the
dependency freeze in `.tdn/fedora-environment-freeze.txt`. It does not overwrite
the historical CARC environment record. Setup and running jobs share the
same `.cache/carc-phase.lock` lifecycle lock.

## Run the experiment

The follow-up bounded physical gate and mean/spatial comparison uses the
separate [consistency runbook](CONSISTENCY.md) and
`scripts/fedora_consistency.sh`. The commands below retain the original premix
accuracy/scaling experiment.

Start with the smoke workflow to verify the native desktop driver, Slurm GRES
assignment, interpreter and launch path together:

```bash
cd /home/rahel/TDN
bash scripts/fedora_slurm.sh plan --smoke
bash scripts/fedora_slurm.sh run --smoke
bash scripts/fedora_slurm.sh status latest
bash scripts/fedora_slurm.sh logs latest --lines 200
bash scripts/fedora_slurm.sh paths latest
```

After the smoke workflow completes successfully, submit the full experiment:

```bash
bash scripts/fedora_slurm.sh plan
bash scripts/fedora_slurm.sh run
bash scripts/fedora_slurm.sh status latest
bash scripts/fedora_slurm.sh logs latest --lines 200
bash scripts/fedora_slurm.sh paths latest

# After all stages finish, collect the complete review archive.
bash scripts/fedora_slurm.sh collect latest
```

The four stages run in order: **accuracy → scaling → prepare → neural**. The
first three use CPU allocations. The neural stage requires one actual
Slurm-assigned RTX 4090. Dependencies use `afterok`; a failed prerequisite
prevents dependent computation. Serial CPU stages avoid placing three
simultaneous four-core workloads on the eight-core desktop.

Each job requests **4 CPUs, 16 GiB host memory and at most 30 minutes**. The
scientific stage retains its fixed computation budget. There is no pending-job
limit, no automatic retry, and no unbounded extension of a long training job.
The GPU uses an **18 GiB absolute soft cap**, further reduced to **75% of
available dedicated VRAM**, with a 90% device-use hard limit. Desktop display
or other applications may reduce the usable budget. CUDA errors and insufficient
VRAM remain failures; the launcher does not silently switch to CPU execution.

CPU and GPU checks execute in actual Slurm tasks before the relevant scientific
stage. GPU tests must execute on the allocated GPU; mocked scheduler tests or
skipped CUDA tests do not establish hardware readiness. All generated datasets,
checkpoints, logs, references, Tower reports and archives remain under this
project. Neither `/tmp` nor `/scratch1` is used for project outputs or caches.

Keep the checkout and local profile unchanged while the workflow is queued or
running. Source, software, profile and prerequisite fingerprints are checked;
pulling changes or replacing the environment requires a fresh workflow. Retain
failed runs and their archives as evidence.

## Tower and interpretation

`paths` prints the exact report directories and `metrics.jsonl` paths after
workers start. The project's Tower helper recognizes the desktop Slurm
configuration:

```bash
bash scripts/tower.sh list
bash scripts/tower.sh validate latest
bash scripts/tower.sh show latest
```

Use an explicit report directory instead of `latest` when multiple historical
workflows are present. Tower itself remains unchanged; project files supply
the logs, metrics, artifacts and table contracts it consumes.

The scientific protocol and its limitations are described in [PREMIX.md](PREMIX.md).
Compare accuracy and matched-tolerance work on the same device; raw A100 and
4090 timings are different hardware measurements. Completion verifies that the
experiment ran. It does not establish an advantage over the representative
FNO baseline, reproduce an entire FNO paper, or imply performance on other PDEs.

The [implementation validation record](../results/fedora-slurm-validation.json)
records 2,101 passing cloud CPU tests, all four completed local smoke stages,
and native Tower report checks. Those checks do not substitute for the allocated
Fedora GPU smoke above.

## Common startup failures

- **Partition mismatch on the original Fedora launcher:** an older parser
  included Slurm's following `AllocNode:Sid` field in the partition value.
  Pull the current `main` and submit a fresh smoke workflow; keep the failed
  run for review. Current errors print the expected and observed partition.
- **Ambiguous partition:** pass both partition names from `doctor` explicitly.
- **No GPU GRES:** inspect the existing Slurm configuration; a GPU visible to
  `nvidia-smi` alone is insufficient for a Slurm GPU allocation.
- **Python 3.14 selected:** pass `--python python3.13` (or a supported standalone
  3.11/3.12 executable); the pinned scientific dependencies target those versions.
- **Driver below the selected CUDA baseline:** retain the doctor output and
  choose an appropriate supported installation. Setup does not change drivers.
- **Venv locked:** let active workers finish; installation cannot safely mutate
  dependencies while they execute.
- **Existing incompatible venv/profile:** preserve the old environment and
  results. Reconfigure explicitly for this checkout and create a compatible
  project venv; do not copy an environment between operating systems.
