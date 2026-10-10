# Compact, exact-byte evidence uploads

Use this for a finished `c5c9a85` portfolio run without rerunning the science or
first building its full `.tar.gz`. The exporter reads the original run directory
and preserves it. It does not use today's scientific protocol to reinterpret an
old experiment, import Torch, load model objects, or change historical seals.

```bash
cd /home/rahel/TDN
git pull --ff-only origin main
bash scripts/compact_review.sh plan latest
bash scripts/fedora_portfolio.sh compact latest
```

`latest` means `runs/.fedora-portfolio-latest.json`. To avoid selecting a newer
campaign, replace it with the exact portfolio run ID or its original absolute
directory. Other experiment directories can be selected explicitly under
`runs/`; automatic recovery ancestry is currently implemented for portfolio
workflows only. Input is a directory, not a previously collected archive.

For the completed campaign discussed in the upload, pin its identity rather
than following a `latest` pointer which a later rerun may have moved:

```bash
bash scripts/fedora_portfolio.sh compact fedora-portfolio-20261009T183307014599Z
```

The default is now **gzip level 1** for fast packing. If the old `--preset 9`
command was taking too long, Ctrl-C it and rerun without that flag. An already
finished XZ package remains valid and does not need to be regenerated.

The command prints a fresh directory under `runs/review-exports/`, containing
`index.json`, a human-readable `manifest.json`, and either one `.tar.gz` (`.tar.xz`
when XZ is explicitly selected) or
numbered parts. **Upload `index.json` and every archive/part it lists.** The
manifest is also inside the archive, so uploading the external copy is optional.
The default part limit is 24 MiB, below the attachment limit. Parts are ordered
in the index; none can be omitted. Do not send an older full archive as well.

This is a CPU file export, not a scientific experiment. It needs no GPU or new
Slurm job and uses the project's venv. Both codecs use one CPU thread; gzip has
small compression buffers. XZ-1 uses roughly 9 MiB of compressor memory, XZ-6
roughly 100 MiB, and XZ-9 roughly 700 MiB, plus inventory and I/O buffers. Higher
XZ settings can spend many minutes on multi-GB evidence for smaller uploads.
Hashing reads every source file, including omitted payloads. Progress reports
hashing, unique compression input, compressed output bytes and verification,
with elapsed time and average throughput approximately every five seconds while
processing bytes. Percentages are per phase, not an overall completion estimate;
compression input can finish before stream finalization and verification.
Source measurements must be quiescent: run after all jobs finish. A
file-set, content, metadata, or recovery-closure change during export aborts
without publishing the package. Failed or interrupted runs are welcome once
their jobs are terminal; their failures and partial records are retained.

## What is represented

| Content | Default `review` | Optional `full` |
|---|---|---|
| All original JSON/JSONL/CSV, losses, endpoint records, paired timing rounds, logs, failures, text reports | Exact bytes | Exact bytes |
| Protocols, IDs, parameter reports, selected-model catalogs, source/software identities, seals, saved scheduler accounting | Exact bytes | Exact bytes |
| Compressed chart data, including learning-range data | Exact bytes | Exact bytes |
| Binary `.pt`, `.pth`, `.safetensors` checkpoints and `.npy`/`.npz` arrays | Hash, size, path, omission reason | Exact bytes |
| Rendered PNG/PDF/SVG/JPEG/WebP figures | Hash, size, path, omission reason | Exact bytes |
| Whole portfolio recovery ancestors, original attempts and their costs | Same rules as the primary run | Same rules as the primary run |
| Disposable pytest work/cache and `__pycache__` directories | Excluded, directories listed | Excluded, directories listed |

Unknown file types are retained. No table is filtered by outcome. Nothing is
rounded, downsampled, smoothed, quantized, text-normalized, or summarized in place
of the raw records. In particular, sealed CSV CRLF line endings remain intact.
The size reduction has three separately reported sources: explicitly omitted
replay/rendered payloads, identical-file deduplication, and stream compression.
Identical retained files are stored once with standard tar hardlinks; their
original paths remain in the archive. Related table/catalog filenames are placed
together to reuse the compression dictionary across stages. The supplied restore writes independent
files, so editing a restored copy will not modify a deduplicated sibling.

Review is sufficient for the recorded quantitative analysis, but cannot rerun
inference, independently recompute errors from fields, inspect omitted spatial
arrays, or resume training without the original binary payloads. It is **not a
complete replay or independent scientific-seal verification**. Keep the original
run and existing full archive. The original science manifests are preserved;
their references to omitted files are intentional and enumerated, not repaired
or presented as verified. Both modes preserve file contents, recorded modes and
mtimes in the manifest; they do not preserve filesystem ownership, empty
directories, or filesystem-specific metadata. Symlinks and special files are
rejected. Historical absolute paths in scientific metadata stay unchanged.

Accounting reflects saved snapshots. The exporter does not poll Slurm, infer
missing costs, or label an old partial accounting record as final. The exporter's
own source hash is separate from the run's original source/version records.

## Verify and restore

Use the actual index path printed by the exporter:

```bash
bash scripts/compact_review.sh verify runs/review-exports/EXPORTED-DIRECTORY/index.json
bash scripts/compact_review.sh unpack runs/review-exports/EXPORTED-DIRECTORY/index.json \
  --output runs/restored-portfolio-review
```

Verification checks each part's length/SHA-256, the concatenated stream, the
inventory hash, and the SHA-256 of every included file. It rejects missing,
duplicate, unexpected or unsafe members and invalid deduplication links. This
establishes transport integrity, not a scientific result or an authenticated
signature from a third party. Restore refuses an existing output directory,
checks everything before publication and does not need a concatenated archive.
Files appear under `runs/restored-portfolio-review/runs/<original-run-id>/`.

Compression streams straight into parts; it creates neither a full staging
copy of the run nor a second concatenated archive. An index is published only
after verification. Temporary export/restore work remains in a unique sibling
directory under the project and is removed on ordinary errors or Ctrl-C. Ctrl-C
returns exit code 130 with a concise cancellation message, skips finishing the
compression stream, and publishes no incomplete package. A machine crash
or SIGKILL can leave a `.partial-*` directory; it is not a finished upload. Source
runs are never deleted or changed. New invocations create fresh output names.

## Options

```bash
# Keep all binary data, models and figures too (larger, exact file contents).
bash scripts/fedora_portfolio.sh compact RUN-ID --mode full

# Explicit fast default, no extra packages required.
bash scripts/fedora_portfolio.sh compact RUN-ID --compression gzip --gzip-level 1

# Smaller uploads at the expense of compression time; start at XZ-1.
bash scripts/fedora_portfolio.sh compact RUN-ID --compression xz --preset 1

# Explicit directory and smaller attachment pieces.
bash scripts/compact_review.sh pack RUN-ID --part-mib 16 \
  --output runs/review-exports/my-finished-review
```

The old `fedora_portfolio.sh collect` remains available for the original full
gzip collector. Compact export does not change its format or historical source
guards. An explicit legacy `--preset N` still selects XZ. `--compression xz`
without a preset now means XZ-1; gzip and XZ level flags cannot be mixed.
Old compact XZ indices without a compression field remain readable and
verifiable. Do not pull new code while an active scientific workflow still requires
its original source for execution or recovery.

## Measured compression check

On the existing **CPU smoke** `portfolio-local-review-v2`, the original files
total 138,435,823 bytes. The legacy gzip representation is 30,692,960 bytes.
The original XZ-6 review retains 1,188 files with 124,533,755 logical bytes and
produces a **7,771,872-byte** verified upload (74.7% smaller than legacy gzip),
including the inventory and deduplication metadata. Export and verification took
23.2 seconds in this cloud CPU check. The 55 targeted transport/workflow tests
passed, including exact restore, corruption, recovery ancestry, source mutation
and unsafe-path cases.
For the speed fix, a read-only comparison used that same exact review selection,
filename ordering and deduplication (116,843,859 unique bytes), writing to a
counting sink rather than disk:

| Codec | Compressed bytes | Compression-only seconds |
|---|---:|---:|
| gzip-1 (new default) | 23,103,431 | 0.66 |
| gzip-3 | 21,748,509 | 1.02 |
| XZ-1 | 14,223,676 | 3.01 |
| XZ-3 | 9,965,504 | 9.96 |

These compression-only times exclude verification and filesystem output. They
show the size/time tradeoff, not a timing forecast for a larger Fedora run.
The updated real-file gzip-1 export, including hashing and complete archive/file
verification, took **1.72 seconds** and produced **23,103,433 bytes** on this
saved smoke. Its greater size is the explicit price of faster packing.
The updated exporter/workflow regression suite passed **76 tests**, including
legacy XZ restoration, gzip, cancellation and progress inside large files.
Every codec retains the same observations and integrity checks. Already
compressed or high-entropy
data have limited lossless compressibility; the tool reports actual sizes
rather than promising a fixed ratio.
