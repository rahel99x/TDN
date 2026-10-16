# P1 packaging and export repair — 16 October 2026

**CPU verified; native export retry pending.** The user reported
`Campaign reporting requires a bounded regular file` while exporting
`launch-0e144975eb4b433492fe55d75d209ef2-a1`. That native attempt has not yet
been uploaded, so its exact rejected file and scientific completion status
remain unverified.

The implementation has a reproducible two-stage size mismatch. A full P1
bulk bank has 7,680 files, exceeding the packing inventory writer's original
1 MiB limit. The ZIP is closed before that inventory write, leaving an archive
without `packing.json` when the write fails. The exporter then includes that
archive and can reject it at its original 64 MiB per-file limit. Its own final
export inventory also used the insufficient 1 MiB writer limit.

## Repair and checks

The repair streams evidence in 1 MiB chunks and gives packing/export
inventories explicit 16 MiB bounds. It retains the existing 512 MiB packing
and 1 GiB total review budgets, regular-file and confinement checks, terminal
requirements, hashes and mutation detection. The shared small-metadata default
remains 1 MiB; rejected reads/writes now identify their path, size and limit.
An orphan archive remains original evidence and is included during export.
No original status, scientific arrays, logs or partial bundle is rewritten.

All **404/404 required CPU checks** passed, with no failures, errors, skips,
duplicates or missing cases. This adds 16 export checks to the original 388.
The separate **51/51 Raven compatibility checks** passed against
`c34f0892d7bddc685c5af10489ecb114fd6adab6`; 13 overlap the required roster.
These compatibility checks include the shared diagnostic coordinator helper.
The required verifier took 93.06 seconds; compatibility took 68.04 seconds.

Actual regression fixtures cover:

- A 68,157,440-byte (65 MiB) orphan file, preserved in a multipart export,
  with per-part, reassembled-archive and extracted-content hash checks.
- 7,680 tiny evidence files producing a 1,329,024-byte packing inventory and
  a 1,437,421-byte export inventory, both beyond the previous writer bound.
- Bounded copying/hashing; aggregate caps; growing, replaced and same-size
  changed files; symlinks, FIFOs, directories and paths outside the run;
  both native and Raven terminal-state requirements.

These are transport fixtures, not new scientific observations. The mandatory
roster also exercises inherited numerical fixtures. Model, loss, reference
policy, data populations and scientific work budgets are unchanged. The
verification roster changes for future prepared attempts; historical export
continues to require no match with current source or protocol identity.

## Retained evidence and recovery

`verification/` retains the exact required JUnit roster, logs, source/software
identities and original artifact manifest. `raven-compatibility.xml` and its
log retain the integration cases. `size-reproduction.json` records the old
bound failures on the transport fixtures. `repair-source.json` includes hashes
for the shared helper and modified files in addition to the broader native
source identity. `receipt.json` records commands and scope; `manifest.json`
hashes all retained files except itself. Temporary fixture trees are not copied.

Update the checkout after its active jobs finish, then use the existing
[export command](../../docs/RESPONSE_PRECONDITIONER_SCALING_PHASE1.md) for the
original attempt. No scientific resubmission is needed for export recovery.
Review the uploaded evidence before deciding whether any scientific work
remains incomplete or whether another phase is eligible.
