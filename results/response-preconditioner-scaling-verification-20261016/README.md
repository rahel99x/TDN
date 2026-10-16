# Phase 1 scaling implementation verification — 16 October 2026

**CPU implementation verified; native study pending.** The required verifier
passed exactly **388/388** cases, with no failures, errors, skips, duplicate
or missing cases. Eight additional Raven compatibility cases passed against
`c34f0892d7bddc685c5af10489ecb114fd6adab6`; the 13 mandatory launcher cases
were deselected in that separate run, so these eight do not overlap the 388.
Compatibility uses captured runtime/scheduler fixtures, not actual Slurm.

The source fingerprint is
`53758e02afc69d1e665caed9f971ae8735ba8830f5c7487ec904f6395101e1bc`.
Before/after snapshots agree. The complete verifier, collection and sealing
took 59.55 seconds in this cloud CPU environment. No new dependencies or
environment configuration were needed.

## Verified behavior

Independent physical-parent generation and hashes; balanced nested cohorts;
source prefixes and split isolation; original parent-balanced positive-ridge
objective and fixed model; validation-only selection; all-cell seals before
protected truth; independent/refined references; fresh inference/design parity;
physical metric replay from saved predictions; explicit unresolved, failed and
missing denominators; per-parent and family reporting; shared label-cost
accounting; partial timing persistence; frozen work budgets; native admission,
same-attempt verification release, Raven reporting, and portable split exports.

The actual numerical fixture completed **48 accepted U/G query pairs**, four
selected cells, **288 measured metric rows**, and **12 timing bundles**, with no
failed/unresolved reference or metric. It uses four training, two validation
and two diagnostic parents, one nonzero-start interval, and 2/4 parent and
source prefixes. Its files are retained under `reduced-fixture/`. The fixture
protocol was exported afterward; its hash matches the prospective journal.
NPZ files load with `allow_pickle=False`.

These seeds are prefixes of the production development seeds. The affected
fields are therefore inspected development data, not untouched confirmation.
The fixture establishes implementation behavior, not scientific benefit or a
learning law. Historical smoke fixtures also execute through inherited tests;
they are not additional native campaigns.

The full P1 bank was **not executed here**. Its frozen `study-protocol.json`
specifies 32 training, eight validation and eight diagnostic parents, nine
8/16/32-by-2/4/8 fits, 2,688 query pairs, 16,128 metric rows and 1,008 timing
bundles within 4 CPUs, 8 GiB and 20 minutes. A shared 900-second scientific
deadline can stop acquisition early; incomplete evidence must be reviewed,
never silently expanded. This work changes neither the old R7 evidence nor
the historical B STOP, and does not promote an encoder, GPU or speedup claim.

## Retained files

- `receipt.json`: exact scope, commands, environment and counts.
- `verification/`: original exact roster, JUnit, logs, source/software/protocol
  identities and manifest, excluding temporary pytest trees.
- `raven-compatibility.xml` and log: the eight additional compatibility cases.
- `reduced-fixture/`: parents, queries, fresh predictions, metrics, selections,
  reference diagnostics, complete timing, costs and access journal.
- `manifest.json`: hashes of every retained file except itself.

Use the [Phase 1 runbook and term guide](../../docs/RESPONSE_PRECONDITIONER_SCALING_PHASE1.md)
for the terminal-safe native launch and evidence-export commands. Review the
native evidence before selecting a next phase or enlarging the cohort.
