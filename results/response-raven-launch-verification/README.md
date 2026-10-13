# Response campaign through Raven launch

The diagnostic campaign now supports Raven's actual existing-script launch
contract. It validates the captured shell/original-checkout context, retains
coordinator logs and infrastructure progress, and declares small run-local
receipt/decision/research-path outputs. Raven owns the coordinator lifecycle;
the native TDN workers retain independent allocation verification, scientific
source guards and exact child Raven records. The opt-in `response-diagnostic`
recipe requests Experiment, Evidence and Artifacts without coordinator GPU,
trace or CPU telemetry. No Raven application files were changed.

The launch was checked against trusted Raven source
`c34f0892d7bddc685c5af10489ecb114fd6adab6`. The real CLI produced a read-only
local launch plan with the documented recipe and exact application arguments.
The actual private Raven runtime executed captured campaigns in project paths
with spaces. Scientific and scheduler commands were controlled fixtures;
these checks do not establish actual Slurm or CUDA readiness.

The combined targeted regression passed **129 tests in 61.96 seconds**. It
covers launch dispatch, controller contracts, native environment isolation,
archive safety, and captured Raven execution/readers. Two final archive
race/rollback refinements then passed **24 archive tests in 0.43 seconds**:
contradictory retained live scheduler states block moves, and interrupted
restoration retains changed roots without overwriting other evidence. Python
compilation, Bash syntax and `git diff --check` also passed. Commands, scopes
and exact source hashes are saved in [verification.json](verification.json).

The tests confirm that completed infrastructure can retain a negative research
decision, infrastructure failures propagate their status, stdout/stderr are
drained before Raven finishes, final native status counts remain retained,
and mutated source/status evidence is rejected. Full scientific metrics stay
in the exact child research projects; coordinator counters never substitute
for scientific acceptance.

## Clean the cancelled campaign and restart from fresh calibration

Run these on the configured Fedora host. Stop the old foreground coordinator
with Ctrl-C if it is still polling; cancelling its Slurm jobs alone does not
necessarily stop it immediately.

```bash
cd /home/rahel/TDN
git pull --ff-only
bash scripts/response_diagnostic_archive.sh --plan
bash scripts/response_diagnostic_archive.sh
raven launch scripts/response_diagnostic_campaign.sh \
  --local --project "$PWD" --recipe response-diagnostic \
  -- --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

The archive command selects the newest diagnostic campaign receipt by default.
It checks terminal coordinator and actual scheduler records, then moves only
that campaign's native run pair, receipt directory and matching latest pointer
under `runs/archived-response-diagnostics/<fresh-id>/`. Use `--campaign TOKEN`
to fix an explicit selection. It preserves all artifacts and unrelated runs;
it neither deletes files nor cancels jobs. The archive manifest records original
paths and actual scheduler evidence. Outer Raven launch records stay in place.

`--local` applies to the lightweight submitting coordinator. The fresh campaign
still executes its numerical work in actual native Slurm allocations and needs
new verified CUDA checks and calibration. Options for the campaign follow `--`.

The cloud checks did not clean the user's machine or launch the native science.
Those actions are established by running the commands on Fedora. Read the
[archive guide](../../docs/RESPONSE_DIAGNOSTIC_ARCHIVE.md) and
[runbook](../../docs/RESPONSE_DIAGNOSTIC_RUNBOOK.md) for exact selection,
preservation behavior and Raven registration.
