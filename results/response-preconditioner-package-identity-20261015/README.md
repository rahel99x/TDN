# Native preparation package-discovery repair — 15 October 2026

The user reported `Duplicate installed distribution: pip` during native launch
contract preparation. The original check rejected every repeated normalized
package name. Python's metadata discovery can enumerate one installation
twice through repeated search paths or Fedora-style `lib`/`lib64` aliases.
That mechanism was reproduced locally; the user's exact package paths have
not been inspected.

`installed_packages()` now merges discoveries only when normalized name,
version and canonical existing metadata directory all match. Distinct metadata
directories remain errors even with equal versions. Inconsistent versions and
unverifiable duplicate locations also fail, with the conflicting versions and
locations printed. The package-name/version software snapshot schema remains
unchanged. No packages were removed or reinstalled.

Verification:

- **184 mandatory CPU cases passed**, with no failures, errors or skips, via
  the actual `response_preconditioner_local.sh verify` wrapper. This includes
  six new regressions for package discovery. See `verification.xml`,
  `summary.json`, `checks.json` and `source.json`.
- **33 launch cases passed**, including the six new regressions and trusted
  Raven compatibility checks. They overlap the mandatory suite and must not
  be added as distinct tests. See `launch-tests.xml`.
- Real Python 3.12.14 venv inventory: repeating the actual pip site-packages
  search path produces two pip discoveries at one canonical metadata location;
  the repaired software snapshot equals its original 30-package inventory.
  See `actual-venv-alias.json`.
- Actual Python 3.13.5 stdlib fixture: a lib/lib64 alias is deduplicated, while
  separate same-version metadata remains rejected. See `python313.json`.

The code uses the supported CPython `PathDistribution._path` metadata location
behind a guarded check. The public `locate_file('')` returns a shared install
root and cannot distinguish competing metadata there; `files` can be absent
without RECORD. Unknown metadata implementations cannot justify merging a
duplicate name.

The numerical methods, scientific thresholds and historical A PASS/B STOP
remain unchanged. No native Slurm jobs, GPU work or scientific campaigns ran.
The local verifier's bounded numerical test fixtures remain implementation
evidence. A fresh native contract is required after updating because existing
contracts bind the original source. The actual Fedora retry remains pending.

See the [runbook](../../docs/RESPONSE_PRECONDITIONER_RUNBOOK.md) and
[standing research program](../../docs/TDN_RESEARCH_PROGRAM.md).
