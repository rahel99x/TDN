# Native roadmap reporting review

All 10 reports passed the unchanged Tower native API, their indexed page contracts, and exact source-projection comparison. No archive code was executed and no scientific artifacts or Tower files were modified.

| Stage | Experiments | Checks | Values | Pages | Live progress records |
|---|---:|---:|---:|---:|---:|
| audit | 52 | 254 | 1,863 | 20 | 6 |
| headroom | 1,273 | 4,972 | 43,020 | 397 | 18 |
| prepare | 60 | 300 | 16,364 | 134 | 6 |
| train | 115 | 1,035 | 8,925 | 83 | 7 |
| confirm_prepare | 48 | 240 | 15,808 | 129 | 6 |
| confirm | 70,748 | 597,356 | 4,840,310 | 43,589 | 713 |
| policy | 1,111 | 10,632 | 57,932 | 556 | 17 |
| transfer | 595 | 4,176 | 33,369 | 305 | 11 |
| scaling | 147 | 729 | 5,919 | 58 | 7 |
| report | 1 | 3 | 10 | 6 | 4 |

In total: 45,277 pages in 361 page contracts; 74,150 experiment and metric records; 619,697 checks; 5,023,520 metric/configuration/evidence values; 29 mechanism/combination summaries; 475 artifact-inventory rows. Indexed pages contain 2,394,178,683 bytes.

Validation checked every indexed page digest, byte count, native row/schema contract, contiguous page offset and contract coverage. Every projected experiment/check/value/metric cell was compared with a fresh projection of its canonical streamed experiment, including exact source pointers and declared row hashes. All 29 mechanism IDs, verdicts and scores were checked against the canonical mechanism summary. Every registered log path resolves to an existing file within the archive. Every page directory exactly matches the catalogs and contracts, with no missing or unindexed files. All indexes report zero omissions, consistent with the independent checks.

Tower implementation: clean local clone at commit 356861e5b83cd9e162fb39727604e1bc52bcf5a9. All 10 ordinary project output contracts also passed. The validation script was trusted project code; the uploaded archive was treated only as data.

Interpretation limits:

- Root metrics.jsonl contains bounded live progress. Complete per-experiment metric points are paged in outputs/roadmap and indexed by outputs/roadmap-tables.json. A short live stream is not missing experimental data.
- The final report stage has one infrastructure experiment; scientific conclusions come from report/mechanism_summary.json and experiment_summary.csv, not that inventory check.
- GOOD/BAD/NA and 1–100 are weighted evidence-attainment assessments. The checks carry NOT_A_PROOF; page validity is not mathematical proof or an efficiency/superiority result.
- 74,150 recorded experiments include the one reporting infrastructure record and many paired, correlated comparisons. They are not 74,150 independent data samples.
- The reporting exporter deliberately labels binary artifact hashes as declared/unverified. Separate provenance validation must establish checkpoint/science-seal integrity; this reporting pass did not deserialize checkpoints.
- The native Tower reader was executed in the cloud against a preserved copy of the native Fedora results. This verifies artifact-reader compatibility, not a new GPU execution or a manual GUI inspection.
