# Independent uploaded-run provenance review

Run: `fedora-roadmap-20261008T022526329292Z`. Full protocol. All ten science and execution seals verified against current source `e46ae9f03dbffc80a0cc90c061dbb001a121a2db`.

Verified 435 science artifacts, 74150 experiment rows, 115 checkpoint file hashes, and both teacher manifests without unpickling uploaded models or executing uploaded code.

Native CPU readiness: 449 passed, 1 optional skip. Every GPU stage has exactly all 48 required passing identities (192 passes across four stages).

| Stage | Job | Numerical s | Allocation s | Max task RSS GiB |
|---|---:|---:|---:|---:|
| audit | 197 | 0.771 | 28 | 0.409 |
| headroom | 198 | 109.967 | 118 | 0.574 |
| prepare | 199 | 104.648 | 114 | 2.910 |
| train | 200 | 519.241 | 536 | 1.469 |
| confirm_prepare | 201 | 138.218 | 148 | 2.907 |
| confirm | 202 | 1485.579 | 1757 | 9.125 |
| policy | 203 | 767.490 | 1066 | 4.681 |
| transfer | 204 | 92.224 | 390 | 3.616 |
| scaling | 205 | 79.930 | 382 | 4.834 |
| report | 206 | 75.727 | 140 | 3.273 |

Total allocation elapsed 4,679 s (77 min 59 s); GPU allocations 3,741 s (62 min 21 s); allocated CPU 21,908 CPU-s. Numerical-stage sum 3373.795 s. These are different accounting scopes, not additive costs. No monetary estimate is available.

All artifact validation establishes integrity of retained evidence, not mathematical proof, fresh replication, or dominance over an official published benchmark. Native Slurm accounting is reviewed as an archived snapshot.
