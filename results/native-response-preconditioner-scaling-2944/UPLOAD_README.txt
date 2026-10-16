TDN Phase 1 independent-parent scaling review

Upload index.json, README.txt, and every file listed in index.json parts.
The ZIP contains the exact Raven run directory, raw research arrays, logs, and telemetry.
Original run files are preserved. The duplicate native numerical ZIP is omitted only
when every original numerical file is present and its hash matches packing.json.
Test fixture directories named pytest-temp are excluded.
Verify each part's SHA-256 against index.json before reassembly.

Reassemble from this export directory:
  cat -- launch-0e144975eb4b433492fe55d75d209ef2-a1-scaling-review.zip.part* > launch-0e144975eb4b433492fe55d75d209ef2-a1-scaling-review.zip
Then verify the complete archive SHA-256 in index.json before extracting.
