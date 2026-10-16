TDN retained-data regularization review

Upload index.json, README.txt, and every file listed in index.json parts.
The ZIP contains the exact Raven run directory, raw research arrays, logs, and telemetry.
Original run files are preserved. The duplicate native numerical ZIP is omitted only
when every original numerical file is present and its hash matches packing.json.
Test fixture and plot-cache directories (pytest-temp, .matplotlib-cache, .font-cache) are excluded.
Verify each part's SHA-256 against index.json before reassembly.

Reassemble from this export directory:
  cat -- launch-630d054468ab4bb7915d33cc9482ebf2-a1-regularization-review.zip.part* > launch-630d054468ab4bb7915d33cc9482ebf2-a1-regularization-review.zip
Then verify the complete archive SHA-256 in index.json before extracting.
