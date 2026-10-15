"""Re-solve only the uploaded training matrix; acquire no new reference data."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tdn.analysis.response_preconditioner.fitting import fit_parent_balanced_ridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("retained-fit-replay.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Replay output must remain inside this checkout")
    if output.exists():
        raise FileExistsError("Preserve the existing replay receipt; select a fresh --output")
    archive = ROOT / "results/native-response-preconditioner-2930/response-preconditioner-2930-review-20261015T225818448985Z.zip"
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sha == "242904177750e2c8fc5e6b2d6852980e6003aef94380207ebbfde76a98b3bfb5"
    with zipfile.ZipFile(archive) as bundle:
        read = lambda name: json.loads(bundle.read("outputs/smoke/" + name))
        index, selection = read("array_index.json"), read("selection.json")
        references = read("references.json")
        with np.load(io.BytesIO(bundle.read("outputs/smoke/arrays.npz")), allow_pickle=False) as arrays:
            records = [{**row, **{name: arrays[f"query_{i:03d}_{name}"].copy()
                                  for name in ("design", "teacher", "base")}}
                       for i, row in enumerate(index["queries"])]
    train = [row for row in records if row["split"] == "train"]
    A = np.concatenate([row["design"].reshape(40, -1).T for row in train])
    b = np.concatenate([(row["teacher"] - row["base"]).ravel() for row in train])
    ids = [row["parent_id"] for row in train for _ in row["teacher"].ravel()]
    weights = np.array([1 / (len(set(ids)) * ids.count(parent)) for parent in ids])
    weighted_A, weighted_b = A * np.sqrt(weights[:, None]), b * np.sqrt(weights)
    U, singular, Vh = np.linalg.svd(weighted_A, full_matrices=False)
    comparisons = []
    for old in selection["candidates"]:
        if old["ridge"] is None:
            continue
        lam = old["ridge"]
        fit = fit_parent_balanced_ridge(A, b, ids, ridge=lam)
        w = fit["coefficients"].numpy()
        independent = Vh.T @ ((singular / (singular**2 + lam)) * (U.T @ weighted_b))
        original = np.array(old["coefficients"])
        np.testing.assert_allclose(w, independent, rtol=1e-10, atol=1e-13)
        independent_objective = float(np.sum((weighted_A @ independent - weighted_b)**2)
                                      + lam * np.sum(independent**2))
        assert abs(fit["diagnostics"]["objective"] - independent_objective) <= 1e-12 * independent_objective
        rows = []
        for row, reference in zip(records, references):
            columns = row["design"].reshape(40, -1).T
            difference = columns @ (w-original)
            error = row["base"].ravel() + columns @ w - row["teacher"].ravel()
            rows.append({key: row[key] for key in ("parent_id", "split", "kind", "horizon", "direction")} | {
                "prediction_change_rms": float(np.sqrt(np.mean(difference**2))),
                "prediction_change_maximum": float(np.max(np.abs(difference))),
                "repaired_prediction_rms_error": float(np.sqrt(np.mean(error**2))),
                "reference_uncertainty_rms": reference["reference"]["uncertainty_rms"],
            })
        comparisons.append(dict(ridge=lam, old_augmented_condition=old["diagnostics"]["augmented_condition_number"],
            repaired_diagnostics=fit["diagnostics"], independent_objective=independent_objective,
            independent_coefficient_maximum_difference=float(np.max(np.abs(w-independent))),
            old_coefficient_maximum_difference=float(np.max(np.abs(w-original))), predictions=rows))
    result = dict(scope="retained native training-matrix replay; no teacher acquisition or fresh evidence",
                  archive_sha256=sha, torch=torch.__version__, numpy=np.__version__,
                  source_sha256=hashlib.sha256((ROOT / "tdn/analysis/response_preconditioner/fitting.py").read_bytes()).hexdigest(),
                  original_objective_preserved=True, scientific_promotion=False,
                  comparisons=comparisons)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in comparisons:
        print(f"ridge={row['ridge']:g}: condition {row['old_augmented_condition']:.6g} -> "
              f"{row['repaired_diagnostics']['augmented_condition_number']:.6g}; "
              f"rank={row['repaired_diagnostics']['original_design_numerical_rank']}/40")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
