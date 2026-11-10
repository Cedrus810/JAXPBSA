#!/usr/bin/env python3
"""CPU-only review-bundle integrity, ranking and figure reproduction.

Does not import JAX/OpenMM or run molecular calculations. Outputs go to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "SNAPSHOT.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for entry in record["files"]:
            path = root / entry["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise SystemExit(f"Snapshot content changed: {entry['path']}")
        print(f"Snapshot verified: {len(record['files'])} files", flush=True)
    else:
        print("Working checkout: no frozen-bundle integrity check", flush=True)
    reports = root / "experiments/2026-10-17_small_molecule_benchmark/reports"
    sys.path.insert(0, str(root / "scripts"))
    from build_paper_assets import check_inputs
    from rank_hou_sampling import exact_statistics
    result = check_inputs(reports)
    result["four_sample_exact_test"] = exact_statistics([-30,-22,-21,-19], [-12,-10,-9,-8])
    expected = 2/24
    if abs(result["four_sample_exact_test"]["spearman_p_exact_two_sided"] - expected) > 1e-12:
        raise SystemExit("Exact permutation reference failed")
    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(root / "scripts/build_paper_assets.py"),
                    "--reports", str(reports), "--out", str(args.out / "figures")], check=True)
    reference = root / "docs/paper_assets"
    regenerated = args.out / "figures"
    csvs = sorted(reference.glob("*.csv"))
    if not csvs:
        raise SystemExit("Missing frozen figure-source CSVs")
    for path in csvs:
        if path.read_bytes() != (regenerated / path.name).read_bytes():
            raise SystemExit(f"Figure-data mismatch: {path.name}")
    result["identical_figure_csvs"] = len(csvs)
    (args.out / "reproduction.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
