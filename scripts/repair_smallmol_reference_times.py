#!/usr/bin/env python3
"""Repair reference CSV step/time labels from record.json, preserving all energies.

Static metadata only: no Amber, OpenMM, trajectory or PB work. Default is a dry
run; --apply keeps a byte-for-byte backup before an atomic CSV replacement.
Paths are derived from the confirmed package runs manifest, never recursively scanned.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

from smallmol_records import write_json

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/2026-10-17_small_molecule_benchmark"


def repaired_csv(data, frames):
    rows = list(csv.DictReader(io.StringIO(data)))
    by_frame = {int(f["frame"]): f for f in frames}
    if len(by_frame) != len(frames) or len(rows) != len(frames):
        raise ValueError("reference frame metadata count/identity mismatch")
    seen, changed = set(), 0
    for r in rows:
        k = int(r["frame"])
        if k in seen or k not in by_frame:
            raise ValueError("duplicate or unknown reference frame")
        seen.add(k)
        correct = by_frame[k]
        for key in ("step", "time_ps"):
            if float(r[key]) != float(correct[key]):
                changed += 1
            r[key] = str(correct[key])
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    # This assertion guards every stored energy, flag, and frame ID, not just
    # a selected component whose unchanged value could hide a pairing mistake.
    original = list(csv.DictReader(io.StringIO(data)))
    for a, b in zip(original, rows):
        assert {k: v for k, v in a.items() if k not in ("step", "time_ps")} == {
            k: v for k, v in b.items() if k not in ("step", "time_ps")}
    return out.getvalue(), changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--frame-set", default="q3")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not a.frame_set or Path(a.frame_set).name != a.frame_set:
        ap.error("frame-set must be a single directory name")
    changes = []
    for run in csv.DictReader((a.package / "manifests/runs.csv").open()):
        if run["target"] != "brd4":
            continue
        rep = Path(run["output_directory"]).name
        d = a.package / "reference/amber_mmpbsa" / a.frame_set / run["target"] / run["sample_id"] / run["model_id"] / rep
        f = d / "comparison.csv"
        if not f.exists():
            continue
        original = f.read_bytes()
        fixed, n = repaired_csv(original.decode(), json.loads((d / "record.json").read_text())["frames"])
        if not n:
            continue
        backup = f.with_name("comparison.before_timefix_20261024.csv")
        item = dict(run_id=run["run_id"], path=str(f), changed_metadata_fields=n,
                    before_sha256=hashlib.sha256(original).hexdigest(), backup=str(backup))
        if a.apply:
            if backup.exists() and backup.read_bytes() != original:
                raise RuntimeError(f"existing backup differs: {backup}; nothing overwritten")
            if not backup.exists():
                shutil.copy2(f, backup)
            tmp = f.with_name(".comparison.timefix.partial")
            tmp.write_text(fixed)
            tmp.replace(f)
            item["after_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        changes.append(item)
    report = dict(applied=a.apply, frame_set=a.frame_set, files_changed=len(changes),
                  energy_and_frame_fields_unchanged=True, files=changes)
    if a.apply and changes:
        report_path = a.package / "reports" / f"reference_timefix_{a.frame_set}_20261024.json"
        write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
