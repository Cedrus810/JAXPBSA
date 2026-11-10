#!/usr/bin/env python3
"""Re-run every recorded summary/statistics command in a mirror package and compare with reports/.

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
        experiments/2026-10-17_small_molecule_benchmark/scripts/reproduce_reports.py \
        --mirror <empty dir outside the repository> --out <dir for reproduction.csv/.json>

Paper outline §8 W1 / plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §6 ("现有统计表能由记录的命令复现").
No molecular computation: only the read-only summary scripts that write into reports/.  The mirror
holds a real copy of reports/ and scripts/ and read-only symlinks to every other package entry
(md/, scores/, reference/, manifests/, ...), plus symlinks to the repository's scripts/ and jaxpbsa/,
so the scripts (which locate the package from their own path) write only into the mirror.  Scripts
that write elsewhere (make_runs_csv, prepare_*, build_thrombin_loop_modeller) and the retired
summarize_hou_online.py (docs/HOU_SESSION_HANDOFF_20261030.md §4) are not run.

Every file the commands rewrite is compared with the original: identical bytes; else numeric leaves
of CSV/JSON (max |difference|, string mismatches counted, path-like strings ignored); else "differs".
Original report files no command rewrites are listed as "no recorded command run here".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]
REL = PKG.relative_to(ROOT)
COMMANDS = [   # (label, argv relative to the mirror root; {M} = mirror package)
    ("brd4_round1 default", ["{M}/scripts/summarize_brd4_round1.py"]),
    ("brd4_round1 h0.4", ["{M}/scripts/summarize_brd4_round1.py", "h0.4"]),
    ("brd4 grids", ["{M}/scripts/compare_brd4_grids.py"]),
    ("reference q3 brd4", ["{M}/scripts/summarize_reference_q3.py", "brd4"]),
    ("reference q3 hou", ["{M}/scripts/summarize_reference_q3.py", "hou"]),
    ("grid diag D0", ["{M}/scripts/grid_diag_d0.py"]),
    ("grid diag D1", ["{M}/scripts/grid_diag_d1.py"]),
    ("hou grids", ["{M}/scripts/compare_hou_grids.py"]),
    ("hou replica diag", ["{M}/scripts/diag_hou_replicas.py"]),
    ("hou acceptance", ["{M}/scripts/summarize_hou_acceptance.py"]),
    ("hou online acceptance", ["{M}/scripts/summarize_hou_online_acc.py", "19854"]),
    ("hou sampling (package)", ["{M}/scripts/summarize_hou_sampling.py"]),
    ("hou strict audit online", ["scripts/summarize_hou_sampling.py", "--package", "{M}",
                                 "--out", "{M}/reports/hou_sampling_20261030_online"]),
    ("hou strict audit verified", ["scripts/summarize_hou_sampling.py", "--package", "{M}",
                                   "--recovery-root", "{M}/scores/window10_default/hou",
                                   "--out", "{M}/reports/hou_sampling_20261030_verified"]),
    ("hou ranking", ["scripts/rank_hou_sampling.py", "--package", "{M}",
                     "--analysis-dir", "{M}/reports/hou_sampling_20261030_verified",
                     "--out", "{M}/reports/hou_ranking_20261030", "--bootstrap-draws", "10000", "--seed", "20261030"]),
    ("P1.0 inventory", ["{M}/scripts/build_baseline_inventory.py"]),
    ("B1 collect", ["{M}/scripts/collect_start_scores.py"]),
    ("BRD4 ligand files", ["{M}/scripts/check_brd4_ligand_files.py"]),
    ("P1 v1", ["{M}/scripts/compare_baselines.py", "--out", "{M}/reports/baseline_20261101"]),
    ("P1 v2", ["{M}/scripts/compare_baselines.py", "--out", "{M}/reports/baseline_20261101_v2"]),
    ("P2 v1", ["{M}/scripts/brd4_statistics.py", "--out", "{M}/reports/brd4_statistics_20261101"]),
    ("P2 v2", ["{M}/scripts/brd4_statistics.py", "--out", "{M}/reports/brd4_statistics_20261101_v2"]),
    ("B1 repeat", ["{M}/scripts/compare_b1_repeat.py"]),
    ("BRD4-8 components", ["{M}/scripts/diag_brd4_8.py"]),
]
FRESH_OUT = ["hou_sampling_20261030_online", "hou_sampling_20261030_verified", "hou_ranking_20261030"]  # tools refuse existing --out


def build_mirror(root: Path) -> Path:
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} must be empty or absent")
    if root.resolve().is_relative_to(ROOT):
        raise SystemExit("put the mirror outside the repository")
    m = root / REL
    m.mkdir(parents=True)
    for name in ("scripts", "jaxpbsa"):
        (root / name).symlink_to(ROOT / name)
    for entry in PKG.iterdir():
        if entry.name not in ("reports", "scripts"):
            (m / entry.name).symlink_to(entry)
    shutil.copytree(PKG / "reports", m / "reports", symlinks=True)
    (m / "scripts").mkdir()
    for f in (PKG / "scripts").glob("*.py"):
        shutil.copy2(f, m / "scripts" / f.name)
    for name in FRESH_OUT:
        shutil.rmtree(m / "reports" / name)
    return m


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def as_num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compare(a: Path, b: Path):
    if a.read_bytes() == b.read_bytes():
        return "identical", 0.0, 0
    if a.suffix == ".csv":
        ra, rb = list(csv.reader(a.open())), list(csv.reader(b.open()))
        if len(ra) != len(rb) or any(len(x) != len(y) for x, y in zip(ra, rb)):
            return "differs (shape)", None, None
        pa = [("", v) for row in ra for v in row]
        pb = [("", v) for row in rb for v in row]
    elif a.suffix == ".json":
        pa, pb = list(leaves(json.loads(a.read_text()))), list(leaves(json.loads(b.read_text())))
        if [k for k, _ in pa] != [k for k, _ in pb]:
            return "differs (keys)", None, None
    elif a.suffix == ".pdf" and a.with_suffix(".png").exists() and b.with_suffix(".png").exists() \
            and a.with_suffix(".png").read_bytes() == b.with_suffix(".png").read_bytes():
        return "pdf differs, same-name png identical (pdf metadata timestamp)", None, None
    else:
        return "differs (binary/text)", None, None
    dmax, nstr = 0.0, 0
    for (_, x), (_, y) in zip(pa, pb):
        fx, fy = as_num(x), as_num(y)
        if fx is not None and fy is not None:
            dmax = max(dmax, abs(fx - fy))
        elif x != y and not ("/" in str(x) and "/" in str(y)):   # mirror vs real paths are expected
            nstr += 1
    return ("numeric only" if nstr == 0 else "differs (strings)"), dmax, nstr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.mirror.resolve()
    m = build_mirror(root)
    t0 = time.time()
    time.sleep(1.1)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(root / ".mpl"), OMP_NUM_THREADS="8")
    runs = []
    for label, argv in COMMANDS:
        cmd = [sys.executable] + [x.replace("{M}", str(m)) for x in argv]
        t = time.time()
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        (root / f".log_{label.replace(' ', '_').replace('/', '_')}.txt").write_text(p.stdout + p.stderr)
        runs.append(dict(label=label, command=" ".join(argv), rc=p.returncode, wall_s=round(time.time() - t, 1)))
        print(f"[{p.returncode}] {label} {runs[-1]['wall_s']} s", flush=True)
    rows = []
    for orig in sorted(x for x in (PKG / "reports").rglob("*") if x.is_file()):
        rel = orig.relative_to(PKG / "reports")
        new = m / "reports" / rel
        if not new.exists():
            rows.append(dict(file=str(rel), status="not regenerated (removed fresh-out dir, file absent)", max_abs_diff="", string_mismatches=""))
        elif new.stat().st_mtime < t0:
            rows.append(dict(file=str(rel), status="no recorded command run here", max_abs_diff="", string_mismatches=""))
        else:
            st, d, n = compare(orig, new)
            rows.append(dict(file=str(rel), status=st, max_abs_diff="" if d is None else f"{d:.3g}", string_mismatches="" if n is None else n))
    for new in sorted(x for x in (m / "reports").rglob("*") if x.is_file()):
        if not (PKG / "reports" / new.relative_to(m / "reports")).exists():
            rows.append(dict(file=str(new.relative_to(m / "reports")), status="new file (not in reports/)", max_abs_diff="", string_mismatches=""))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "reproduction.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "reproduction_runs.json").write_text(json.dumps(dict(mirror=str(root), python=sys.executable, runs=runs), indent=1) + "\n")
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
