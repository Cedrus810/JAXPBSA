#!/usr/bin/env python3
"""Freeze a local source + report review bundle; excludes raw MD and third-party papers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "experiments/2026-10-17_small_molecule_benchmark"


def recorded_git_base():
    """Read local metadata without invoking Git on the shared NFS checkout."""
    gitdir = ROOT / ".git"
    try:
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head.removeprefix("ref: ")
        path = gitdir / ref
        if path.is_file():
            return path.read_text().strip()
        for line in (gitdir / "packed-refs").read_text().splitlines():
            if not line.startswith(("#", "^")) and line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unavailable in this copy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = {ROOT / n for n in ("LICENSE", "pyproject.toml", "README.md", "README.zh.md",
                                   "RESULTS.md", "CHANGELOG.md", "TODO.md")}
    # Only named, shallow code/report directories; never traverse md/ or data/.
    code_dirs = [ROOT / "jaxpbsa" / name for name in
                 ("", "pb", "mm", "sa", "openmm_io", "benchmark", "pb/_tables")]
    code_dirs += [ROOT / "scripts", ROOT / "tests", PKG / "scripts"]
    for directory in code_dirs:
        selected.update(p for p in directory.iterdir() if p.suffix in (".py", ".npz"))
    report_dirs = [PKG / "reports" / name for name in (
        "", "baseline_20261101", "baseline_20261101_v2", "brd4_statistics_20261101",
        "brd4_statistics_20261101_v2", "evidence_20261102", "hou_ranking_20261030",
        "hou_sampling_20261030_online", "hou_sampling_20261030_verified")]
    for directory in report_dirs:
        selected.update(p for p in directory.iterdir() if p.suffix in (".csv", ".json", ".md", ".patch"))
    selected.update((PKG / "manifests").glob("*.csv"))
    selected.add(PKG / "sources/brd4/LICENSE")
    selected.update((ROOT / "docs").glob("ONLINE_MMPBSA_PAPER*.md"))
    for name in ("PAPER_REPRODUCIBILITY.md", "PAPER_SOFTWARE_COMPARISON.md", "PAPER_CLOSEOUT_20261106.md",
                 "ONLINE_SCHEDULING.md", "BRD4_HOU_NEXT_STEPS_PLAN_20261101.md"):
        selected.add(ROOT / "docs" / name)
    selected.update((ROOT / "docs/paper_assets").glob("*"))
    selected = sorted(p for p in selected if p.is_file() and "__pycache__" not in p.parts)
    files = []
    payloads = []
    for path in selected:
        data = path.read_bytes()
        name = path.relative_to(ROOT).as_posix()
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        payloads.append((name, data))
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "pytest", "jax", "jaxlib", "openmm", "mdtraj", "rdkit"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = "not installed"
    record = dict(date="2026-11-06", base_commit=recorded_git_base(),
        scope="Current working-source snapshot, including pre-existing uncommitted changes; not a historical run snapshot",
        environment_capture="Metadata only, at bundle creation; not an assertion of historical job versions",
        python=platform.python_version(), platform=platform.platform(), packages=versions, files=files,
        omissions=["Raw MD trajectories and original per-frame scoring files", "prepared topology/coordinate files",
                   "Third-party article/SI full text", "Private machine configuration and credentials"],
        reproduction="python scripts/reproduce_paper.py --out experiments/paper-reproduction")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads + [("SNAPSHOT.json", (json.dumps(record, indent=2)+"\n").encode())]:
            info = zipfile.ZipInfo(name, date_time=(2026,11,6,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    print(json.dumps(dict(bundle=str(args.out), files=len(files), uncompressed_bytes=sum(f['bytes'] for f in files), bytes=args.out.stat().st_size,
                          sha256=hashlib.sha256(args.out.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
