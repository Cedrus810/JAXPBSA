#!/usr/bin/env python3
"""Audit the fixed Hou 6 x 10 ns budget without importing a molecular runtime.

Retain every committed online row. A cohort mean is withheld if any required
frame is invalid; do not drop flagged frames or average only surviving replicas.
Optional recovery replaces a WHOLE replica with an explicitly bound offline
replay, and is always labelled as offline recovery, never successful online MD.
Historical 3 x 5 ns reports and original CSV/status files are read only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys

from smallmol_records import (
    ENERGY_FIELDS, file_identity, online_rows, read_result_csv, row_violation,
)

PACKAGE = Path(__file__).resolve().parents[1] / "experiments/2026-10-17_small_molecule_benchmark"
COMPONENTS = ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa")
COHORTS = (("original3_0_5", (1, 2, 3), 0, 5000),
           ("original3_0_10", (1, 2, 3), 0, 10000),
           ("all6_0_5", tuple(range(1, 7)), 0, 5000),
           ("all6_5_10", tuple(range(1, 7)), 5000, 10000),
           ("all6_0_10", tuple(range(1, 7)), 0, 10000))


def certify_frame_times(times, status, rows):
    """Check the exact strip frame/step/time mapping, not just its row count."""
    if len(times) != 1000 or len(rows) != 1000:
        raise RuntimeError("expected exactly 1000 committed and stripped frames")
    for k, (t, r) in enumerate(zip(times, rows)):
        step = (k + 1) * status["frame_steps"]
        if (int(t["frame"]) != k or int(t["step"]) != step or int(r["step"]) != step
                or not math.isclose(float(t["time_ps"]), step * status["dt_ps"],
                                    rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError(f"strip time/step mismatch at frame {k}")


def cohort_statistics(replicas):
    """Each input is one required replica's complete, preselected window."""
    if not replicas or any(not rr or any(r["invalid"] for r in rr) for rr in replicas):
        return dict(available=False, mean=None, replica_se=None, replica_range=None)
    values = [st.fmean(float(r["delta_g_mmpbsa"]) for r in rr) for rr in replicas]
    return dict(available=True, mean=st.fmean(values),
                replica_se=st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
                replica_range=max(values) - min(values))


def load_recovery(directory, run, status, times):
    """Admit only a complete default-grid whole-window replay of these inputs."""
    protocol = json.loads((directory / "protocol.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (protocol.get("schema") != 2 or protocol.get("run_id") != run["run_id"]
            or protocol.get("target") != "hou" or protocol.get("sample") != run["sample_id"]
            or protocol.get("model") != run["model_id"]
            or protocol.get("window_ns") != [0, 10] or protocol.get("stride_frames") != 1
            or "pilot" in protocol or summary.get("pilot") != "window"
            or summary.get("n_frames_scored") != 1000 or summary.get("n_valid") != 1000
            or summary.get("n_flagged") != 0):
        raise RuntimeError(f"{directory}: recovery must be a complete 0-10 ns window-pilot replay")
    reference = status["online_inputs"]
    analyzer = dict(protocol["analyzer"])
    if set(analyzer) != {"h", "h_lig", "pb_params", "radii_model", "gamma", "beta",
                         "sa_k", "sa_points", "precision_bits"}:
        raise RuntimeError(f"{directory}: incomplete recovery analyzer protocol")
    precision = analyzer.pop("precision_bits")
    if precision != reference["precision_bits"] or any(
            reference["analyzer"].get(k) != v for k, v in analyzer.items()):
        raise RuntimeError(f"{directory}: recovery model/precision differs from the online protocol")
    if analyzer.get("h") != .5 or analyzer.get("h_lig") != .25:
        raise RuntimeError(f"{directory}: recovery must use the frozen default grid")
    for identity in protocol["inputs"].values():
        if file_identity(identity["path"]) != identity:
            raise RuntimeError(f"{directory}: recovery input identity changed")
    source_inputs = status["source_inputs"]
    # Current file identities and the original MD contract must both agree.
    for key in ("system", "setup_record", "prmtop"):
        if protocol["inputs"].get(key) != source_inputs.get(key):
            raise RuntimeError(f"{directory}: recovery {key} is not the MD input")
    trajectory = protocol["trajectory"]
    rep = Path(summary["replica"])
    if (trajectory["dcd"] != str((rep / "analysis/solute_dry.dcd").resolve())
            or trajectory["frame_times"] != str((rep / "analysis/frame_times.csv").resolve())
            or trajectory["n_atoms"] != reference["n_solute_atoms"]
            or str(rep.resolve()) != str(Path(status["setup"]).parent / f"rep{int(run['replica']):02d}")):
        raise RuntimeError(f"{directory}: recovery trajectory identity differs")
    rows = read_result_csv(directory / "frames.csv", ("frame", "step", "time_ps", *ENERGY_FIELDS,
                                                        "converged", "sa_ok", "margin_A", "margin_lig_A", "flag"))
    if len(rows) != 1000:
        raise RuntimeError(f"{directory}: incomplete recovery CSV")
    rows.sort(key=lambda r: int(r["frame"]))
    threshold = float(summary["sizing"]["margin_flag"])
    first_segment = status["online"][0]
    online_threshold = first_segment.get("margin_flag", first_segment.get("sizing", {}).get("margin_flag"))
    if online_threshold is None or threshold != float(online_threshold):
        raise RuntimeError(f"{directory}: recovery boundary threshold differs")
    for k, (r, t) in enumerate(zip(rows, times)):
        if (int(r["frame"]) != k or int(r["step"]) != int(t["step"])
                or not math.isclose(float(r["time_ps"]), float(t["time_ps"]), rel_tol=0, abs_tol=1e-6)
                or not math.isfinite(float(r["margin_lig_A"]))
                or row_violation(r, threshold)):
            raise RuntimeError(f"{directory}: invalid recovery frame {k}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(package, out, recovery_root=None):
    runs_path = package / "manifests/runs.csv"
    runs = [r for r in csv.DictReader(runs_path.open()) if r["target"] == "hou"]
    expected = {(f"a{s}", r) for s in range(1, 8) for r in range(1, 7)}
    if len(runs) != 42 or {(r["sample_id"], int(r["replica"])) for r in runs} != expected:
        raise RuntimeError("manifest is not exactly seven samples x six replicas")
    replicas, blocks, flagged, receipts, replay_diffs = [], [], [], [], []
    chosen, original = {}, {}
    for run in runs:
        rep = package / run["output_directory"]
        status_path = rep / "status.json"
        status = json.loads(status_path.read_text())
        if (not status.get("complete") or not status.get("md_complete") or not status.get("online_complete")
                or status["chunks_done"] != 100 or status["chunk_steps"] != 50000
                or status["frame_steps"] != 5000 or status["dt_ps"] != .002
                or status["run_id"] != run["run_id"]):
            raise RuntimeError(f"{run['run_id']}: production/online coverage is not complete at 10 ns")
        rows = online_rows(status)
        times_path = rep / "analysis/frame_times.csv"
        times = list(csv.DictReader(times_path.open()))
        certify_frame_times(times, status, rows)
        bad = [row_violation(r, r["_margin_flag"]) for r in rows]
        if status.get("online_valid") != (not any(bad)):
            raise RuntimeError(f"{run['run_id']}: status.online_valid disagrees with committed rows")
        for k, (r, reason) in enumerate(zip(rows, bad)):
            if reason:
                flagged.append(dict(run_id=run["run_id"], frame=k, step=int(r["step"]),
                                    time_ps=int(r["step"]) * status["dt_ps"], reason=reason,
                                    margin_A=r["margin_A"], margin_lig_A=r["margin_lig_A"]))
        receipts.extend(file_identity(f) for f in (status_path, times_path))
        receipts.extend(file_identity(s["csv"]) for s in status["online"] if s["to_chunk"] > s["from_chunk"])
        source = "online_default"
        selected = rows
        original[(run["sample_id"], int(run["replica"]))] = [
            dict(r, invalid=reason, time_ps=int(r["step"]) * status["dt_ps"])
            for r, reason in zip(rows, bad)]
        recovered = False
        recovery = recovery_root / run["sample_id"] / run["model_id"] / rep.name if recovery_root else None
        if any(bad) and recovery and recovery.exists():
            selected = load_recovery(recovery, run, status, times)
            recovered, source = True, "offline_recovered_whole_replica_default"
            receipts.extend(file_identity(recovery / f) for f in ("frames.csv", "protocol.json", "summary.json"))
            for label, indices in (("all", list(range(1000))),
                                   ("original_valid", [k for k in range(1000) if not bad[k]]),
                                   ("original_flagged", [k for k in range(1000) if bad[k]])):
                replay_diffs.append(dict(run_id=run["run_id"], subset=label, n_frames=len(indices),
                    **{f"mean_delta_{key}": st.fmean(float(selected[k][key]) - float(rows[k][key]) for k in indices)
                       for key in COMPONENTS},
                    max_abs_delta_total=max(abs(float(selected[k]["delta_g_mmpbsa"]) - float(rows[k]["delta_g_mmpbsa"]))
                                            for k in indices)))
        normalized = [dict(r, invalid="" if recovered else reason, time_ps=int(r["step"]) * status["dt_ps"])
                      for r, reason in zip(selected, bad)]
        chosen[(run["sample_id"], int(run["replica"]))] = normalized
        stat = cohort_statistics([normalized])
        replicas.append(dict(run_id=run["run_id"], sample=run["sample_id"], replica=int(run["replica"]),
                             source=source, n_frames=1000, online_flagged=sum(bool(x) for x in bad),
                             online_valid=not any(bad), recovered_offline=recovered,
                             min_online_margin_A=min(float(r["margin_A"]) for r in rows),
                             min_online_margin_lig_A=min(float(r["margin_lig_A"]) for r in rows),
                             usable_for_full_window=stat["available"], mean=stat["mean"]))
        for block in range(10):
            rr = normalized[block * 100:(block + 1) * 100]
            usable = not any(r["invalid"] for r in rr)
            blocks.append(dict(run_id=run["run_id"], sample=run["sample_id"], replica=int(run["replica"]),
                               block_start_ns=block, block_end_ns=block + 1, n_frames=len(rr),
                               invalid_frames=sum(bool(r["invalid"]) for r in rr), source=source,
                               **{f"mean_{key}": st.fmean(float(r[key]) for r in rr) if usable else None
                                  for key in COMPONENTS}))
    samples = []
    for cohort, ids, start, end in COHORTS:
        for sample in (f"a{i}" for i in range(1, 8)):
            # The historical baseline must remain the original online scores,
            # even after whole-replica replay changes valid as well as bad frames.
            source_rows = original if cohort == "original3_0_5" else chosen
            rr = [[r for r in source_rows[(sample, i)] if start < r["time_ps"] <= end] for i in ids]
            if any(len(r) != (end - start) // 10 for r in rr):
                raise RuntimeError(f"{sample}/{cohort}: incomplete analysis window")
            stat = cohort_statistics(rr)
            samples.append(dict(sample=sample, cohort=cohort, n_replicas_required=len(ids),
                                start_ns=start / 1000, end_ns=end / 1000,
                                offline_recovered_replicas=0 if cohort == "original3_0_5" else sum(
                                    r["recovered_offline"] for r in replicas
                                    if r["sample"] == sample and r["replica"] in ids),
                                invalid_frames=sum(bool(r["invalid"]) for x in rr for r in x),
                                **stat))
    pairs = []
    full = {r["sample"]: r for r in samples if r["cohort"] == "all6_0_10"}
    for a, b in itertools.combinations(full, 2):
        x, y = full[a], full[b]
        valid = x["available"] and y["available"]
        pairs.append(dict(sample_a=a, sample_b=b, available=valid,
                          mean_a_minus_b=x["mean"] - y["mean"] if valid else None,
                          combined_replica_se=math.hypot(x["replica_se"], y["replica_se"]) if valid else None))
    audit = dict(schema=1, manifest=file_identity(runs_path), analysis_script=file_identity(__file__),
                 n_replicas=42, n_frames=42000,
                 online_valid_replicas=sum(r["online_valid"] for r in replicas),
                 original_online_flagged_frames=len(flagged),
                 recovered_offline_replicas=sum(r["recovered_offline"] for r in replicas),
                 full_window_usable_replicas=sum(r["usable_for_full_window"] for r in replicas),
                 full_window_available_samples=[s for s, r in full.items() if r["available"]],
                 recovery_required=[r["run_id"] for r in replicas if not r["usable_for_full_window"]],
                 sources=receipts,
                 limitations=["Default-grid conditional sampling diagnostics; not a grid-convergence claim.",
                              "Offline-recovered replicas do not certify the original online sizing.",
                              "Same crystal start with different seeds does not establish equilibrium state populations.",
                              "Replica SE uses six replica means; 1-ns blocks are diagnostics, not independent replicas.",
                              "DCD content is not historically digest-bound; frame metadata and current MD input bindings are checked.",
                              "No experimental ranking or automatic significance threshold is evaluated."])
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in (("replicas", replicas), ("blocks", blocks), ("samples", samples), ("pairs", pairs)):
        write_csv(out / f"{filename}.csv", data)
    if flagged:
        write_csv(out / "original_online_flagged.csv", flagged)
    if replay_diffs:
        write_csv(out / "recovery_paired_differences.csv", replay_diffs)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in audit.items()
                      if k not in ("sources", "limitations", "manifest", "analysis_script")}))
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=PACKAGE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recovery-root", type=Path)
    args = ap.parse_args()
    analyze(args.package, args.out, args.recovery_root)


if __name__ == "__main__":
    main()
