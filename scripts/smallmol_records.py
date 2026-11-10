"""File/coverage contracts for small-molecule runs; stdlib only, no MD or JAX.

Input digests bind future writes, not scientific acceptance or cache validity.
Existing result files are never deleted when a contract differs.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path


ENERGY_FIELDS = ("e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand",
                 "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa")
AREA_FIELDS = ("sasa_complex", "sasa_receptor", "sasa_ligand")


def read_result_csv(path, required_fields, *, ignore_incomplete_tail=False):
    """Read a complete rectangular CSV; never certify a row with missing values.

    An online file may end in an uncommitted partial write. Discard only that
    unterminated final record; the caller's exact coverage check still rejects a
    missing committed frame. Offline resume requires a complete tail.
    """
    path = Path(path)
    data = path.read_text()
    if data and not data.endswith("\n"):
        if not ignore_incomplete_tail:
            raise RuntimeError(f"{path}: incomplete CSV tail")
        data = data[:data.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(data, newline=""), strict=True)
    try:
        names = reader.fieldnames or []
        if (not names or len(set(names)) != len(names) or "" in names
                or not set(required_fields) <= set(names)):
            raise ValueError("missing/duplicate/invalid CSV header")
        rows = list(reader)
        for row in rows:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("CSV row has extra or missing cells")
            for key in (*ENERGY_FIELDS, *AREA_FIELDS, "margin_A", "margin_lig_A", "wall_ms"):
                if key in row:
                    float(row[key])  # NaN is retained as a flagged failed frame, not a malformed number.
            for key in ("converged", "sa_ok"):
                if key in row and row[key] not in ("True", "False"):
                    raise ValueError(f"invalid boolean {key}")
        return rows
    except (csv.Error, TypeError, ValueError) as exc:
        raise RuntimeError(f"{path}: invalid result CSV: {exc}") from exc


def steps_from_ps(duration_ps, dt_ps, name):
    """Reject time requests that cannot be represented by whole MD steps."""
    if not math.isfinite(dt_ps) or dt_ps <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(duration_ps) or duration_ps <= 0:
        raise ValueError(f"{name} must be finite and positive")
    ratio = duration_ps / dt_ps
    if not math.isfinite(ratio):
        raise ValueError(f"{name} exceeds the supported step range")
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{name} must be an integer multiple of dt={dt_ps} ps")
    return steps


def setup_input_contract(row):
    return dict(schema=1, sample=row["sample"],
                inputs={k: file_identity(row[k]) for k in ("prmtop", "coordinates")},
                n_system_atoms=int(row["n_system_atoms"]), n_solute_atoms=int(row["n_solute_atoms"]),
                ligand_residue=row["ligand_residue"], ligand_residue_1based=int(row["ligand_residue_1based"]),
                ligand_local_idx=[int(row["ligand_atom_start_0based"]), int(row["ligand_atom_end_exclusive"])])


def validate_setup(setup, row, *, contract=None):
    """Match cached preparation to current inputs before skipping or using it."""
    setup = Path(setup)
    saved = json.loads((setup / "setup_record.json").read_text())
    current = setup_input_contract(row) if contract is None else contract
    for key in ("sample", "n_solute_atoms", "ligand_residue", "ligand_local_idx"):
        if saved.get(key) != current[key]:
            raise RuntimeError(f"{setup}: cached setup {key} differs from inputs; use a new setup/model")
    if saved.get("n_atoms") != current["n_system_atoms"]:
        raise RuntimeError(f"{setup}: cached setup atom count differs from inputs")
    for key in ("prmtop", "coordinates"):
        if Path(saved.get(key, "")).resolve() != Path(row[key]).resolve():
            raise RuntimeError(f"{setup}: cached setup {key} source path differs from inputs")
    if "input_contract" in saved and saved["input_contract"] != current:
        raise RuntimeError(f"{setup}: cached setup input content/selection changed; use a new setup/model")
    return saved


def bind_online_inputs(status, current):
    """Bind pilot and analyzer options before a new production session starts."""
    previous = status.get("online_inputs")
    if previous is not None and previous != current:
        raise RuntimeError("online pilot/analyzer settings changed; use the original inputs or a new run")
    if previous is None and status.get("chunks_done", 0):
        status["legacy_online_input_history_unverified"] = True
    status["online_inputs"] = copy.deepcopy(current)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def bind_score_protocol(out, protocol, *, csv_name="frames.csv"):
    """Write before the first CSV row; a crash cannot erase resume identity."""
    out = Path(out)
    record = out / "protocol.json"
    if record.exists():
        if json.loads(record.read_text()) != protocol:
            raise RuntimeError(f"{out}: scoring inputs/protocol changed; use a new --out directory")
    elif (out / csv_name).exists():
        raise RuntimeError(
            f"{out}: legacy {csv_name} has no protocol.json; retained without modification. "
            "Use a new --out directory for new scoring; do not append unbound results.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(record, protocol)


def scored_frames(path, expected, *, required_fields=()):
    """Reject mixed windows, duplicate IDs, wrong time labels and truncated rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    seen = set()
    rows = read_result_csv(path, ("frame", "step", "time_ps", "delta_g_mmpbsa", "flag", "wall_ms", *required_fields))
    for row in rows:
        try:
            k = int(row["frame"])
            step, time_ps = expected[k]
            if (k in seen or int(row["step"]) != step
                    or not math.isclose(float(row["time_ps"]), time_ps, rel_tol=0, abs_tol=1e-6)):
                raise ValueError("duplicate, wrong metadata or incomplete row")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{path}: invalid saved score row: {row}") from exc
        seen.add(k)
    return seen


def sensitivity_frames(path, expected, times, *, required_fields=()):
    """Per-(configuration, frame, rotation) admission, including partial configs."""
    path = Path(path)
    seen = set()
    if path.exists() and path.stat().st_size:
        rows = read_result_csv(path, ("config", "frame", "rotation", "time_ps", "flag", "wall_ms", *required_fields))
        for row in rows:
            try:
                key = row["config"], int(row["frame"]), int(row["rotation"])
                if (key not in expected or key in seen or row.get("wall_ms") is None
                        or row.get("flag") is None
                        or not math.isclose(float(row["time_ps"]), times[key[1]], rel_tol=0, abs_tol=1e-6)):
                    raise ValueError("wrong metadata, duplicate or incomplete row")
            except (KeyError, ValueError, TypeError) as exc:
                raise RuntimeError(f"{path}: invalid sensitivity row: {row}") from exc
            seen.add(key)
    return seen


def validate_md_resume(status, row, setup, seeds, source_inputs):
    expected = dict(run_id=row["run_id"], target=row["target"], sample_id=row["sample_id"],
                    model_id=row["model_id"], replica=int(row["replica"]),
                    setup=str(setup), seeds=seeds)
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError(f"MD resume {key} changed; use the original inputs or a new run")
    prior = status.get("source_inputs")
    if prior is not None and prior != source_inputs:
        raise RuntimeError("MD resume inputs changed (prmtop, coordinates or setup); use a new run")
    # Legacy status files can bind only the present inputs, not reconstruct
    # historical bytes. Keep that limitation explicit rather than invalidate MD.
    if prior is None:
        status["legacy_input_history_unverified"] = True
    status["source_inputs"] = source_inputs


def online_rows(status, *, through_step=None):
    """Select only committed segment ranges, then certify exact step/time coverage.

    Rows from an uncommitted chunk are retained on disk and ignored. Missing or
    overlapping committed segments fail, including after a final worker failure.
    This checks completeness, not the physical/numerical validity of energies.
    """
    cs, fs, dt = status["chunk_steps"], status["frame_steps"], status["dt_ps"]
    done = status["chunks_done"]
    if (any(type(x) is not int for x in (cs, fs, done)) or min(cs, fs) <= 0 or done < 0
            or cs % fs or not math.isfinite(dt) or dt <= 0):
        raise RuntimeError("online coverage: invalid MD step/time metadata")
    committed_end = done * cs
    end = committed_end if through_step is None else through_step
    if type(end) is not int or not 0 <= end <= committed_end or end % fs:
        raise RuntimeError("online coverage: requested end is outside committed frames or off the frame grid")
    expected = set(range(fs, end + 1, fs))
    selected = {}
    for seg in status.get("online", []):
        first, last = seg["from_chunk"], seg["to_chunk"]
        if type(first) is not int or type(last) is not int or not 0 <= first <= last <= done:
            raise RuntimeError("online coverage: invalid committed segment range")
        lo, hi = first * cs, min(last * cs, end)
        if hi <= lo:
            continue
        try:
            required = ("step", "time_ps", *ENERGY_FIELDS, "converged", "sa_ok", "margin_A", "margin_lig_A")
            rows = read_result_csv(seg["csv"], required, ignore_incomplete_tail=True)
            threshold = seg.get("margin_flag", seg.get("sizing", {}).get("margin_flag"))
            if threshold is None or not math.isfinite(float(threshold)) or float(threshold) < 0:
                raise ValueError("missing or invalid boundary flag threshold")
            for row in rows:
                step = int(row["step"])
                if not lo < step <= hi:
                    continue
                t = float(row["time_ps"])
                if (step not in expected or step in selected or not math.isfinite(t)
                        # OpenMM accumulates time by adding dt each step: 1.0e-6 ps off at 9.9 ns
                        # (Hou 10 ns, 2026-10-29). Any offset below dt/2 still names one step.
                        or not math.isclose(t, step * dt, rel_tol=0, abs_tol=0.25 * dt)):
                    raise ValueError(f"duplicate/unexpected step or wrong time: {step}")
                row["_margin_flag"] = float(threshold)
                selected[step] = row
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"online coverage invalid in {seg['csv']}: {exc}") from exc
    missing = expected - selected.keys()
    if missing:
        raise RuntimeError(
            f"MD has committed frames but online scores are missing: {len(missing)} steps "
            f"(first {min(missing)}). MD is retained; replay saved frames to a separate "
            "offline score directory. This run cannot be certified as complete online scoring.")
    return [selected[k] for k in sorted(selected)]


def online_rows_by_frame(rep):
    """Committed online rows of one replica keyed by solute_dry frame index.

    Frames are matched by MD step through analysis/frame_times.csv.  Each row gets the
    reporter flag (row_violation with the segment's own threshold) and wall_ms from
    analysis_s, so it can stand in for an offline frames.csv row.  Read-only.
    """
    rep = Path(rep)
    status = json.loads((rep / "status.json").read_text())
    by_step = {int(r["step"]): r for r in online_rows(status)}
    out = {}
    for t in csv.DictReader((rep / "analysis/frame_times.csv").open()):
        r = by_step.get(int(t["step"]))
        if r is not None:
            out[int(t["frame"])] = dict(r, frame=t["frame"], flag=row_violation(r, r["_margin_flag"]),
                                        wall_ms=round(float(r.get("analysis_s", "nan")) * 1000, 2))
    return out, [seg["csv"] for seg in status.get("online", [])]


def row_violation(row, margin_flag):
    """CSV counterpart of the reporter rule, including finite-energy checks."""
    if not math.isfinite(margin_flag) or margin_flag < 0:
        raise ValueError("boundary flag threshold must be finite and nonnegative")
    if row.get("flag"):
        return row["flag"]
    for key in ("e_coul_rl", "e_lj_rl", "g_pb_complex", "g_pb_receptor", "g_pb_ligand",
                "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa", "sasa_complex",
                "sasa_receptor", "sasa_ligand", "margin_A"):
        if key in row and not math.isfinite(float(row[key])):
            return f"{key} contains NaN/Inf"
    if row["converged"] != "True" or row["sa_ok"] != "True":
        return "PB/SA failed"
    if float(row["margin_A"]) < margin_flag or float(row["margin_lig_A"]) < 0:
        return "boundary margin below threshold"
    if math.isinf(float(row["margin_lig_A"])):
        return "margin_lig_A contains Inf"
    return ""
