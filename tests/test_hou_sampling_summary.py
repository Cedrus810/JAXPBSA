"""Analysis must not silently turn incomplete/invalid sampling into a result."""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smallmol_records import ENERGY_FIELDS, file_identity
from summarize_hou_sampling import certify_frame_times, cohort_statistics, load_recovery


def test_invalid_replica_withholds_cohort_instead_of_dropping_frames():
    valid = [dict(delta_g_mmpbsa=-10, invalid="") for _ in range(100)]
    other = [dict(delta_g_mmpbsa=-20, invalid="") for _ in range(100)]
    other[-1]["invalid"] = "boundary margin below threshold"
    result = cohort_statistics([valid, other])
    assert result["mean"] is None and not result["available"]
    assert cohort_statistics([valid, []])["mean"] is None


def test_cohort_weights_replica_means_equally():
    # The production reader separately enforces frame counts; statistics retain
    # the independently specified equal-replica weighting contract.
    result = cohort_statistics([[dict(delta_g_mmpbsa=-10, invalid="")],
                                [dict(delta_g_mmpbsa=-20, invalid="")] * 4])
    assert result["mean"] == -15 and result["replica_se"] == 5


def test_strip_duplicate_frame_cannot_pass_a_count_only_audit():
    times = [dict(frame=i, step=(i + 1) * 5000, time_ps=(i + 1) * 10) for i in range(1000)]
    rows = [dict(step=r["step"]) for r in times]
    status = dict(frame_steps=5000, dt_ps=.002)
    certify_frame_times(times, status, rows)
    times[700]["frame"] = 699
    with pytest.raises(RuntimeError, match="frame 700"):
        certify_frame_times(times, status, rows)


@pytest.fixture
def recovery(tmp_path):
    setup = tmp_path / "md/hou/a1/ph74_gaff2/setup"
    setup.mkdir(parents=True)
    rep = setup.parent / "rep05"
    rep.mkdir()
    inputs = {}
    for key in ("system", "setup_record", "prmtop"):
        path = setup / key
        path.write_text("source")
        inputs[key] = file_identity(path)
    analyzer = dict(h=.5, h_lig=.25, pb_params=dict(tol=1e-5), radii_model="mbondi2",
                    gamma=.005, beta=0, sa_k=192, sa_points=960)
    status = dict(setup=str(setup), source_inputs=inputs, online=[dict(margin_flag=.8)],
                  online_inputs=dict(analyzer=analyzer, precision_bits=32, n_solute_atoms=100))
    run = dict(run_id="a1_ph74_gaff2_rep05", sample_id="a1", model_id="ph74_gaff2", replica="5")
    directory = tmp_path / "recovery"
    directory.mkdir()
    protocol = dict(schema=2, run_id=run["run_id"], target="hou", sample="a1", model="ph74_gaff2",
                    window_ns=[0, 10], stride_frames=1, analyzer=dict(analyzer, precision_bits=32),
                    inputs=inputs, trajectory=dict(dcd=str(rep / "analysis/solute_dry.dcd"),
                                                  frame_times=str(rep / "analysis/frame_times.csv"), n_atoms=100))
    (directory / "protocol.json").write_text(json.dumps(protocol))
    summary = dict(replica=str(rep), pilot="window", n_frames_scored=1000, n_valid=1000,
                   n_flagged=0, sizing=dict(margin_flag=.8))
    (directory / "summary.json").write_text(json.dumps(summary))
    times = [dict(frame=i, step=(i + 1) * 5000, time_ps=(i + 1) * 10) for i in range(1000)]
    rows = [dict(t, **{k: 0 for k in ENERGY_FIELDS}, converged="True", sa_ok="True",
                 margin_A=10, margin_lig_A=3, flag="") for t in times]
    return directory, run, status, times, rows, protocol


def save_rows(directory, rows):
    with (directory / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_recovery_requires_whole_replica_and_current_input_binding(recovery):
    directory, run, status, times, rows, protocol = recovery
    save_rows(directory, rows)
    assert len(load_recovery(directory, run, status, times)) == 1000
    Path(protocol["inputs"]["system"]["path"]).write_text("changed source")
    with pytest.raises(RuntimeError, match="identity changed"):
        load_recovery(directory, run, status, times)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_time", "flagged", "nan_margin",
                                    "wrong_grid", "wrong_precision", "missing_model", "relaxed_threshold"])
def test_recovery_rejects_partial_or_incompatible_results(recovery, damage):
    directory, run, status, times, rows, protocol = recovery
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[-2]
    elif damage == "wrong_time":
        rows[123]["time_ps"] = 0
    elif damage == "flagged":
        rows[123]["flag"] = "boundary margin below threshold"
    elif damage == "nan_margin":
        rows[123]["margin_lig_A"] = "nan"
    elif damage == "wrong_grid":
        protocol["analyzer"]["h"] = .4
        (directory / "protocol.json").write_text(json.dumps(protocol))
    elif damage == "wrong_precision":
        protocol["analyzer"]["precision_bits"] = 64
        (directory / "protocol.json").write_text(json.dumps(protocol))
    elif damage == "missing_model":
        del protocol["analyzer"]["pb_params"]
        (directory / "protocol.json").write_text(json.dumps(protocol))
    elif damage == "relaxed_threshold":
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["sizing"]["margin_flag"] = 0
        summary_path.write_text(json.dumps(summary))
    save_rows(directory, rows)
    with pytest.raises(RuntimeError):
        load_recovery(directory, run, status, times)
