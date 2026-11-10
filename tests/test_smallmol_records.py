"""Resume/coverage/reference metadata regressions; no molecular runtime is invoked."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from smallmol_records import (bind_score_protocol, file_identity, online_rows, row_violation,
                             scored_frames, sensitivity_frames, validate_md_resume)


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def csv_rows(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def online_frame(step):
    return dict(step=step, time_ps=step * .002, converged="True", sa_ok="True", margin_A=2.,
                margin_lig_A=1., e_coul_rl=0., e_lj_rl=0., g_pb_complex=1., g_pb_receptor=0.,
                g_pb_ligand=0., delta_g_pb=1., delta_g_sa=0., delta_g_mmpbsa=1.)


def status_for(tmp_path, rows):
    path = tmp_path / "online.csv"; csv_rows(path, rows)
    return dict(chunk_steps=10, frame_steps=5, dt_ps=.002, chunks_done=2,
                online=[dict(csv=str(path), from_chunk=0, to_chunk=2, margin_flag=.8)])


def test_online_coverage_uses_only_committed_rows(tmp_path):
    s = status_for(tmp_path, [online_frame(x) for x in [5, 10, 15, 20, 25]])
    assert [int(x["step"]) for x in online_rows(s)] == [5, 10, 15, 20]
    s["online"][0]["to_chunk"] = 1
    path = tmp_path / "resumed.csv"; csv_rows(path, [online_frame(15), online_frame(20)])
    s["online"].append(dict(csv=str(path), from_chunk=1, to_chunk=2, margin_flag=.8))
    assert len(online_rows(s)) == 4


@pytest.mark.parametrize("case", ["missing", "duplicate", "time", "truncated", "overlap"])
def test_incomplete_or_ambiguous_online_scores_fail(tmp_path, case):
    rows = [online_frame(x) for x in [5, 10, 15, 20]]
    if case == "missing": rows.pop()
    if case == "duplicate": rows.append(online_frame(20))
    if case == "time": rows[1]["time_ps"] = 999.
    if case == "truncated": rows[-1]["delta_g_mmpbsa"] = ""
    s = status_for(tmp_path, rows)
    if case == "overlap": s["online"].append(s["online"][0].copy())
    with pytest.raises(RuntimeError, match="online"):
        online_rows(s)


def test_csv_validity_checks_finite_values_and_positive_flag_threshold():
    row = {k: str(v) for k, v in online_frame(5).items()}
    assert not row_violation(row, .8)
    row["margin_A"] = ".2"
    assert row_violation(row, .8)
    row["margin_A"] = "2"; row["delta_g_mmpbsa"] = "nan"
    assert "NaN/Inf" in row_violation(row, .8)


def test_score_protocol_survives_crash_without_summary(tmp_path):
    proto = dict(run_id="a1", window=[0, 5], inputs={"system": "abc"})
    bind_score_protocol(tmp_path, proto)
    (tmp_path / "frames.csv").write_text("frame,step,time_ps,flag,wall_ms,delta_g_mmpbsa\n0,5,.01,,1,2\n")
    assert not (tmp_path / "summary.json").exists()
    bind_score_protocol(tmp_path, proto)
    for change in (dict(proto, run_id="a2"), dict(proto, window=[0, 10]),
                   dict(proto, inputs={"system": "changed"})):
        with pytest.raises(RuntimeError, match="changed"):
            bind_score_protocol(tmp_path, change)
    assert scored_frames(tmp_path / "frames.csv", {0: (5, .01)}) == {0}


def test_legacy_scores_are_retained_without_silent_rebinding(tmp_path):
    path = tmp_path / "frames.csv"; path.write_text("historical results")
    with pytest.raises(RuntimeError, match="legacy"):
        bind_score_protocol(tmp_path, {"run_id": "a1"})
    assert path.read_text() == "historical results" and not (tmp_path / "protocol.json").exists()


def test_score_resume_rejects_mixed_or_duplicate_frame_ids(tmp_path):
    path = tmp_path / "frames.csv"
    row = dict(frame=0, step=5, time_ps=.01, flag="", wall_ms=1, delta_g_mmpbsa=2)
    csv_rows(path, [row, row])
    with pytest.raises(RuntimeError): scored_frames(path, {0: (5, .01)})
    csv_rows(path, [dict(row, frame=1)])
    with pytest.raises(RuntimeError): scored_frames(path, {0: (5, .01)})


def test_partial_sensitivity_config_keeps_missing_frame_work(tmp_path):
    path = tmp_path / "sensitivity.csv"
    csv_rows(path, [dict(config="h0.4", frame=124, rotation=0, time_ps=1250., flag="", wall_ms=1)])
    expected = {("h0.4", 124, 0), ("h0.4", 249, 0)}
    assert expected - sensitivity_frames(path, expected, {124: 1250., 249: 2500.}) == {("h0.4", 249, 0)}


@pytest.mark.parametrize("data", ["wrong,header\n", "frame,step,time_ps,flag,wall_ms,delta_g_mmpbsa\n0,5,.01,,1,2"])
def test_score_resume_refuses_malformed_header_or_incomplete_tail(tmp_path, data):
    path = tmp_path / "frames.csv"; path.write_text(data)
    with pytest.raises(RuntimeError): scored_frames(path, {0: (5, .01)})


def test_empty_csv_can_resume_before_first_header(tmp_path):
    path = tmp_path / "frames.csv"; path.touch()
    assert not scored_frames(path, {0: (5, .01)})


def test_md_resume_checks_seeds_and_source_bytes(tmp_path):
    row = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica="1")
    s = {**row, "replica": 1, "setup": str(tmp_path), "seeds": {"velocity": 1}}
    f = tmp_path / "source"; f.write_text("old")
    src = {"system": file_identity(f)}
    validate_md_resume(s, row, tmp_path, {"velocity": 1}, src)
    assert s["legacy_input_history_unverified"]
    with pytest.raises(RuntimeError, match="seeds"):
        validate_md_resume(s, row, tmp_path, {"velocity": 2}, src)
    f.write_text("new")
    with pytest.raises(RuntimeError, match="inputs changed"):
        validate_md_resume(s, row, tmp_path, {"velocity": 1}, {"system": file_identity(f)})


def test_failed_worker_drain_is_persisted(tmp_path):
    m = script("run_smallmol_md")
    def fail(): raise RuntimeError("worker crashed")
    rep = SimpleNamespace(close=fail, n_submitted=4, n_reports=3, n_flagged=0, n_skipped=0,
                          total_pbsa_s=1., max_latency_s=.5, warmup_time_s=.1, sizing={}, worker_config={})
    seg = dict(from_chunk=0, to_chunk=0); s = dict(chunks_done=2, complete=False, online=[seg])
    path = tmp_path / "status.json"
    with pytest.raises(RuntimeError, match="worker crashed"):
        m.finish_online(rep, seg, s, path, 30)
    saved = json.loads(path.read_text())
    assert saved["complete"] is False
    assert saved["online"][0]["state"] == "failed"
    assert saved["online"][0]["to_chunk"] == 2 and saved["online"][0]["n_reports"] == 3


@pytest.mark.parametrize("complete_scores", [False, True])
def test_completed_md_retry_requires_online_coverage_without_starting_md(tmp_path, monkeypatch, complete_scores):
    m = script("run_smallmol_md")
    out, setup = tmp_path / "rep01", tmp_path / "setup"
    out.mkdir(); setup.mkdir(); (setup / "setup_record.json").write_text("{}")
    row = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica="1",
               velocity_seed="1", integrator_seed="2", barostat_seed="3")
    seeds = dict(velocity=1, integrator=2, barostat=3)
    s = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica=1,
             setup=str(setup), seeds=seeds, dt_ps=.002, frame_steps=5000, chunk_steps=50000,
             nvt_steps=50000, npt_steps=500000, nvt_done=True, npt_done=True, chunks_done=50,
             prod_target_history=[2500000], contexts=[], stage_wall_s={}, complete=False,
             online_required=True, online=[])
    if complete_scores:
        f = out / "online.csv"; csv_rows(f, [online_frame(k) for k in range(5000, 2500001, 5000)])
        s["online"] = [dict(csv=str(f), from_chunk=0, to_chunk=50, margin_flag=.8)]
    (out / "status.json").write_text(json.dumps(s))
    monkeypatch.setattr(m, "run_row", lambda *a: row)
    monkeypatch.setattr(m, "replica_paths", lambda *a: (out, setup, dict(prmtop="unused", coordinates="unused")))
    monkeypatch.setattr(m, "file_identity", lambda p: str(p))
    monkeypatch.setattr(m, "validate_setup", lambda *a: {})
    def no_md(*a): raise AssertionError("MD/device initialization must not be reached")
    monkeypatch.setattr(m, "make_platform", no_md)
    args = SimpleNamespace(package=tmp_path, inputs=tmp_path, run_id="a1_rep01", dcd_ps=10., chunk_ps=100.,
                           prod_ns=5., nvt_ps=100., npt_ps=1000., dry_run=False, online=True)
    if complete_scores:
        m.cmd_run(args)
    else:
        with pytest.raises(RuntimeError, match="online scores are missing"):
            m.cmd_run(args)
    saved = json.loads((out / "status.json").read_text())
    assert saved["md_complete"] is True
    assert saved["complete"] == complete_scores
    assert saved["online_complete"] == complete_scores


def test_reference_time_labels_follow_the_energy_frame_order():
    m = script("amber_reference_smallmol")
    used = [dict(frame=str(k), step=str(step), time_ps=str(t)) for k, step, t in
            [(124, 625000, 1250.), (249, 1250000, 2500.), (374, 1875000, 3750.)]]
    rows = m.comparison_rows([124, 249, 374], used, {}, [{}, {}, {}], [{}, {}, {}])
    assert [(int(r["frame"]), float(r["time_ps"])) for r in rows] == [(124, 1250.), (249, 2500.), (374, 3750.)]


def test_reference_metadata_repair_preserves_energies_and_is_idempotent():
    m = script("repair_smallmol_reference_times")
    text = "frame,step,time_ps,ours_e_coul,amber_EEL\n124,1875000,3750.0,-1.740249697730239,-1.7419\n"
    used = [dict(frame="124", step="625000", time_ps="1250.0")]
    fixed, n = m.repaired_csv(text, used)
    assert n == 2
    assert "124,625000,1250.0,-1.740249697730239,-1.7419" in fixed
    assert m.repaired_csv(fixed, used)[1] == 0
