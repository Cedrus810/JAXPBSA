"""Cached-input and damaged-record regressions, without molecular computation."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from smallmol_records import (ENERGY_FIELDS, bind_online_inputs, file_identity, online_rows,
                             read_result_csv, scored_frames, sensitivity_frames,
                             setup_input_contract, steps_from_ps, validate_setup)


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def csv_rows(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def setup_case(tmp_path):
    setup = tmp_path / "setup"; setup.mkdir()
    for name in ("input.prmtop", "input.rst7"):
        (tmp_path / name).write_text("original")
    row = dict(sample="a1", prmtop=str(tmp_path / "input.prmtop"), coordinates=str(tmp_path / "input.rst7"),
               n_system_atoms="10", n_solute_atoms="3", ligand_residue="LIG", ligand_residue_1based="2",
               ligand_atom_start_0based="2", ligand_atom_end_exclusive="3")
    saved = dict(sample="a1", prmtop=row["prmtop"], coordinates=row["coordinates"], n_atoms=10,
                 n_solute_atoms=3, ligand_residue="LIG", ligand_local_idx=[2, 3],
                 input_contract=setup_input_contract(row))
    (setup / "setup_record.json").write_text(json.dumps(saved))
    return setup, row, saved


@pytest.mark.parametrize("change", ["coordinates", "prmtop", "selection", "sample"])
def test_completed_setup_is_not_reused_for_changed_inputs(tmp_path, change):
    setup, row, saved = setup_case(tmp_path)
    assert validate_setup(setup, row) == saved
    if change in ("coordinates", "prmtop"):
        Path(row[change]).write_text("replaced at the same path")
    elif change == "selection":
        row["ligand_atom_start_0based"] = "1"
    else:
        row["sample"] = "a2"
    with pytest.raises(RuntimeError, match="cached setup"):
        validate_setup(setup, row)
    assert json.loads((setup / "setup_record.json").read_text()) == saved


def test_setup_cli_checks_cache_before_openmm_work(tmp_path, monkeypatch, capsys):
    m = script("run_smallmol_md")
    setup, row, _ = setup_case(tmp_path)
    monkeypatch.setattr(m, "input_row", lambda *a: row)
    monkeypatch.setattr(m, "setup_dir", lambda *a: setup)
    args = SimpleNamespace(inputs=tmp_path, sample="a1", package=tmp_path, target="hou",
                           model_id="m", dry_run=False, force=False)
    m.cmd_setup(args)
    assert '"skip_setup_done"' in capsys.readouterr().out
    Path(row["coordinates"]).write_text("changed")
    with pytest.raises(RuntimeError, match="input content/selection changed"):
        m.cmd_setup(args)


def test_legacy_setup_checks_identity_without_inventing_past_hashes(tmp_path):
    setup, row, saved = setup_case(tmp_path)
    saved.pop("input_contract")
    (setup / "setup_record.json").write_text(json.dumps(saved))
    assert "input_contract" not in validate_setup(setup, row)
    row["prmtop"] = str(tmp_path / "other.prmtop")
    Path(row["prmtop"]).write_text("original")
    with pytest.raises(RuntimeError, match="source path"):
        validate_setup(setup, row)


def test_online_pilot_bytes_and_analyzer_defaults_are_bound(tmp_path):
    pilot = tmp_path / "pilot.npy"; pilot.write_bytes(b"static test data")
    current = dict(pilot=file_identity(pilot), analyzer=dict(h=.5, h_lig=.25), precision_bits=32)
    status = dict(chunks_done=0)
    bind_online_inputs(status, current)
    for key, replacement in [("analyzer", dict(h=.4, h_lig=.2)), ("precision_bits", 64)]:
        other = {**current, key: replacement}
        with pytest.raises(RuntimeError, match="pilot/analyzer"):
            bind_online_inputs(status, other)
    pilot.write_bytes(b"replaced pilot")
    with pytest.raises(RuntimeError, match="pilot/analyzer"):
        bind_online_inputs(status, {**current, "pilot": file_identity(pilot)})
    assert status["online_inputs"] == current
    legacy = dict(chunks_done=1)
    bind_online_inputs(legacy, current)
    assert legacy["legacy_online_input_history_unverified"]


def test_online_binding_takes_an_owned_snapshot():
    status = dict(chunks_done=0)
    current = dict(analyzer=dict(h=.5))
    bind_online_inputs(status, current)
    current["analyzer"]["h"] = .4
    with pytest.raises(RuntimeError, match="pilot/analyzer"):
        bind_online_inputs(status, current)


def test_online_contract_records_actual_defaults_without_device_work(tmp_path):
    m = script("run_smallmol_md")
    pilot = tmp_path / "pilot.npy"; pilot.write_bytes(b"static input identity only")
    contract = m.online_input_contract(pilot, dict(n_solute_atoms=3, ligand_local_idx=[2, 3]), 5000)
    assert contract["analyzer"]["h"] == .5 and contract["analyzer"]["h_lig"] == .25
    assert contract["analyzer"]["pb_params"]["precond"] == "auto"
    assert contract["precision_bits"] in (32, 64)
    json.dumps(contract, allow_nan=False)


def test_changed_online_pilot_fails_before_production_context(tmp_path, monkeypatch):
    m = script("run_smallmol_md")
    out, setup = tmp_path / "rep01", tmp_path / "setup"
    out.mkdir(); setup.mkdir(); (setup / "setup_record.json").write_text("{}")
    row = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica="1",
               velocity_seed="1", integrator_seed="2", barostat_seed="3")
    path = out / "online.csv"
    csv_rows(path, [dict(step=k, time_ps=k*.002, **{f: 1. for f in ENERGY_FIELDS}, converged="True",
                        sa_ok="True", margin_A=2., margin_lig_A=1.) for k in [5, 10]])
    s = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica=1,
             setup=str(setup), seeds=dict(velocity=1, integrator=2, barostat=3), dt_ps=.002,
             frame_steps=5, chunk_steps=10, nvt_steps=10, npt_steps=10, nvt_done=True, npt_done=True,
             chunks_done=1, prod_target_history=[20], contexts=[], stage_wall_s={}, complete=False,
             online_required=True, online_inputs={"pilot": "original"},
             online=[dict(csv=str(path), from_chunk=0, to_chunk=1, margin_flag=.8)])
    (out / "status.json").write_text(json.dumps(s))
    monkeypatch.setattr(m, "run_row", lambda *a: row)
    monkeypatch.setattr(m, "replica_paths", lambda *a: (out, setup, dict(prmtop="unused", coordinates="unused")))
    monkeypatch.setattr(m, "file_identity", lambda p: str(p))
    monkeypatch.setattr(m, "validate_setup", lambda *a: dict(n_solute_atoms=3, ligand_local_idx=[2, 3]))
    monkeypatch.setattr(m, "online_input_contract", lambda *a: {"pilot": "changed"})
    def no_device(*a): raise AssertionError("production device must not be initialized")
    monkeypatch.setattr(m, "make_platform", no_device)
    args = SimpleNamespace(package=tmp_path, inputs=tmp_path, run_id="a1_rep01", dcd_ps=.01, chunk_ps=.02,
                           prod_ns=.00004, nvt_ps=.02, npt_ps=.02, dry_run=False, online=True)
    with pytest.raises(RuntimeError, match="pilot/analyzer settings changed"):
        m.cmd_run(args)
    saved = json.loads((out / "status.json").read_text())
    assert saved["chunks_done"] == 1 and saved["online_inputs"] == {"pilot": "original"}


@pytest.mark.parametrize("duration", [.001, .003, .0105, 0., -1., float("nan"), float("inf"), 1e308])
def test_md_time_requests_are_not_silently_rounded(duration):
    with pytest.raises(ValueError):
        steps_from_ps(duration, .002, "duration")


def test_normal_md_durations_have_exact_step_counts():
    assert [steps_from_ps(x, .002, "duration") for x in [10., 100., 1000., 5000.]] == [5000, 50000, 500000, 2500000]


def test_short_online_pilot_fails_in_dry_run_before_any_input_io(tmp_path, monkeypatch):
    m = script("run_smallmol_md")
    monkeypatch.setattr(m, "run_row", lambda *a: {})
    monkeypatch.setattr(m, "replica_paths", lambda *a: (tmp_path, tmp_path, {}))
    args = SimpleNamespace(package=tmp_path, run_id="fake", inputs=tmp_path, dcd_ps=10., chunk_ps=100.,
                           prod_ns=5., nvt_ps=100., npt_ps=10., online=True, dry_run=True)
    with pytest.raises(ValueError, match="two NPT pilot frames"):
        m.cmd_run(args)


def online_case(tmp_path, steps=(5, 10, 15, 20)):
    path = tmp_path / "online.csv"
    rows = [dict(step=k, time_ps=k*.002, **{f: 1. for f in ENERGY_FIELDS}, converged="True",
                 sa_ok="True", margin_A=2., margin_lig_A=1.) for k in steps]
    csv_rows(path, rows)
    return dict(chunk_steps=10, frame_steps=5, chunks_done=2, dt_ps=.002,
                online=[dict(csv=str(path), from_chunk=0, to_chunk=2, margin_flag=.8)]), path


def test_uncommitted_partial_tail_does_not_hide_valid_committed_scores(tmp_path):
    status, path = online_case(tmp_path)
    with path.open("a") as fh: fh.write("25,0.05,unfinished")
    assert len(online_rows(status)) == 4


def test_partial_committed_frame_still_fails_coverage(tmp_path):
    status, path = online_case(tmp_path, (5, 10, 15))
    with path.open("a") as fh: fh.write("20,0.04,unfinished")
    with pytest.raises(RuntimeError, match="scores are missing"):
        online_rows(status)


@pytest.mark.parametrize("case", ["future_end", "off_grid", "bad_range", "negative_range", "zero_frame_steps", "nan_dt",
                                   "unknown_threshold", "nan_threshold"])
def test_online_audit_rejects_unprovable_ranges_and_thresholds(tmp_path, case):
    status, _ = online_case(tmp_path)
    kwargs = {}
    if case == "future_end": kwargs["through_step"] = 25
    if case == "off_grid": kwargs["through_step"] = 19
    if case == "bad_range": status["online"][0]["to_chunk"] = 3
    if case == "negative_range": status["online"][0]["from_chunk"] = -1
    if case == "zero_frame_steps": status["frame_steps"] = 0
    if case == "nan_dt": status["dt_ps"] = float("nan")
    if case == "unknown_threshold": status["online"][0].pop("margin_flag")
    if case == "nan_threshold": status["online"][0]["margin_flag"] = float("nan")
    with pytest.raises(RuntimeError): online_rows(status, **kwargs)


@pytest.mark.parametrize("case", ["duplicate_header", "extra_cell", "invalid_boolean", "invalid_energy"])
def test_result_csv_structure_and_values_are_checked(tmp_path, case):
    path = tmp_path / "result.csv"
    if case == "duplicate_header": data = "energy,energy\n1,2\n"
    elif case == "extra_cell": data = "energy\n1,2\n"
    elif case == "invalid_boolean": data = "energy,converged\n1,yes\n"
    else: data = "energy,delta_g_pb\n1,broken\n"
    path.write_text(data)
    with pytest.raises(RuntimeError, match="CSV"):
        read_result_csv(path, ["energy"])


def test_sensitivity_configuration_without_energy_columns_is_not_complete(tmp_path):
    path = tmp_path / "sensitivity.csv"
    csv_rows(path, [dict(config="h0.4", frame=124, rotation=0, time_ps=1250., flag="", wall_ms=1.)])
    with pytest.raises(RuntimeError, match="header"):
        sensitivity_frames(path, {("h0.4", 124, 0)}, {124: 1250.}, required_fields=ENERGY_FIELDS)


def test_offline_resume_checks_the_full_requested_energy_schema(tmp_path):
    path = tmp_path / "frames.csv"
    csv_rows(path, [dict(frame=0, step=5, time_ps=.01, delta_g_mmpbsa=2., flag="", wall_ms=1.)])
    with pytest.raises(RuntimeError, match="header"):
        scored_frames(path, {0: (5, .01)}, required_fields=ENERGY_FIELDS)


def test_offline_extension_uses_committed_step_and_time_with_stub_engine(tmp_path, monkeypatch):
    """Drive the actual runner through a production chunk with a no-physics engine."""
    m = script("run_smallmol_md")
    out, setup = tmp_path / "rep01", tmp_path / "setup"
    (out / "prod").mkdir(parents=True); setup.mkdir()
    (setup / "setup_record.json").write_text("{}")
    (setup / "system.xml").write_text("system")
    (out / "prod/chunk_0000_state.xml").write_text("state")
    row = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica="1",
               velocity_seed="1", integrator_seed="2", barostat_seed="3")
    status = dict(run_id="a1_rep01", target="hou", sample_id="a1", model_id="m", replica=1,
                  setup=str(setup), seeds=dict(velocity=1, integrator=2, barostat=3), dt_ps=.002,
                  frame_steps=5, chunk_steps=10, nvt_steps=10, npt_steps=10, nvt_done=True, npt_done=True,
                  chunks_done=1, prod_target_history=[10], contexts=[], stage_wall_s={}, complete=True)
    (out / "status.json").write_text(json.dumps(status))
    simulations, frame_times = [], []

    class State:
        def getPeriodicBoxVectors(self): return (1, 1, 1)
        def getPositions(self): return []
        def getVelocities(self): return []

    class Context:
        time = 0.
        def setPeriodicBoxVectors(self, *a): pass
        def setPositions(self, *a): pass
        def setVelocities(self, *a): pass
        def setTime(self, t): self.time = t
        def getState(self, **kw): return State()

    class Simulation:
        def __init__(self, *a):
            self.context, self.currentStep, self.reporters = Context(), 0, []
            simulations.append(self)
        def step(self, n):
            self.currentStep += n; self.context.time += n * .002

    class DCD:
        def __init__(self, *a, **kw): pass
        def writeModel(self, *a, **kw):
            frame_times.append((simulations[-1].currentStep, simulations[-1].context.time))

    class Seeded:
        def __init__(self, *a): pass
        def setRandomNumberSeed(self, *a): pass

    module = SimpleNamespace(
        XmlSerializer=SimpleNamespace(deserialize=lambda s: SimpleNamespace(addForce=lambda f: None) if s == "system" else State(),
                                      serialize=lambda s: "saved"),
        LangevinMiddleIntegrator=Seeded, MonteCarloBarostat=Seeded, __version__="stub",
        unit=SimpleNamespace(kelvin=1., picosecond=1., picoseconds=1., atmosphere=1.),
        app=SimpleNamespace(Simulation=Simulation, AmberPrmtopFile=lambda p: SimpleNamespace(topology=None),
                            DCDFile=DCD, StateDataReporter=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, "openmm", module)
    monkeypatch.setattr(m, "run_row", lambda *a: row)
    monkeypatch.setattr(m, "replica_paths", lambda *a: (out, setup, dict(prmtop="unused", coordinates="unused")))
    monkeypatch.setattr(m, "file_identity", lambda p: str(p))
    monkeypatch.setattr(m, "validate_setup", lambda *a: dict(n_solute_atoms=3, ligand_local_idx=[2, 3]))
    monkeypatch.setattr(m, "make_platform", lambda *a: (None, {}))
    monkeypatch.setattr(m, "device_info", lambda *a: {"platform": "stub"})
    args = SimpleNamespace(package=tmp_path, inputs=tmp_path, run_id="a1_rep01", dcd_ps=.01, chunk_ps=.02,
                           prod_ns=.00004, nvt_ps=.02, npt_ps=.02, dry_run=False, online=False,
                           platform="stub", device_index=None)
    m.cmd_run(args)
    assert [step for step, _ in frame_times] == [15, 20]
    assert [t for _, t in frame_times] == pytest.approx([.03, .04])
    saved = json.loads((out / "status.json").read_text())
    assert saved["chunks_done"] == 2 and saved["complete"] is True
