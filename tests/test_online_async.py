"""Exercise real process/IPC scheduling without needing a CUDA driver."""
from __future__ import annotations

import csv
import gc
from dataclasses import dataclass
import os
from pathlib import Path
import time
import warnings
import weakref
from types import SimpleNamespace

import numpy as np
from openmm import unit
import pytest

from jaxpbsa.online_async import AsyncPBSAReporter, _worker_environment


@dataclass
class ControlledFactory:
    gate: str | None = None
    started: str | None = None
    failure: str | None = None

    def __call__(self):
        if self.failure == "init":
            raise ValueError("invalid analyzer configuration")
        return ControlledAnalyzer(self)


class ControlledAnalyzer:
    ref_extent = 2.0
    margin_flag = 0.5
    warmup_time_s = 0.0

    def __init__(self, factory):
        self.factory = factory

    def __call__(self, coords):
        if self.factory.started:
            Path(self.factory.started).touch()
        if self.factory.gate:
            deadline = time.monotonic() + 20
            while not Path(self.factory.gate).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test did not release the analyzer")
                time.sleep(0.01)
        if self.factory.failure == "compute":
            raise ValueError("analysis failed")
        if self.factory.failure == "crash":
            os._exit(17)  # This test factory owns no CUDA context.
        energy = float(coords.sum())
        return dict(
            e_coul_rl=energy, e_lj_rl=0.0,
            g_pb_complex=3.0, g_pb_receptor=1.0, g_pb_ligand=1.0, delta_g_pb=1.0,
            sasa_complex=3.0, sasa_receptor=2.0, sasa_ligand=2.0, delta_g_sa=-1.0,
            delta_g_mmpbsa=energy, solver_iters=[3, 2, 1], converged=True,
            sa_ok=True, margin_A=-1.0 if self.factory.failure == "margin" else 2.0,
            margin_lig_A=1.0)


class State:
    def __init__(self, coords, time_ps=2.0):
        self.coords, self.time_ps = coords, time_ps

    def getPositions(self, asNumpy=False):
        return unit.Quantity(self.coords, unit.angstrom)

    def getTime(self):
        return unit.Quantity(self.time_ps, unit.picosecond)


def wait_started(marker):
    deadline = time.monotonic() + 10
    while not marker.exists():
        assert time.monotonic() < deadline, "worker did not begin the analysis"
        time.sleep(0.01)


def test_report_returns_before_analysis_and_preserves_snapshot(tmp_path):
    gate, marker = tmp_path / "gate", tmp_path / "started"
    path = tmp_path / "async.csv"
    rep = AsyncPBSAReporter(ControlledFactory(str(gate), str(marker)), 500, [0, 2],
                            out_csv=path, max_pending=2)
    sim = SimpleNamespace(currentStep=1000)
    coords = np.array([[0., 0., 0.], [999., 999., 999.], [2., 0., 0.]])
    try:
        rep.report(sim, State(coords))
        wait_started(marker)
        assert rep.n_reports == 0 and rep.pending == 1
        # MD can advance before the worker is released; source storage is reused.
        sim.currentStep = 1200
        coords[:] += 7.0
        assert rep.describeNextReport(sim) == (300, True, False, False, False, False)
        gate.touch()
        sim.currentStep = 1500
        rep.report(sim, State(coords, time_ps=3.0))
    finally:
        gate.touch()
        rep.close(timeout=20)
    assert rep.n_submitted == rep.n_reports == 2 and rep.pending == 0
    assert rep.latest_result["step"] == 1500
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1000, 1500]
    assert [float(row["time_ps"]) for row in rows] == [2.0, 3.0]
    assert float(rows[0]["e_coul_rl"]) == 2.0
    assert float(rows[1]["e_coul_rl"]) == 44.0
    assert all(float(row["latency_s"]) >= float(row["analysis_s"]) for row in rows)
    assert rep._process.poll() == 0 and not rep._thread.is_alive()
    assert not rep._finalizer.alive
    rep.close()  # Idempotent; no duplicate CSV rows.


def test_latency_uses_shared_clock_despite_parent_perf_counter_offset(monkeypatch):
    counter = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: counter() + 1_000_000)
    with AsyncPBSAReporter(ControlledFactory(), 500, [0, 1]) as rep:
        rep.report(SimpleNamespace(currentStep=500),
                   State(np.array([[0., 0., 0.], [2., 0., 0.]])))
    result = rep.latest_result
    assert 0 <= result["queue_wait_s"] < 60
    assert 0 <= result["analysis_s"] <= result["latency_s"] < 60


def test_abandoned_reporter_warns_and_requests_nonblocking_drain(tmp_path):
    gate, marker = tmp_path / "gate", tmp_path / "started"
    path = tmp_path / "abandoned.csv"
    rep = AsyncPBSAReporter(ControlledFactory(str(gate), str(marker)), 500, [0, 1],
                            out_csv=path)
    thread, process = rep._thread, rep._process
    reference = weakref.ref(rep)
    try:
        rep.report(SimpleNamespace(currentStep=500),
                   State(np.array([[0., 0., 0.], [2., 0., 0.]])))
        wait_started(marker)
        with pytest.warns(ResourceWarning, match="without close.*1 pending frames"):
            del rep
            gc.collect()
        assert reference() is None  # Pump/finalizer must not hold the reporter.
        assert thread.is_alive() and process.poll() is None  # GC did not block/kill.
    finally:
        gate.touch()
        thread.join(20)
    assert not thread.is_alive() and process.poll() == 0
    with path.open() as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_abandoned_idle_reporter_exits_without_pending_warning():
    rep = AsyncPBSAReporter(ControlledFactory(), 500, [0, 1])
    thread, process = rep._thread, rep._process
    reference = weakref.ref(rep)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        del rep
        gc.collect()
        thread.join(20)
    assert reference() is None
    assert not thread.is_alive() and process.poll() == 0
    assert not [item for item in caught if issubclass(item.category, ResourceWarning)]


def test_context_exit_timeout_keeps_worker_alive_and_is_retryable(tmp_path):
    gate, marker = tmp_path / "gate", tmp_path / "started"
    rep = AsyncPBSAReporter(ControlledFactory(str(gate), str(marker)), 500, [0, 1],
                            close_timeout=0.02)
    try:
        with pytest.raises(TimeoutError, match="draining"):
            with rep:
                rep.report(SimpleNamespace(currentStep=500),
                           State(np.array([[0., 0., 0.], [2., 0., 0.]])))
                wait_started(marker)
        assert rep._process.poll() is None
    finally:
        gate.touch()
        rep.close(timeout=20)
    assert rep.pending == 0 and not rep._finalizer.alive


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_context_close_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="close_timeout"):
        AsyncPBSAReporter(ControlledFactory(), 500, [0, 1], close_timeout=timeout)


@pytest.mark.parametrize("policy", ["raise", "skip"])
def test_bounded_backlog_is_explicit_and_close_timeout_is_retryable(tmp_path, policy):
    gate, marker = tmp_path / "gate", tmp_path / "started"
    rep = AsyncPBSAReporter(ControlledFactory(str(gate), str(marker)), 500, [0, 1],
                            max_pending=1, on_backlog=policy)
    sim = SimpleNamespace(currentStep=500)
    state = State(np.array([[0., 0., 0.], [2., 0., 0.]]))
    try:
        rep.report(sim, state)
        wait_started(marker)
        sim.currentStep = 1000
        if policy == "raise":
            with pytest.raises(RuntimeError, match="PBSA backlog"):
                rep.report(sim, state)
        else:
            with pytest.warns(RuntimeWarning, match="skipped"):
                rep.report(sim, state)
            assert rep.n_skipped == 1
        assert rep.n_submitted == 1
        with pytest.raises(TimeoutError, match="draining"):
            rep.close(timeout=0.02)
        assert rep._process.poll() is None
    finally:
        gate.touch()
        rep.close(timeout=20)
    assert rep.n_reports == 1 and rep.pending == 0


@pytest.mark.parametrize("failure, message", [
    ("compute", "analysis failed"), ("crash", "17|EOFError"), ("margin", "step 500.*margin"),
])
def test_worker_failures_propagate_and_process_is_reaped(failure, message):
    rep = AsyncPBSAReporter(ControlledFactory(failure=failure), 500, [0, 1],
                            on_violation="raise")
    rep.report(SimpleNamespace(currentStep=500), State(np.array([[0., 0., 0.], [2., 0., 0.]])))
    with pytest.raises(RuntimeError, match=message):
        rep.close(timeout=20)
    assert rep._process.poll() is not None and not rep._thread.is_alive()


def test_initialization_error_reaches_parent():
    with pytest.raises(RuntimeError, match="invalid analyzer configuration"):
        AsyncPBSAReporter(ControlledFactory(failure="init"), 500, [0, 1])


def test_violation_is_written_by_worker_without_parent_poll(tmp_path):
    path = tmp_path / "flag.csv"
    rep = AsyncPBSAReporter(ControlledFactory(failure="margin"), 500, [0, 1], out_csv=path)
    try:
        rep.report(SimpleNamespace(currentStep=500), State(np.array([[0., 0., 0.], [2., 0., 0.]])))
        deadline = time.monotonic() + 10
        while not path.exists() or path.stat().st_size == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1 and float(rows[0]["margin_A"]) < 0
        assert rep.n_reports == 0  # Results persist even between MD callbacks.
    finally:
        rep.close(timeout=20)
    assert rep.n_flagged == 1


def test_worker_environment_isolated_and_requires_mps(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    env = _worker_environment(None, None, 1)
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["OMP_NUM_THREADS"] == "1"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    monkeypatch.setattr("jaxpbsa.online_async.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="MPS requested"):
        _worker_environment(20, 1, 1)
    with pytest.raises(ValueError, match="mps_thread_percentage"):
        _worker_environment(0, None, 1)


def test_mps_limits_apply_only_to_worker(monkeypatch):
    monkeypatch.setattr("jaxpbsa.online_async.shutil.which", lambda name: "/fake/mps-control")
    monkeypatch.setattr("jaxpbsa.online_async.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.delenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", raising=False)
    monkeypatch.delenv("CUDA_MPS_CLIENT_PRIORITY", raising=False)
    env = _worker_environment(20, 1, 1)
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "20"
    assert env["CUDA_MPS_CLIENT_PRIORITY"] == "1"
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ
    assert "CUDA_MPS_CLIENT_PRIORITY" not in os.environ


def test_missing_mps_daemon_and_process_launch_failures_are_clear(monkeypatch):
    monkeypatch.setattr("jaxpbsa.online_async.shutil.which", lambda name: "/fake/mps-control")
    monkeypatch.setattr("jaxpbsa.online_async.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="daemon is unavailable"):
        AsyncPBSAReporter(ControlledFactory(), 500, [0, 1], mps_thread_percentage=20)

    def cannot_launch(*args, **kwargs):
        raise FileNotFoundError("Python executable missing")

    monkeypatch.setattr("jaxpbsa.online_async.subprocess.Popen", cannot_launch)
    with pytest.raises(FileNotFoundError, match="Python executable missing"):
        AsyncPBSAReporter(ControlledFactory(), 500, [0, 1])


def _two_atom_system():
    import openmm as mm
    from openmm import app

    topology, system = app.Topology(), mm.System()
    nonbonded = mm.NonbondedForce()
    for chain_id, charge in (("A", 1.0), ("B", -1.0)):
        residue = topology.addResidue("GLY", topology.addChain(chain_id))
        topology.addAtom("O", app.element.oxygen, residue)
        system.addParticle(16.0)
        nonbonded.addParticle(charge, 0.3, 0.0)
    system.addForce(nonbonded)
    return system, topology, np.array([[0., 0., 0.], [4., 0., 0.]])


def test_openmm_keeps_stepping_with_other_reporters_while_worker_is_busy(tmp_path):
    import openmm as mm
    from openmm import app

    system, topology, coords = _two_atom_system()
    sim = app.Simulation(topology, system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(unit.Quantity(coords, unit.angstrom))

    class FrequentReporter:
        def __init__(self):
            self.steps = []

        def describeNextReport(self, simulation):
            return (2 - simulation.currentStep % 2, True, False, False, False, False)

        def report(self, simulation, state):
            self.steps.append(simulation.currentStep)

    gate, marker = tmp_path / "gate", tmp_path / "started"
    frequent = FrequentReporter()
    factory = ControlledFactory(str(gate), str(marker))
    # ControlledAnalyzer has ref_extent=2; use a translated two-Å reference here.
    sim.context.setPositions(unit.Quantity(coords / 2, unit.angstrom))
    rep = AsyncPBSAReporter(factory, 3, [0, 1])
    sim.reporters.extend([frequent, rep])
    try:
        sim.step(6)  # Must return while the analysis gate is still closed.
        wait_started(marker)
        assert sim.currentStep == 6 and frequent.steps == [2, 4, 6]
        assert rep.n_submitted == 2 and rep.n_reports == 0
    finally:
        gate.touch()
        rep.close(timeout=20)
    assert rep.n_reports == 2 and rep.latest_result["step"] == 6
    assert rep.latest_result["time_ps"] == pytest.approx(0.006)


def test_real_analyzer_in_child_matches_synchronous_result(tmp_path):
    from jaxpbsa.online import OnlineMMPBSA
    from jaxpbsa.pb import PBParams

    system, topology, coords = _two_atom_system()
    options = dict(h=1.0, h_lig=None, padding=8.0, sa_k=4, sa_points=32,
                   pb_params=PBParams(precond="jacobi", swin=0.0))
    synchronous = OnlineMMPBSA(system, topology, [0, 1], [1], coords, **options)
    expected = synchronous(coords)
    with AsyncPBSAReporter.from_system(
        system, topology, [0, 1], [1], coords, interval_steps=500,
        analyzer_kwargs=options, cache_dir=str(tmp_path / "jax-cache"),
    ) as rep:
        rep.report(SimpleNamespace(currentStep=500), State(coords))
    actual = rep.latest_result
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=1e-10, atol=0.0,
                                   err_msg=f"worker changed {key}")
