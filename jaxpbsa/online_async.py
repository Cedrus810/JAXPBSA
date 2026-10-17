"""Single-GPU OpenMM/PBSA scheduling with a persistent background process.

The reporter copies coordinates into a bounded host queue and returns. The worker
owns the analyzer, JAX executable/cache and CSV writer. CUDA/MPS environment is
set at process creation, before any worker imports. No CUDA context is forked.
MPS must be enabled separately before starting OpenMM; requested limits are
checked against a running daemon and apply only to the PBSA worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings
import weakref

import numpy as np
from openmm import XmlSerializer, unit

__all__ = ["AsyncPBSAReporter"]


def _receive(conn, process):
    while not conn.poll(0.1):
        if process.poll() is not None:
            raise RuntimeError(f"PBSA worker exited with code {process.returncode}")
    return conn.recv()


def _finalize_reporter(tasks, state):
    # Never wait for GPU work in a finalizer. The independent pump owns IPC and
    # reaps the worker after this best-effort request to drain normally.
    if not state["closing"]:
        tasks.put_nowait(None)  # Admission reserves a shutdown slot.
        state["closing"] = True
    if state["pending"] > 0:
        warnings.warn(
            f"AsyncPBSAReporter discarded without close(): {state['pending']} "
            "pending frames; results may be lost. Use a context manager or close().",
            ResourceWarning, stacklevel=2)


@dataclass
class _OnlineFactory:
    system_xml: str
    topology: object
    solute_idx: np.ndarray
    ligand_local_idx: np.ndarray
    ref_coords_A: np.ndarray
    analyzer_kwargs: dict
    precision: int
    cache_dir: str | None

    def __call__(self):
        import jaxpbsa
        from .online import OnlineMMPBSA

        jaxpbsa.set_precision(self.precision)
        jaxpbsa.enable_compilation_cache(self.cache_dir)
        return OnlineMMPBSA(
            XmlSerializer.deserialize(self.system_xml), self.topology,
            self.solute_idx, self.ligand_local_idx, self.ref_coords_A,
            **self.analyzer_kwargs)


def _worker_environment(mps_thread_percentage, mps_priority, cpu_threads):
    if isinstance(cpu_threads, bool) or int(cpu_threads) != cpu_threads or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if mps_thread_percentage is not None and (
        isinstance(mps_thread_percentage, bool)
        or int(mps_thread_percentage) != mps_thread_percentage
        or not 1 <= mps_thread_percentage <= 100
    ):
        raise ValueError("mps_thread_percentage must be an integer in [1, 100]")
    if mps_priority is not None and mps_priority not in (0, 1):
        raise ValueError("mps_priority must be 0 (normal) or 1 (below normal)")
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(int(cpu_threads))
    # Include importable user factories (and the source checkout) in the child.
    # This preserves sys.path[0], often scripts/: scripts must not shadow library
    # or standard-library modules imported by the worker.
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        [os.path.abspath(p or os.curdir) for p in sys.path]
        + env.get("PYTHONPATH", "").split(os.pathsep)))
    if mps_thread_percentage is not None or mps_priority is not None:
        control = shutil.which("nvidia-cuda-mps-control")
        if control is None:
            raise RuntimeError("MPS requested but nvidia-cuda-mps-control is unavailable")
        try:
            check = subprocess.run(
                [control], input="get_server_list\n", text=True, env=env,
                capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Cannot contact the MPS daemon") from exc
        if check.returncode:
            raise RuntimeError(
                "MPS requested but its daemon is unavailable. Start MPS before "
                "creating the OpenMM Context; use the same CUDA_MPS_PIPE_DIRECTORY.")
    if mps_thread_percentage is not None:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_thread_percentage))
    if mps_priority is not None:
        env["CUDA_MPS_CLIENT_PRIORITY"] = str(mps_priority)
    return env


class AsyncPBSAReporter:
    """OpenMM reporter: enqueue immutable solute snapshots without waiting for PBSA.

    Prefer ``from_system`` so the analyzer is constructed/preheated only in the
    worker. ``analyzer_factory`` must be picklable and return an OnlineMMPBSA-like
    callable in that process; do not pass an existing analyzer or JAX arrays.

    ``max_pending`` includes the active frame. Backlog raises by default, without
    silently losing a sample or blocking on GPU work. ``on_backlog='skip'`` is an
    explicit monitoring-only alternative, with a visible skip counter/warning.
    Errors (including on_violation='raise') surface at the next poll/report or
    close. CSVs are flushed in the worker as soon as each result completes.

    Use a context manager or ``close()`` to finish every submitted frame and reap
    the process. ``close(timeout=...)`` can be retried after a timeout; it never
    kills an active CUDA/MPS client. Context exit uses ``close_timeout`` (60 s by
    default; None waits indefinitely). Omitting close warns on pending frames at
    finalization, but cannot guarantee draining at interpreter shutdown.
    Async scheduling requires POSIX CLOCK_MONOTONIC shared by both processes.
    """

    def __init__(self, analyzer_factory, interval_steps: int, solute_idx, *,
                 out_csv: str | None = None, on_violation: str = "flag",
                 margin_min: float | None = None, max_pending: int = 2,
                 on_backlog: str = "raise", mps_thread_percentage: int | None = None,
                 mps_priority: int | None = None, cpu_threads: int = 1,
                 close_timeout: float | None = 60.0):
        if os.name != "posix" or not hasattr(time, "CLOCK_MONOTONIC"):
            raise RuntimeError("AsyncPBSAReporter requires POSIX CLOCK_MONOTONIC")
        if close_timeout is not None and (
            not np.isfinite(close_timeout) or close_timeout < 0
        ):
            raise ValueError("close_timeout must be finite and nonnegative, or None")
        if int(interval_steps) != interval_steps or interval_steps <= 0:
            raise ValueError("interval_steps must be a positive integer")
        if int(max_pending) != max_pending or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if on_backlog not in ("raise", "skip"):
            raise ValueError("on_backlog must be 'raise' or 'skip'")
        if on_violation not in ("flag", "raise"):
            raise ValueError("on_violation must be 'flag' or 'raise'")
        idx = np.asarray(solute_idx, dtype=int)
        if idx.ndim != 1 or not idx.size or (idx < 0).any() or np.unique(idx).size != idx.size:
            raise ValueError("solute_idx must contain distinct nonnegative indices")
        env = _worker_environment(mps_thread_percentage, mps_priority, cpu_threads)
        self.interval_steps, self.solute_idx = int(interval_steps), idx.copy()
        self.max_pending, self.on_backlog = int(max_pending), on_backlog
        self.close_timeout = close_timeout
        self.n_submitted = self.n_reports = self.n_skipped = self.n_flagged = 0
        self.total_pbsa_s = self.total_submit_s = self.max_latency_s = 0.0
        self.latest_result = None
        self._failure = None
        self._closed = self._closing = False
        # Admission is bounded by pending; reserve one slot for shutdown even
        # when the IPC thread has not yet picked up the first submitted frame.
        self._tasks = queue.Queue(maxsize=self.max_pending + 1)
        self._results = queue.Queue()
        parent_sock, child_sock = socket.socketpair()
        self._conn = Connection(parent_sock.detach())
        child_fd = child_sock.fileno()
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "jaxpbsa._online_worker", str(child_fd)],
                env=env, pass_fds=(child_fd,))
        except BaseException:
            self._conn.close()
            raise
        finally:
            child_sock.close()
        try:
            self._conn.send((analyzer_factory, dict(
                interval_steps=self.interval_steps, solute_idx=self.solute_idx,
                out_csv=None if out_csv is None else os.path.abspath(out_csv),
                on_violation=on_violation, margin_min=margin_min)))
            kind, data = _receive(self._conn, self._process)
            if kind != "ready":
                raise RuntimeError(f"PBSA worker initialization failed:\n{data}")
            self.ref_extent = float(data["ref_extent"])
            self.margin_min = float(data["margin_min"])
            self.warmup_time_s = float(data["warmup_time_s"])
            self.sizing = data["sizing"]
            self.worker_config = data["worker_config"]
        except BaseException:
            # Closing IPC lets initialization finish and the worker exit normally;
            # force-killing a CUDA client can also disrupt OpenMM under MPS.
            self._conn.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # Do not kill an initializing CUDA/MPS client.
            raise
        self._finalizer_state = {"pending": 0, "closing": False}
        self._finalizer = weakref.finalize(
            self, _finalize_reporter, self._tasks, self._finalizer_state)
        # A bound method here would keep the reporter alive forever and prevent
        # weakref.finalize from observing an abandoned reporter.
        self._thread = threading.Thread(
            target=AsyncPBSAReporter._pump,
            args=(self._conn, self._process, self._tasks, self._results),
            name="pbsa-ipc", daemon=True)
        self._thread.start()

    @classmethod
    def from_system(cls, system, topology, solute_idx, ligand_local_idx,
                    ref_coords_A, *, interval_steps: int, analyzer_kwargs=None,
                    cache_dir: str | None = None, **reporter_kwargs):
        """Serialize host parameters; build/compile the analyzer in the worker only.

        ``analyzer_kwargs`` accepts the same options as OnlineMMPBSA, e.g.
        pilot_coords_A, h, h_lig, padding, pb_params, sa_k and sa_points.
        The parent's array precision is forwarded explicitly to the worker.
        """
        from . import dtype

        factory = _OnlineFactory(
            XmlSerializer.serialize(system), topology,
            np.array(solute_idx, dtype=int, copy=True),
            np.array(ligand_local_idx, dtype=int, copy=True),
            np.array(ref_coords_A, dtype=np.float64, copy=True),
            dict(analyzer_kwargs or {}), np.dtype(dtype()).itemsize * 8, cache_dir)
        return cls(factory, interval_steps, solute_idx, **reporter_kwargs)

    @staticmethod
    def _pump(conn, process, tasks, results):
        try:
            while True:
                task = tasks.get()
                conn.send(task)
                kind, data = _receive(conn, process)
                if kind == "error":
                    raise RuntimeError(data)
                if task is None:
                    if kind != "closed":
                        raise RuntimeError(f"Unexpected shutdown response: {kind}")
                    return
                if kind != "result":
                    raise RuntimeError(f"Unexpected worker response: {kind}")
                results.put(data)
        except Exception as exc:
            # EOFError has an empty str(); preserve its type for a crashed child.
            results.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
        finally:
            conn.close()
            process.wait()

    @property
    def pending(self):
        """Submitted minus collected frames. Call poll() to refresh completions."""
        return self.n_submitted - self.n_reports

    def poll(self) -> list[dict]:
        """Collect completed results/errors without waiting for GPU computation."""
        completed = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(result, Exception):
                self._failure = str(result)
            else:
                self.n_reports += 1
                self._finalizer_state["pending"] = self.pending
                self.n_flagged += int(result.pop("_flagged"))
                self.total_pbsa_s += result["analysis_s"]
                self.max_latency_s = max(self.max_latency_s, result["latency_s"])
                self.latest_result = result
                completed.append(result)
        if not self._closing and not self._closed and self._process.poll() is not None:
            self._failure = self._failure or f"worker exited with code {self._process.returncode}"
        if self._failure is not None:
            raise RuntimeError(f"PBSA background worker failed:\n{self._failure}")
        return completed

    def describeNextReport(self, simulation):
        self.poll()
        steps = self.interval_steps - simulation.currentStep % self.interval_steps
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        start = time.perf_counter()
        self.poll()
        if self._closing or self._closed:
            raise RuntimeError("PBSA reporter is closing/closed")
        if self.pending >= self.max_pending:
            if self.on_backlog == "raise":
                raise RuntimeError(
                    f"PBSA backlog: {self.pending} unfinished frames (max_pending="
                    f"{self.max_pending}); increase the sampling interval or worker resources")
            self.n_skipped += 1
            if self.n_skipped == 1:
                warnings.warn("PBSA queue full: monitoring frame skipped; see n_skipped",
                              RuntimeWarning, stacklevel=2)
            return
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        # Queue serialization runs later in the IPC thread: take an owned copy now.
        coords = np.array(np.asarray(pos)[self.solute_idx], dtype=np.float64, copy=True)
        if coords.shape != (self.solute_idx.size, 3) or not np.isfinite(coords).all():
            raise ValueError("Invalid solute coordinates (shape or NaN/Inf)")
        extent = float(np.linalg.norm(coords.max(0) - coords.min(0)))
        if self.n_submitted == 0 and not (1 / 3 < extent / self.ref_extent < 3):
            raise ValueError("Solute extent differs from reference: check nm/Å and solute_idx")
        # POSIX CLOCK_MONOTONIC has the same epoch in parent and worker; do not
        # subtract unrelated per-process performance counters for queue latency.
        task = (coords, int(simulation.currentStep),
                float(state.getTime().value_in_unit(unit.picosecond)),
                time.clock_gettime(time.CLOCK_MONOTONIC))
        self._tasks.put_nowait(task)
        self.n_submitted += 1
        self._finalizer_state["pending"] = self.pending
        self.total_submit_s += time.perf_counter() - start

    def close(self, timeout: float | None = None):
        """Drain accepted frames and shut down normally, including after errors."""
        if self._closed:
            return
        if not self._closing:
            self._closing = True
            self._finalizer_state["closing"] = True
            # There is a spare queue slot: max_pending counts the active task too.
            self._tasks.put_nowait(None)
        deadline = None if timeout is None else time.monotonic() + timeout
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("PBSA is still draining; call close() again to finish")
        self._conn.close()
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            self._process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("PBSA worker is still exiting; retry close()") from exc
        self._closed = True
        self._finalizer.detach()
        self.poll()
        if self._process.returncode:
            raise RuntimeError(f"PBSA worker exited with code {self._process.returncode}")
        if self.pending:
            raise RuntimeError(f"PBSA closed with {self.pending} unaccounted frames")
        print(f"[AsyncPBSAReporter] submitted={self.n_submitted} completed={self.n_reports} "
              f"skipped={self.n_skipped} flagged={self.n_flagged} "
              f"max_latency={self.max_latency_s:.3f}s", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(timeout=self.close_timeout)

