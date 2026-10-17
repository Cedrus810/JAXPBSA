"""Private worker entry point. Environment is supplied before Python starts."""
from __future__ import annotations

from multiprocessing.connection import Connection
import os
import sys
import time
import traceback


def run(conn):
    sink = None
    try:
        factory, options = conn.recv()
        from .online import PBSAReporter

        analyzer = factory()
        sink = PBSAReporter(analyzer, **options)
        sink._CSV_FIELDS += ("analysis_s", "queue_wait_s", "latency_s")
        conn.send(("ready", dict(
            ref_extent=analyzer.ref_extent, margin_min=sink.margin_min,
            warmup_time_s=analyzer.warmup_time_s, sizing=getattr(analyzer, "sizing", {}),
            worker_config={key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "CUDA_MPS_CLIENT_PRIORITY", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        )))
        while True:
            task = conn.recv()
            if task is None:
                sink.close()
                sink = None
                conn.send(("closed", None))
                return
            coords, step, time_ps, submitted = task
            # Shared POSIX monotonic clock, also used for parent submission time.
            start = time.clock_gettime(time.CLOCK_MONOTONIC)
            out = analyzer(coords)
            out.update(step=step, time_ps=time_ps,
                       analysis_s=time.clock_gettime(time.CLOCK_MONOTONIC) - start,
                       queue_wait_s=start - submitted)
            out["latency_s"] = time.clock_gettime(time.CLOCK_MONOTONIC) - submitted
            sink.total_pbsa_s += out["analysis_s"]
            flagged_before = sink.n_flagged
            sink._record_result(out)
            out["_flagged"] = sink.n_flagged > flagged_before
            conn.send(("result", out))
    except (EOFError, BrokenPipeError):
        # Parent closed during initialization/cancellation. Exit normally after
        # any already submitted analysis has synchronized inside the analyzer.
        pass
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        if sink is not None:
            sink.close()
        conn.close()


if __name__ == "__main__":
    run(Connection(int(sys.argv[1])))
