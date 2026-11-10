#!/usr/bin/env python3
"""Hou same-card online cost and recovery acceptance: summary (read-only; run on 183).

Plan: docs/HOU_ONLINE_COST_ACCEPTANCE_PLAN_20261028.md §1, §7.
    python summarize_hou_online_acc.py <job tag>      e.g. 19854
  A  RESULT lines of acceptance_online_<date>/full_<tag>.log: overhead = 1 - ns_day / median(baseline ns_day);
     MD extra wall vs PBSA busy time; latency vs the MD wall time of one 5000-step interval.
  B  status.json of accbase / acconline: production s/ns per chunk; online coverage of acconline through
     smallmol_records.online_rows (committed steps, no gap / duplicate), segments and their warm-up.
  C  scores/replay_npt/default/hou/<s>/ph74_gaff2/rep01/frames.csv against the online rows of the same frames.
Writes reports/hou_online_acc_{cost,production,replay}_<tag>.csv.
B is restricted to the first ACCEPTANCE_CHUNKS chunks (1 ns, the 10-28 acceptance run): smoke_extend.sh
later (10-28 19:27) extended acconline to chunk 12 as the extension smoke of the 6 x 10 ns plan
(fixed 2026-11-02, P3 reproduction).
"""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
A = P / "acceptance_online_20261028"
sys.path.insert(0, str(P.parents[1] / "scripts"))
from smallmol_records import online_rows, online_rows_by_frame

tag = sys.argv[1]
INTERVAL = 5000
ACCEPTANCE_CHUNKS = 10


def acceptance_status(st):
    """status.json as it was at the end of the 1-ns acceptance run (chunks 0..9)."""
    st = dict(st, chunks_done=min(st["chunks_done"], ACCEPTANCE_CHUNKS))
    st["stage_wall_s"] = {k: v for k, v in st["stage_wall_s"].items()
                          if not k.startswith("prod_chunk") or int(k.rsplit("_", 1)[1]) < ACCEPTANCE_CHUNKS}
    if "online" in st:
        st["online"] = [s for s in st["online"] if s["to_chunk"] <= ACCEPTANCE_CHUNKS]
    return st

rows = []
for line in (A / f"full_{tag}.log").read_text().splitlines():
    if line.startswith("RESULT "):
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        rows.append(dict(config=line.split()[1], **kv))
a = pd.DataFrame(rows)
for c in ("wall", "pbsa_total", "analyzer_warmup"):
    a[c] = a[c].str.rstrip("s")
for c in ("wall", "ns_day", "pbsa_total", "pbsa_ms_per_report", "analyzer_warmup", "drain_s", "total_ns_day",
          "max_latency_s", "steps", "expected", "submitted", "reports", "skipped"):
    a[c] = pd.to_numeric(a[c])
base = a[a.config == "baseline"]
b_ns, b_wall = base.ns_day.median(), base.wall.median()
a["scheme"] = np.where(a.config == "baseline", "baseline",
                       a["mode"] + np.where(a.worker_mps_percentage == "30", "+MPS30", ""))
a["overhead_pct"] = (1 - a.ns_day / b_ns) * 100
a["md_extra_s"] = a.wall - b_wall
a["frame_interval_wall_s"] = INTERVAL * b_wall / a.steps
cost = a[["scheme", "grid", "order", "ns_day", "overhead_pct", "md_extra_s", "pbsa_total", "pbsa_ms_per_report",
          "max_latency_s", "frame_interval_wall_s", "drain_s", "total_ns_day", "analyzer_warmup",
          "expected", "submitted", "reports", "skipped"]].round(3)

prod = []
for rid in ("accbase", "acconline"):
    out = A / f"md/hou/a1/ph74_gaff2/{rid}"
    st = acceptance_status(json.loads((out / "status.json").read_text()))
    ch_ns = st["chunk_steps"] * st["dt_ps"] / 1000
    walls = [w for k, w in sorted(st["stage_wall_s"].items()) if k.startswith("prod_chunk")]
    row = dict(run=rid, complete=st.get("complete"), chunks_done=st["chunks_done"],
               prod_s_per_ns_median=round(float(np.median(walls)) / ch_ns, 1),
               prod_s_per_ns_min=round(min(walls) / ch_ns, 1), prod_s_per_ns_max=round(max(walls) / ch_ns, 1),
               nvt_s=round(st["stage_wall_s"].get("nvt", float("nan")), 1),
               npt_s=round(st["stage_wall_s"].get("npt", float("nan")), 1))
    if rid == "acconline":
        got = online_rows(st)
        steps = [int(x["step"]) for x in got]
        row.update(online_valid=st.get("online_valid"), online_frames=len(got),
                   expected_frames=st["chunks_done"] * st["chunk_steps"] // st["frame_steps"],
                   steps_unique=len(set(steps)) == len(steps),
                   # an interrupted segment keeps state "running" and no counters; its rows are
                   # still counted above from the CSV (committed steps only)
                   segments=";".join(f"{s['from_chunk']}-{s['to_chunk']}:{s['state']}/{s['n_reports']}rep/"
                                     f"warm{s.get('warmup_time_s', float('nan')):.1f}s/"
                                     f"lat{s.get('max_latency_s', float('nan')):.2f}s" for s in st["online"]),
                   flagged=sum(s["n_flagged"] for s in st["online"]))
    prod.append(row)
prod = pd.DataFrame(prod)

rep = []
FIELDS = ("e_coul_rl", "e_lj_rl", "delta_g_pb", "delta_g_sa", "delta_g_mmpbsa")
for s in [f"a{i}" for i in range(1, 8)]:
    rdir = P / f"md/hou/{s}/ph74_gaff2/rep01"
    on, _ = online_rows_by_frame(rdir)
    off_dir = P / f"scores/replay_npt/default/hou/{s}/ph74_gaff2/rep01"
    off = list(csv.DictReader((off_dir / "frames.csv").open()))
    summ = json.loads((off_dir / "summary.json").read_text())
    online_grid = json.loads((rdir / "status.json").read_text())["online"][-1]["sizing"]
    d = {f: np.array([float(r[f]) - float(on[int(r["frame"])][f]) for r in off]) for f in FIELDS}
    rep.append(dict(sample=s, n_frames=len(off), flagged=sum(1 for r in off if r["flag"]),
                    same_grid=(list(summ["sizing"]["grid"]) == list(online_grid["grid"])
                               and list(summ["sizing"]["grid_lig"]) == list(online_grid["grid_lig"])),
                    **{f"max_abs_{f}": float(np.abs(d[f]).max()) for f in FIELDS},
                    mean_d_mmpbsa=float(d["delta_g_mmpbsa"].mean()), warmup_s=summ["warmup_s"],
                    ms_per_frame=summ["ms_per_frame_this_call"]["median"] if summ["ms_per_frame_this_call"] else None))
rep = pd.DataFrame(rep)

cost.to_csv(P / f"reports/hou_online_acc_cost_{tag}.csv", index=False)
prod.to_csv(P / f"reports/hou_online_acc_production_{tag}.csv", index=False)
rep.to_csv(P / f"reports/hou_online_acc_replay_{tag}.csv", index=False)
pd.set_option("display.width", 250)
print(f"baseline ns/day {base.ns_day.round(1).tolist()} median {b_ns:.1f}; wall median {b_wall:.2f} s")
print(cost.to_string(index=False))
print(prod.T.to_string())
print(rep.to_string(index=False))
