#!/usr/bin/env python3
"""W3 cost table from run records (no logs re-parsed by hand, no new measurement); read-only.

    python3 experiments/2026-10-17_small_molecule_benchmark/scripts/cost_table.py

Paper outline §8 W3 / plan docs/BRD4_HOU_NEXT_STEPS_PLAN_20261101.md §6 ("从现有日志整理…缺测项留缺").
Sources: md/*/*/*/setup/setup_record.json (setup + minimisation), md/*/*/*/rep*/status.json (NVT, NPT,
every production chunk with the device of the context that ran it, online segments: PBSA busy time,
latency, warm-up), scores/**/summary.json (offline scoring: init incl. compile, ms/frame), prep_record.json
(ligand parametrisation), for the main package and the BRD4-5 / B1-repeat packages.  Offline scoring
records only the host, mapped to its GPU by HOST_GPU (from the node lines given by the user and RESULTS).
Writes reports/evidence_20261102/cost_{setup,md,online,scoring,prep}.csv; groups by device, never mixes
devices or precisions in one number.  Not measured anywhere (left missing): queue wait, file I/O,
strip time, Amber reference per-frame cost (only in logs / RESULTS text).
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / "reports/evidence_20261102"
PACKAGES = {"main": P, "brd4_5_alprazolam": P / "brd4_5_alprazolam_20261101",
            "b1_repeat": P / "b1_minimisation_repeat_20261101"}
HOST_GPU = {"yayoi07": "RTX A4000", "yayoi04": "RTX 3090", "yayoi10": "RTX 3090", "yayoi26": "RTX 3090",
            "yayoi28": "RTX 3090", "yayoi30": "RTX 3090", "yayoi36": "RTX 5080", "yayoi37": "RTX 5080",
            "yayoi43": "RTX 4080 SUPER", "kasuga02": "RTX 2080 Ti", "fulda": "RTX 5090"}


def gpu_of_host(h):
    return HOST_GPU.get((h or "").split(".")[0], f"unknown host {h}")


def short(name):
    return name.replace("NVIDIA ", "").replace("GeForce ", "")


def q(v):
    return dict(median=round(st.median(v), 2), min=round(min(v), 2), max=round(max(v), 2), n=len(v)) if v else {}


def write(name, rows):
    with (OUT / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    setup_rows, md, nvt, npt, online, scoring, prep = [], defaultdict(list), defaultdict(list), defaultdict(list), \
        defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    for pkg, root in PACKAGES.items():
        for rec in sorted(root.glob("md/*/*/*/setup/setup_record.json")):
            r = json.loads(rec.read_text())
            setup_rows.append(dict(package=pkg, target=r["target"], sample=r["sample"], model=r["model_id"],
                                   n_atoms=r["n_atoms"], gpu=short(r["device"]["DeviceName"]), host=r.get("host"),
                                   setup_wall_s=r["wall_s"], minimisation_wall_s=r["minimisation"]["wall_s"]))
        for sp in sorted(root.glob("md/*/*/*/rep*/status.json")):
            s = json.loads(sp.read_text())
            if "chunk_steps" not in s:
                continue
            ns = s["chunk_steps"] * s["dt_ps"] / 1000
            ctx = [c for c in s.get("contexts", []) if c["stage"] == "prod"]
            starts = sorted(((c["chunk"] or 0), short(c["device"]["DeviceName"])) for c in ctx)
            mode = "online" if s.get("online_required") else "offline"
            for k, w in s["stage_wall_s"].items():
                if k in ("nvt", "npt"):
                    dev = next((short(c["device"]["DeviceName"]) for c in s.get("contexts", []) if c["stage"] == k), "?")
                    (nvt if k == "nvt" else npt)[(pkg, s["target"], dev)].append(w)
                elif k.startswith("prod_chunk_"):
                    i = int(k.rsplit("_", 1)[1])
                    dev = [d for c0, d in starts if c0 <= i][-1] if starts else "?"
                    md[(pkg, s["target"], dev, mode)].append((w / ns, w))
            for seg in s.get("online", []):
                if seg.get("n_reports"):
                    dev = [d for c0, d in starts if c0 <= seg["from_chunk"]][-1] if starts else "?"
                    online[(pkg, s["target"], dev, seg.get("mps_thread_percentage"))].append(seg)
        for sm in sorted(root.glob("scores/**/summary.json")):
            d = json.loads(sm.read_text())
            rel = sm.relative_to(root).parts
            kind = rel[1]   # default, h0.4, start_default, replay_npt, window10_default, smoke
            if kind == "smoke":
                continue
            if kind == "replay_npt":
                kind = f"replay_npt/{rel[2]}"
            key = (pkg, d.get("target", rel[-4] if len(rel) > 4 else ""), kind, gpu_of_host(d.get("host")))
            ms = (d.get("ms_per_frame_this_call") or {}).get("median") or d.get("ms_score")
            if ms is not None:
                scoring[key]["ms_per_frame"].append(ms)
            if d.get("init_incl_warmup_s") is not None:
                scoring[key]["init_s"].append(d["init_incl_warmup_s"])
            scoring[key]["frames"].append(d.get("n_frames_scored", 1))
        for pr in sorted(root.glob("prepared/*/*/*/prep_record.json")):
            r = json.loads(pr.read_text())
            lig = r.get("ligand_prep") or r.get("ligand") or {}
            wall = lig.get("wall_s") or {}
            prep.append(dict(package=pkg, record=str(pr.relative_to(root)), host=(r.get("tools") or {}).get("host"),
                             total_wall_s=r.get("wall_s"), antechamber_s=wall.get("antechamber"),
                             n_ligand_atoms=lig.get("n_atoms")))

    write("cost_setup.csv", setup_rows)
    rows = []
    for (pkg, t, dev, mode), v in sorted(md.items()):
        rows.append(dict(package=pkg, target=t, gpu=dev, stage=f"production ({mode})", unit="s/ns",
                         **q([x[0] for x in v]), wall_h_total=round(sum(x[1] for x in v) / 3600, 2)))
    for (pkg, t, dev), v in sorted(nvt.items()):
        rows.append(dict(package=pkg, target=t, gpu=dev, stage="NVT 0.1 ns", unit="s", **q(v)))
    for (pkg, t, dev), v in sorted(npt.items()):
        rows.append(dict(package=pkg, target=t, gpu=dev, stage="NPT 1 ns (online: also the pilot)", unit="s", **q(v)))
    write("cost_md.csv", rows)
    rows = []
    for (pkg, t, dev, mps), segs in sorted(online.items(), key=lambda x: str(x[0])):
        per = [1000 * x["total_pbsa_s"] / x["n_reports"] for x in segs]
        rows.append(dict(package=pkg, target=t, gpu=dev, mps_thread_percentage=mps, segments=len(segs),
                         reports=sum(x["n_reports"] for x in segs), flagged=sum(x["n_flagged"] for x in segs),
                         skipped=sum(x.get("n_skipped", 0) for x in segs),
                         pbsa_ms_per_report_median=round(st.median(per), 1),
                         max_latency_s=round(max(x["max_latency_s"] for x in segs), 3),
                         warmup_s_median=round(st.median(x["warmup_time_s"] for x in segs), 1),
                         warmup_s_max=round(max(x["warmup_time_s"] for x in segs), 1)))
    write("cost_online.csv", rows)
    rows = []
    for (pkg, t, kind, gpu), d in sorted(scoring.items()):
        rows.append(dict(package=pkg, target=t, scoring=kind, gpu=gpu, runs=len(d["frames"]), frames=sum(d["frames"]),
                         **{f"ms_per_frame_{k}": v for k, v in q(d["ms_per_frame"]).items() if k != "n"},
                         **{f"init_incl_compile_s_{k}": v for k, v in q(d["init_s"]).items() if k != "n"}))
    write("cost_scoring.csv", rows)
    write("cost_prep.csv", prep)
    for name in ("cost_setup.csv", "cost_md.csv", "cost_online.csv", "cost_scoring.csv", "cost_prep.csv"):
        print(name, sum(1 for _ in (OUT / name).open()) - 1, "rows")


if __name__ == "__main__":
    main()
