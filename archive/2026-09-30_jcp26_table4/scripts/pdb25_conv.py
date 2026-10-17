"""JCP 545 (2026) 114452 Table 4 的 25 蛋白: surface="gaussian_gap" 单结构溶剂化能的网格收敛(LPB / --npb)。
结果写 pdb25_conv{_npb}{_tag}.pkl: result[pdb][h][phase] = {G, converged, newton_iter, cg_iter, residual, ...}。
原文是 NPB, 绝对值只作粗对照; 看的是 (E(h) − E(0.25)) / E(0.25) 的收敛形态。
每个 h 3 个随机相位; 盒子 = 原子心 max|x−COM| + 10 Å(≈ 原文「vdW 包围盒 + edge 8」)。
    python data/md/pdb25_conv.py [ID,ID,...] [--hs=1.0,0.5,0.25] [--npb] [--phases=3] [--tag=xx] [--sigma=] [--gcs_raw] [--radii=bondi]
"""
import os, sys, time, socket, pickle; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, jax, jaxpbsa
jaxpbsa.enable_compilation_cache()
from prep_pdb25 import TABLE4, VARIANTS
from jaxpbsa.pb import PBParams, make_frame_solver, make_grid, recenter_com

ids = next((a.split(",") for a in sys.argv[1:] if not a.startswith("--")), list(TABLE4))
HS = next((tuple(float(x) for x in a.split("=")[1].split(",")) for a in sys.argv[1:] if a.startswith("--hs=")),
          (1.0, 0.5, 0.25))
NPB = "--npb" in sys.argv
TAG = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--tag=")), "")
NPH = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--phases=")), 3)
OPT = {a.split("=")[0][2:]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
# 单因素开关(JCP26 没写 σ、半径集): --sigma=0.8  --gcs_raw(原文不归一化的 GCS 核)  --radii=bondi
P = PBParams(surface="gaussian_gap", nonlinear=NPB, gap_sigma=float(OPT.get("sigma", PBParams().gap_sigma)),
             gcs_normalize="gcs_raw" not in OPT)
print(f"host {socket.gethostname()}  devices {jax.devices()}  hs={HS}  {'NPB' if NPB else 'LPB'}  phases={NPH}  "
      f"sigma={P.gap_sigma} gcs_norm={P.gcs_normalize} radii={OPT.get('radii', 'npz')}", flush=True)
print(f"{'PDB':5s} " + " ".join(f"{'G(h=' + str(h) + ')':>11s}" for h in HS)
      + "  相对 0.25 的偏差(我们)        | JCP26 GEC(NPB) h=1/0.5/0.25 与相对偏差     | 我们/原文(0.25)  ms@0.5", flush=True)
OUT = os.path.join(ROOT, "data", "md", f"pdb25_conv{'_npb' if NPB else ''}{'_' + TAG if TAG else ''}.pkl")
# result[pdb][h][phase] = {G, converged, newton_iter, cg_iter, residual, residual_ref, ms} 或 {error};
# 某个 h / 相位失败只记 status, 继续 —— 不让一个 h 连累整行(第一轮 11/25 行因 h=1.0 失败整行丢了)。
# 每个蛋白算完就落盘。表里的 G(h) 只平均收敛的相位; 带 * = 有相位没收敛, nan = 全没收敛。
result = {}


def _one(o, dt):
    it = int(o["iters_solvent"])
    return {"G": float(o["g_pb"]), "converged": bool(o["converged"]), "newton_iter": it // 1000,
            "cg_iter": it % 1000, "residual": float(o["relres_solvent"]),
            "residual_ref": float(o["relres_ref"]), "ms": 1e3 * dt}


for pid in ids:
    z = np.load(os.path.join(ROOT, "data", "pdb25", "prep", f"{pid}.npz"))
    q, r, m = z["charge"], z["radii"], z["masses"]
    if "radii" in OPT:
        from openmm.app import PDBFile
        from jaxpbsa.openmm_io import assign_radii
        r = np.asarray(assign_radii(PDBFile(os.path.join(ROOT, "data", "pdb25", "prep", f"{pid}.pdb")).topology,
                                    OPT["radii"]), float)
    X = recenter_com(z["coords_A"], m, np.zeros(3))
    result[pid] = {}
    for h in HS:
        result[pid][h] = {}
        try:
            g = make_grid(X, h, padding=10.0, center=np.zeros(3))
            sv = make_frame_solver(g, r, P)
        except Exception as e:
            result[pid][h] = {s: {"error": f"{type(e).__name__} {str(e)[:200]}"} for s in range(NPH)}
            continue
        for s in range(NPH):
            sh = np.random.default_rng(90 + s).uniform(0, h, 3)
            try:
                t0 = time.perf_counter(); o = sv(X + sh, q, r); float(o["g_pb"])
                result[pid][h][s] = _one(o, time.perf_counter() - t0)
            except Exception as e:
                result[pid][h][s] = {"error": f"{type(e).__name__} {str(e)[:200]}"}
    with open(OUT, "wb") as fh:
        pickle.dump({"hs": HS, "npb": NPB, "phases": NPH, "result": result}, fh)

    G, flag, T, bad = {}, {}, {}, []
    for h in HS:
        ph = result[pid][h]
        ok = [d for d in ph.values() if d.get("converged")]
        G[h] = np.mean([d["G"] for d in ok]) if ok else np.nan
        flag[h] = "*" if len(ok) < NPH else " "
        ts = [d["ms"] for d in ph.values() if "ms" in d]
        T[h] = np.median(ts[1:] if len(ts) > 1 else ts) if ts else np.nan
        for s, d in ph.items():
            if not d.get("converged"):
                bad.append(f"h={h} ph{s}: " + (d["error"][:80] if "error" in d else
                           f"newton={d['newton_iter']} cg={d['cg_iter']} rr={d['residual']:.2e} "
                           f"rr_ref={d['residual_ref']:.1e} G={d['G']:.4g}"))
    ref = G[min(HS)]
    ours = "  ".join(f"{100 * (G[h] / ref - 1):+6.2f}%" for h in HS if h != min(HS))
    t4 = TABLE4[VARIANTS.get(pid, (pid,))[0]]
    theirs = f"{t4[1]:8.2f} {t4[2]:8.2f} {t4[3]:8.2f}  {100 * (t4[1] / t4[3] - 1):+6.2f}% {100 * (t4[2] / t4[3] - 1):+6.2f}%"
    print(f"{pid:5s} " + " ".join(f"{G[h]:10.2f}{flag[h]}" for h in HS) + f"  {ours:28s} | {theirs} | "
          f"{ref / t4[3]:6.3f}  {T.get(0.5, np.nan):6.1f}", flush=True)
    for b in bad:
        print(f"      未收敛 {b}", flush=True)
print("PDB25 DONE", flush=True)
