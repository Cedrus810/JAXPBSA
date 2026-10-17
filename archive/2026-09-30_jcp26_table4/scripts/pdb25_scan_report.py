"""单因素扫描汇总: 读 pdb25_conv_npb_scan_*.pkl, 每个配置对 Table 4 同 h 列算 ratio 与 (我们−原文)/SASA。
    python data/md/pdb25_scan_report.py [h=0.5]
"""
import os, sys, glob, pickle
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from prep_pdb25 import TABLE4, VARIANTS
from jaxpbsa.sa import sasa

H = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("h=")), 0.5))
COL = {1.0: 1, 0.5: 2, 0.25: 3}[H]
_sa = {}


def area(pid):
    if pid not in _sa:
        z = np.load(os.path.join(ROOT, "data", "pdb25", "prep", f"{pid}.npz"))
        _sa[pid] = float(sasa(z["coords_A"], z["radii"], n_points=240))
    return _sa[pid]


print(f"h={H}  ratio = 我们/原文;  Δ = 我们 − 原文 (kcal/mol);  Δ/SASA 单位 kcal per 1000 Å²")
for f in sorted(glob.glob(os.path.join(ROOT, "data", "md", "pdb25_conv_npb_scan_*.pkl"))):
    R = pickle.load(open(f, "rb"))["result"]
    tag = os.path.basename(f)[len("pdb25_conv_npb_scan_"):-4]
    rat, dsa, cells = [], [], []
    for pid, byh in R.items():
        ok = [d["G"] for d in byh.get(H, {}).values() if d.get("converged")]
        t = TABLE4[VARIANTS.get(pid, (pid,))[0]][COL]
        if not ok:
            cells.append(f"{pid}:nan"); continue
        g = float(np.mean(ok))
        rat.append(g / t); dsa.append(1e3 * (g - t) / area(pid)); cells.append(f"{pid}:{g / t:.3f}")
    rat, dsa = np.array(rat), np.array(dsa)
    print(f"{tag:12s} ratio 中位 {np.median(rat):.3f} [{rat.min():.3f},{rat.max():.3f}]  "
          f"Δ/SASA 中位 {np.median(dsa):+6.2f} (sd {dsa.std():.2f})  | " + " ".join(cells))
