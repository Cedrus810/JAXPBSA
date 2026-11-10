#!/usr/bin/env python3
"""Stage 4x: run the unchanged v2.1 finalize (2026-10-11 packet scripts/finalize_allh_models.py,
finalize()) on this control packet: ff14SB name check, CA/omega classes, HA side, heavy atoms
identical to the stage-2x template except mutated side chains, non-bonded contacts.  Only the
module-level PACKET is pointed here.  openmm_dev python on 181 (numpy + XML only).
"""
import importlib.util, json, sys
from pathlib import Path
NEW = Path(__file__).resolve().parents[1]
SRC = NEW.parents[0] / "2026-10-11_public_data_execution_packet/scripts/finalize_allh_models.py"
spec = importlib.util.spec_from_file_location("fin", SRC); fin = importlib.util.module_from_spec(spec); spec.loader.exec_module(fin)
fin.PACKET = NEW
tmpl = fin.ff_templates()
ok = True
for name in ("PMI", "P12Q_L10A"):
    rp = NEW / "inputs/modeller_v21" / name / "finalize_record.json"
    r = fin.finalize(name, "v2mx1", tmpl)
    rp.write_text(json.dumps({"v2mx1": r, "finalize_source": str(SRC)}, indent=2) + "\n")
    ok &= r["passed"]
    print(name, json.dumps({k: r[k] for k in ("passed", "n_atoms", "mutated", "ha_worst_deg", "max_heavy_dev_vs_start_A", "closest_nonbonded", "failures")}))
sys.exit(0 if ok else 1)
