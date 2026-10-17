#!/usr/bin/env python3
"""Per-sample input preparation for the public-data peptide packet (P1 entry).

    python scripts/prep_peptide_sample.py --sample PMI --dry-run      # static plan only
    python scripts/prep_peptide_sample.py --sample PMI PMI-L10A P12Q_L10A
    python scripts/prep_peptide_sample.py --all

Reads `<packet>/sample_structure_audit.csv` and the template PDB, then for each
sample writes `<packet>/prepared/<peptide_name>/<model_id>/`:

    complex_dry.cif / complex_dry.pdb   solute with hydrogens, no solvent
    system_vacuum.xml                   ff14SB NoCutoff system (parameter check only)
    parameters.npz                      coords/charges/LJ/mbondi2 radii/indices/atom names
    prep_record.json                    written last; its presence marks completion

Chemistry and structure decisions follow packet protocol.md §1: free peptide
and receptor termini (no ACE/NME), GAI and all crystal waters removed, altloc A
kept, pH 7.4 hydrogens.  A PDBFixer-built residue that comes out with a D CA
gets its side chain mirrored through the N-CA-C plane (repair_built_chirality;
recorded in prep_record.json).  No minimisation is done here; the clash relief
happens in solvent in `run_peptide_md.py setup`.  The geometry report lists the closest
non-bonded heavy-atom contacts of rebuilt/mutated residues; it is a report, not
an acceptance threshold.

Re-running skips samples whose prep_record.json exists (use --force to redo).
A different template treatment or seed is a different --model-id.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform as host_platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/2026-10-11_public_data_execution_packet"
TEMPLATE = ROOT / "experiments/2026-10-10_peptide_samples/raw/mdm2_3EQS.pdb"
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
AA1 = {v: k for k, v in AA3.items()}
FORCEFIELD = "amber/ff14SB.xml"
PH = 7.4
DEFAULT_SEED = 20261011


# ---------------------------------------------------------------- template text
def read_template(path: Path) -> dict:
    """SEQRES, REMARK 465, HETATM residues and altloc-A coordinates from PDB text."""
    seqres, missing, het, alt = {}, [], {}, {}
    for ln in path.read_text().splitlines():
        if ln.startswith("SEQRES"):
            seqres.setdefault(ln[11], []).extend(ln[19:].split())
        elif ln.startswith("REMARK 465") and len(ln.split()) == 5 and ln.split()[2] in AA1:
            _, _, res, ch, num = ln.split()
            missing.append((ch, int(num), res))
        elif ln.startswith("HETATM"):
            key = f"{ln[21]}:{ln[17:20].strip()}{int(ln[22:26])}"
            het[ln[17:20].strip()] = het.get(ln[17:20].strip(), set()) | {key}
        if ln.startswith(("ATOM", "HETATM")) and ln[16] == "A":
            alt[(ln[21], int(ln[22:26]), ln[12:16].strip())] = (
                float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return dict(seqres={c: "".join(AA1[r] for r in v) for c, v in seqres.items()},
                missing=missing, het={k: sorted(v) for k, v in het.items()}, altloc_A=alt)


def plan_sample(row: dict, tpl: dict, packet: Path, model_id: str) -> dict:
    if row["template_pdb"] != "3EQS":
        raise ValueError(f"{row['peptide_name']}: template {row['template_pdb']} not handled by this entry")
    rc, pc = row["receptor_chain"], row["peptide_chain"]
    ref, seq = tpl["seqres"][pc], row["peptide_sequence"]
    if len(seq) != len(ref):
        raise ValueError(f"{row['peptide_name']}: length {len(seq)} != template peptide {len(ref)}")
    observed = {(c, n) for c, n, _ in tpl["missing"]}
    # template peptide residues with coordinates are 1..len-1 (B12 missing); mutate
    # observed positions, and set the rebuilt terminal residue to the sample residue
    mutations = [f"{AA3[a]}-{i + 1}-{AA3[b]}" for i, (a, b) in enumerate(zip(ref, seq))
                 if a != b and (pc, i + 1) not in observed]
    rebuilt = [dict(chain=c, resnum=n, template=r,
                    built=(AA3[seq[n - 1]] if c == pc else r)) for c, n, r in tpl["missing"]]
    if len(tpl["seqres"][rc]) != int(row["receptor_residues_planned"]):
        raise ValueError(f"receptor SEQRES length {len(tpl['seqres'][rc])} != planned {row['receptor_residues_planned']}")
    return dict(sample_id=row["sample_id"], peptide_name=row["peptide_name"], sequence=seq,
                template_peptide=ref, receptor_sequence=tpl["seqres"][rc],
                receptor_chain=rc, peptide_chain=pc, mutations=mutations, rebuilt_residues=rebuilt,
                removed_heterogens={k: len(v) for k, v in tpl["het"].items()},
                out=str(packet / "prepared" / row["peptide_name"] / model_id))


# ---------------------------------------------------------------- preparation
NON_SIDE_CHAIN = {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3"}
SIDE_CHAIN_STEREO = {"THR", "ILE"}


def repair_built_chirality(fixer, built: set) -> list[dict]:
    """Mirror the side chain of any PDBFixer-built residue whose CA came out D.

    PDBFixer 1.12 can return D residues: its template overlay (_overlayPoints) has no
    reflection check and placed B:12 mirrored in all three P2 samples, and its
    soft-core relief dynamics (soft.xml has no CA chirality term) inverted A:GLU25 in
    PMI-L10A / P12Q_L10A (handoff §9.2, scripts/diag_prep_stereo.py).  Reflecting the
    side chain through the N-CA-C plane keeps every bond length and angle and turns
    D into L; residues with a side-chain stereocentre are refused instead.
    Runs on the heavy-atom model, before hydrogens are added.
    """
    import numpy as np
    import openmm as mm
    from openmm import unit
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import peptide_geometry as geo

    xyz = np.asarray(fixer.positions.value_in_unit(unit.angstrom), dtype=float)
    done = []
    for res in fixer.topology.residues():
        if (res.chain.id, int(res.id)) not in built:
            continue
        at = {a.name: a.index for a in res.atoms()}
        if not all(k in at for k in ("N", "CA", "C", "CB")):
            continue
        before = geo.dihedral(*(xyz[at[k]] for k in ("N", "CA", "C", "CB")))
        if before < 0:
            continue
        label = f"{res.chain.id}:{res.name}{res.id}"
        if res.name in SIDE_CHAIN_STEREO:
            raise RuntimeError(f"{label} built with a D CA ({before:.1f} deg); mirroring would invert its CB")
        side = sorted(name for name in at if name not in NON_SIDE_CHAIN)
        xyz = geo.mirror_side_chain(xyz, at["N"], at["CA"], at["C"], [at[n] for n in side])
        after = geo.dihedral(*(xyz[at[k]] for k in ("N", "CA", "C", "CB")))
        done.append(dict(residue=label, chirality_before_deg=round(before, 1),
                         chirality_after_deg=round(after, 1), atoms_mirrored=side))
    if done:
        fixer.positions = unit.Quantity([mm.Vec3(*p) for p in xyz], unit.angstrom)
    return done


JUNCTIONS = ((("A", 25), ("A", 26)), (("B", 11), ("B", 12)))
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def check_c2(plan, top, xyz_A, system, mutated, v2_start, geo, v21=False) -> dict:
    """v2 prep check C2 (design §5): raises on failure, returns the record otherwise.

    - stereo is checked by the caller (all CA L, non-Pro omega trans)
    - every heavy-atom ff14SB bond/angle term touching the junction residues (A25, A26,
      B11, B12) or a mutated residue within 4 sigma at 298 K (vacuum system parameters)
    - junction / Pro-ring table (geo.junction_report)
    - no non-bonded pair (> 3 bonds) at those residues below 2.2 A (heavy) / 1.0 A (with H)
    - heavy atoms identical to the MODELLER start (<= 0.001 A, matched by residue + element)
      except the side chains of mutated residues
    v2.1 (design §11; every H was built by MODELLER, so the H checks cover all atoms):
    - every ff14SB bond/angle term containing an H within 4 sigma
    - HA of every non-Gly CA on the correct side (geo.ha_direction < 90 deg)
    - no pair with an H below 1.0 A anywhere
    - every atom (H included) identical to the v2.1 start file (<= 0.001 A, by name)
    """
    import numpy as np
    import openmm as mm
    atoms = list(top.atoms())
    focus_res = {r for j in JUNCTIONS for r in j} | mutated
    focus = [a.index for a in atoms if (a.residue.chain.id, int(a.residue.id)) in focus_res]
    lab = lambda i: f"{atoms[i].residue.chain.id}:{atoms[i].residue.name}{atoms[i].residue.id}.{atoms[i].name}"
    fails = []
    bonds, angles = [], []
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for n in range(f.getNumBonds()):
                i, j, r0, k = f.getBondParameters(n)
                bonds.append((i, j, r0._value, k._value))
        elif isinstance(f, mm.HarmonicAngleForce):
            for n in range(f.getNumAngles()):
                i, j, l, t0, k = f.getAngleParameters(n)
                angles.append((i, j, l, t0._value, k._value))
    is_h = [a.element is not None and a.element.symbol == "H" for a in atoms]
    # heavy-atom terms at the focus residues (design §5); v2.1 adds every H-containing term
    heavy_b = [b for b in bonds if not (is_h[b[0]] or is_h[b[1]])]
    heavy_a = [t for t in angles if not (is_h[t[0]] or is_h[t[1]] or is_h[t[2]])]
    out = geo.harmonic_outliers(np.asarray(xyz_A) / 10.0, heavy_b, heavy_a, focus)
    h_atoms = [a.index for a in atoms if is_h[a.index]]
    if v21:
        h_b = [b for b in bonds if is_h[b[0]] or is_h[b[1]]]
        h_a = [t for t in angles if is_h[t[0]] or is_h[t[1]] or is_h[t[2]]]
        out += geo.harmonic_outliers(np.asarray(xyz_A) / 10.0, h_b, h_a, h_atoms)
    for kind, idx, val, x0, lim in out:
        fails.append(f"{kind} {'-'.join(lab(i) for i in idx)} {val} (ff14SB {x0} +- {lim})")
    vals, jf = geo.junction_report(geo.residues_from_topology(top), xyz_A, JUNCTIONS)
    fails += jf
    ha = geo.ha_direction(geo.residues_from_topology(top), xyz_A)
    if v21:
        fails += [f"HA wrong side {l} {d:.1f} deg" for l, d in ha if d >= 90.0]
    clash_atoms = sorted(set(focus) | set(h_atoms)) if v21 else focus
    clashes, closest = geo.close_contacts(xyz_A, [(b1.index, b2.index) for b1, b2 in top.bonds()], clash_atoms, is_h)
    for d, i, j in clashes:
        fails.append(f"clash {d} A {lab(i)}--{lab(j)}")
    dev = 0.0
    if v21:
        # prep v2.1 changes nothing: every atom must match the start file.  Matched by
        # residue + element and nearest position, because OpenMM's PDB reader renames some
        # atoms on load (pdbNames.xml: N-terminal H1 -> H)
        start = {}
        for ln in Path(v2_start).read_text().splitlines():
            if ln.startswith("ATOM"):
                el = ln[76:78].strip() or ln[12:16].strip()[0]
                start.setdefault((ln[21], int(ln[22:26]), el), []).append(
                    [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        got = {}
        for a in atoms:
            got.setdefault((a.residue.chain.id, int(a.residue.id), a.element.symbol), []).append(a.index)
        bad = [k for k in set(start) | set(got) if len(start.get(k, [])) != len(got.get(k, []))]
        if bad:
            fails.append(f"atom counts differ from v2.1 start (chain, res, element): {sorted(bad)[:5]}")
        else:
            for k, idx in got.items():
                ref = np.asarray(start[k])
                for i in idx:
                    dev = max(dev, float(np.linalg.norm(ref - np.asarray(xyz_A[i]), axis=1).min()))
    else:
        # unchanged against the MODELLER start
        start = {}
        for ln in Path(v2_start).read_text().splitlines():
            if ln.startswith("ATOM") and (ln[76:78].strip() or ln[12:16].strip()[0]) != "H":
                start[(ln[21], int(ln[22:26]), ln[12:16].strip())] = np.array(
                    [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        by = {}
        for a in atoms:
            if not is_h[a.index]:
                by.setdefault((a.residue.chain.id, int(a.residue.id), a.element.symbol), []).append(a.index)
        for (c, r, name), v in start.items():
            if (c, r) in mutated and name not in BACKBONE:
                continue
            cand = by.get((c, r, name[0]), [])
            d = min((float(np.linalg.norm(np.asarray(xyz_A[i]) - v)) for i in cand), default=float("inf"))
            dev = max(dev, d)
    if dev > 0.001:
        fails.append(f"atoms moved vs MODELLER start: max {dev:.4f} A")
    if fails:
        raise RuntimeError(f"{'v2.1' if v21 else 'v2'} C2 check failed ({len(fails)}): {fails[:12]}")
    ha_worst = max(ha, key=lambda x: x[1]) if ha else (None, 0.0)
    return dict(passed=True, focus_residues=sorted(f"{c}:{r}" for c, r in focus_res),
                harmonic_terms_checked=("heavy-atom ff14SB bonds/angles touching focus residues"
                                        + (" + every H-containing bond/angle" if v21 else "") + ", 4 sigma at 298 K"),
                junction=vals, closest_nonbonded=dict(d_A=round(closest[0], 3),
                                                      pair=f"{lab(closest[1])}--{lab(closest[2])}"),
                ha_direction_worst=dict(residue=ha_worst[0], deg=round(ha_worst[1], 1),
                                        gate="< 90 deg" if v21 else "recorded only"),
                max_dev_vs_start_A=round(dev, 4),
                dev_scope="all atoms by name" if v21 else "heavy atoms except mutated side chains")


def load_v21(plan: dict, start: Path):
    """v2.1 (design §11): MODELLER already built mutations and every hydrogen; read as is.

    No PDBFixer, no addHydrogens, no minimisation.  The ff14SB templates must match the
    file's atoms exactly (createSystem fails otherwise); C2 checks every atom is unchanged.
    """
    from openmm import app
    pdb = app.PDBFile(str(start))
    mutated = {(plan["peptide_chain"], int(m.split("-")[1])) for m in plan["mutations"]}
    return pdb.topology, pdb.positions, mutated


def prepare(plan: dict, template: Path, tpl: dict, model_id: str, seed: int, v2_start: Path | None = None,
            protocol: str = "v1") -> dict:
    import numpy as np
    import openmm as mm
    from openmm import app, unit
    sys.path.insert(0, str(ROOT))
    from jaxpbsa.openmm_io import assign_radii, chain_indices, extract_nonbonded

    t0 = time.monotonic()
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    ff = app.ForceField(FORCEFIELD)
    if protocol == "v2.1":
        top, positions, mutated = load_v21(plan, v2_start)
        dev, found, missing_atoms, missing_terminals, variants = [], {}, {}, {}, []
        before = after = {r.name for r in top.residues()}
        chirality_repair = "none in v2.1: MODELLER built every atom (design §11)"
    else:
        top, positions, mutated, dev, found, missing_atoms, missing_terminals, before, after, \
            chirality_repair, variants = build_with_pdbfixer(plan, template, tpl, seed, v2_start, ff)
    return finish(plan, template, model_id, seed, v2_start, protocol, ff, top, positions, mutated, dev, found,
                  missing_atoms, missing_terminals, before, after, chirality_repair, variants, t0, out)


def build_with_pdbfixer(plan, template, tpl, seed, v2_start, ff):
    """v1 / v2: PDBFixer rebuild (v1) or MODELLER heavy-atom start (v2), mutations, pH 7.4 H."""
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    random.seed(seed)
    reference = mm.Platform.getPlatformByName("Reference")
    fixer = PDBFixer(filename=str(v2_start or template), platform=reference)

    dev = []
    if v2_start is None:
        # altloc: the loaded coordinates must be the altloc-A coordinates (protocol §1.2)
        pos = fixer.positions.value_in_unit(unit.angstrom)
        loaded = {(a.residue.chain.id, int(a.residue.id), a.name): pos[a.index] for a in fixer.topology.atoms()}
        dev = [max(abs(loaded[k][i] - v[i]) for i in range(3)) for k, v in tpl["altloc_A"].items() if k in loaded]
        absent = [k for k in tpl["altloc_A"] if k not in loaded]
        # identical text parse -> differences are float noise; a real altloc swap is >= 0.1 Å
        if absent or (dev and max(dev) > 1e-4):
            raise RuntimeError(f"altloc A not what PDBFixer loaded: max dev {max(dev):.4f} Å, absent {absent[:5]}")

    before = {r.name for r in fixer.topology.residues()}
    fixer.removeHeterogens(keepWater=False)
    after = {r.name for r in fixer.topology.residues()}
    fixer.findMissingResidues()
    found = {str(k): v for k, v in fixer.missingResidues.items()}
    chains = [c.id for c in fixer.topology.chains()]
    if plan["mutations"]:
        fixer.applyMutations(plan["mutations"], plan["peptide_chain"])
    mutated = {(plan["peptide_chain"], int(m.split("-")[1])) for m in plan["mutations"]}
    if v2_start is not None and fixer.missingResidues:
        raise RuntimeError(f"v2 start {v2_start} still has missing residues: {found}")
    for r in ([] if v2_start is not None else plan["rebuilt_residues"]):
        ci = chains.index(r["chain"])
        res = list(list(fixer.topology.chains())[ci].residues())
        key = (ci, 0) if r["resnum"] < int(res[0].id) else (ci, len(res))
        if key not in fixer.missingResidues:
            raise RuntimeError(f"PDBFixer did not report missing {r}; found {found}")
        fixer.missingResidues[key] = [r["built"]]
    fixer.findMissingAtoms()
    missing_atoms = {f"{k.chain.id}:{k.name}{k.id}": [a.name for a in v] for k, v in fixer.missingAtoms.items()}
    missing_terminals = {f"{k.chain.id}:{k.name}{k.id}": v for k, v in fixer.missingTerminals.items()}
    if v2_start is not None:
        # v2: MODELLER built A:25/B:12 with OXT; PDBFixer may only add mutated side chains
        extra = [k for k in fixer.missingAtoms if (k.chain.id, int(k.id)) not in mutated]
        if extra or fixer.missingTerminals:
            raise RuntimeError(f"v2 start has missing atoms outside mutated residues: {missing_atoms}, "
                               f"terminals {missing_terminals}")
    fixer.addMissingAtoms(seed=seed)
    if v2_start is None:
        chirality_repair = repair_built_chirality(
            fixer, mutated | {(r["chain"], r["resnum"]) for r in plan["rebuilt_residues"]})
    else:
        chirality_repair = "disabled in v2: any D residue fails the stereo check (design §1)"

    modeller = app.Modeller(fixer.topology, fixer.positions)
    random.seed(seed)
    variants = modeller.addHydrogens(ff, pH=PH, platform=reference)
    return (modeller.topology, modeller.positions, mutated, dev, found, missing_atoms, missing_terminals,
            before, after, chirality_repair, variants)


def finish(plan, template, model_id, seed, v2_start, protocol, ff, top, positions, mutated, dev, found,
           missing_atoms, missing_terminals, before, after, chirality_repair, variants, t0, out) -> dict:
    """Shared tail: sequence/termini/His/stereo checks, parameters, C2 (v2, v2.1), outputs."""
    import numpy as np
    import openmm as mm
    from openmm import app, unit
    from jaxpbsa.openmm_io import assign_radii, chain_indices, extract_nonbonded
    chain = {c.id: c for c in top.chains()}
    got = {cid: "".join(AA1[r.name] for r in c.residues()) for cid, c in chain.items()}
    if got.get(plan["peptide_chain"]) != plan["sequence"] or got.get(plan["receptor_chain"]) != plan["receptor_sequence"]:
        raise RuntimeError(f"sequence mismatch after rebuild: {got}")

    termini = {}
    bonded = {}
    for b1, b2 in top.bonds():
        bonded.setdefault(b1.index, []).append(b2); bonded.setdefault(b2.index, []).append(b1)
    for cid, c in chain.items():
        res = list(c.residues())
        n_atom = next(a for a in res[0].atoms() if a.name == "N")
        n_h = sorted(b.name for b in bonded[n_atom.index] if b.element.symbol == "H")
        termini[cid] = dict(n_residue=f"{res[0].name}{res[0].id}", n_H=n_h,
                            c_residue=f"{res[-1].name}{res[-1].id}",
                            has_OXT="OXT" in {a.name for a in res[-1].atoms()})
        if not (termini[cid]["has_OXT"] and len(n_h) == 3):
            raise RuntimeError(f"chain {cid} termini not free NH3+/COO-: {termini[cid]}")
    his = {f"{r.chain.id}:HIS{r.id}": sorted(a.name for a in r.atoms() if a.name in ("HD1", "HE2"))
           for r in top.residues() if r.name == "HIS"}

    # stereochemistry of the prepared model: all CA L, all non-Pro peptide bonds trans
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import peptide_geometry as geo
    stereo = geo.measure(geo.stereo_terms(geo.residues_from_topology(top)),
                         positions.value_in_unit(unit.angstrom))
    bad = geo.prepared_violations(stereo)
    if bad:
        raise RuntimeError(f"prepared model has D-CA or cis non-Pro peptide bonds: "
                           f"{[(b['label'], round(b['deg'], 1)) for b in bad]}")

    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    params = extract_nonbonded(system)
    rec = chain_indices(top, plan["receptor_chain"])
    lig = chain_indices(top, plan["peptide_chain"])
    xyz = np.asarray(positions.value_in_unit(unit.angstrom))

    c2 = (check_c2(plan, top, xyz, system, mutated, v2_start, geo, v21=(protocol == "v2.1"))
          if v2_start is not None else None)

    # geometry report: closest non-bonded heavy-atom pairs touching rebuilt/mutated residues
    focus_res = mutated | {(r["chain"], r["resnum"]) for r in plan["rebuilt_residues"]}
    atoms = list(top.atoms())
    heavy = np.array([a.index for a in atoms if a.element is not None and a.element.symbol != "H"])
    nbr = {a.index: set() for a in atoms}
    for b1, b2 in top.bonds():
        nbr[b1.index].add(b2.index); nbr[b2.index].add(b1.index)
    def within3(i):
        s = {i} | nbr[i]
        for _ in range(2):
            s |= {k for j in s for k in nbr[j]}
        return s
    contacts = []
    for i in heavy:
        a = atoms[i]
        if (a.residue.chain.id, int(a.residue.id)) not in focus_res:
            continue
        d = np.linalg.norm(xyz[heavy] - xyz[i], axis=1)
        excl = within3(int(i))
        for j, dj in zip(heavy, d):
            if int(j) in excl or (j < i and (atoms[j].residue.chain.id, int(atoms[j].residue.id)) in focus_res):
                continue
            b = atoms[j]
            contacts.append((float(dj), f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}.{a.name}",
                             f"{b.residue.chain.id}:{b.residue.name}{b.residue.id}.{b.name}"))
    contacts.sort()

    def tmp(name):
        return out / f".{name}.partial"
    with tmp("complex_dry.cif").open("w") as fh:
        app.PDBxFile.writeFile(top, positions, fh, keepIds=True)
    with tmp("complex_dry.pdb").open("w") as fh:
        app.PDBFile.writeFile(top, positions, fh, keepIds=True)
    tmp("system_vacuum.xml").write_text(mm.XmlSerializer.serialize(system))
    with tmp("parameters.npz").open("wb") as fh:
        np.savez(fh, coords_A=xyz, charge=params.charge, sigma_A=params.sigma,
                 epsilon_kcal=params.epsilon, radii_A=assign_radii(top), receptor_idx=rec, ligand_idx=lig,
                 atom_name=np.array([a.name for a in atoms]), res_name=np.array([a.residue.name for a in atoms]),
                 res_id=np.array([a.residue.id for a in atoms]), chain_id=np.array([a.residue.chain.id for a in atoms]))
    for name in ("complex_dry.cif", "complex_dry.pdb", "system_vacuum.xml", "parameters.npz"):
        os.replace(tmp(name), out / name)

    from importlib.metadata import version
    record = dict(
        **{k: plan[k] for k in ("sample_id", "peptide_name", "sequence", "template_peptide", "receptor_chain",
                                "peptide_chain", "mutations", "rebuilt_residues")},
        model_id=model_id, prep_seed=seed, template=str(template),
        heterogens_in_template=plan["removed_heterogens"], residue_names_removed=sorted(before - after),
        altloc_policy=("altloc A for all residues (protocol §1.2)" if v2_start is None else
                       "altloc A applied when building the MODELLER clean template (v2 design §1)"),
        altloc_A_max_dev_A=max(dev) if dev else None,
        pdbfixer_missing_residues=found, missing_atoms_added=missing_atoms, missing_terminals_added=missing_terminals,
        prep_protocol={"v1": "v1_pdbfixer", "v2": "v2_modeller", "v2.1": "v2.1_modeller_allh"}[protocol],
        v2_start=str(v2_start) if v2_start is not None else None, c2_checks=c2,
        termini=termini, his_states=his, chirality_repair=chirality_repair,
        stereo_check=dict(terms=len(stereo), violations=0,
                          rule="all CA L; all non-Pro omega trans (peptide_geometry.py)"), his_variants_returned=[v for v in variants if v],
        forcefield=FORCEFIELD, pH=PH, caps="none: free termini (protocol §1.1)",
        n_atoms=system.getNumParticles(), n_receptor_atoms=int(rec.size), n_peptide_atoms=int(lig.size),
        net_charge_complex=round(float(params.charge.sum()), 6),
        net_charge_receptor=round(float(params.charge[rec].sum()), 6),
        net_charge_peptide=round(float(params.charge[lig].sum()), 6),
        closest_nonbonded_heavy_contacts_of_rebuilt_or_mutated=[dict(d_A=round(d, 3), a=a, b=b) for d, a, b in contacts[:15]],
        minimisation="none here; done in solvent by run_peptide_md.py setup",
        software=dict(openmm=mm.__version__, pdbfixer=version("pdbfixer"), python=sys.version.split()[0]),
        host=host_platform.node(), wall_s=round(time.monotonic() - t0, 2),
        evidence_scope="prepared template model input; not structure/parameter acceptance",
    )
    (out / "prep_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def start_path(a, plan) -> Path | None:
    """MODELLER start for --protocol v2 / v2.1 (None for v1); model id must match the protocol."""
    if a.protocol == "v1":
        return None
    prefix = "v21m" if a.protocol == "v2.1" else "v2m"
    if not a.model_id.startswith(prefix):
        raise SystemExit(f"--protocol {a.protocol} needs model id {prefix}1/{prefix}2, got {a.model_id}")
    if a.protocol == "v2.1":
        return a.packet / "inputs" / "modeller_v21" / plan["peptide_name"] / f"start_{a.model_id}.pdb"
    return a.packet / "inputs" / "modeller_v2" / AA3[plan["sequence"][11]] / f"start_{a.model_id}.pdb"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", nargs="+", help="peptide_name or sample_id from sample_structure_audit.csv")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--model-id", default="model01")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PDBFixer/hydrogen seed; tied to --model-id")
    ap.add_argument("--force", action="store_true", help="redo even if prep_record.json exists")
    ap.add_argument("--protocol", choices=["v1", "v2", "v2.1"], default="v1",
                    help="v1: PDBFixer rebuilds A:25/B:12 (model01/model02); v2: start from the MODELLER "
                         "model inputs/modeller_v2/<RES12>/start_<model-id>.pdb (model-id v2m1/v2m2); "
                         "v2.1: MODELLER all-atom model inputs/modeller_v21/<peptide>/start_<model-id>.pdb "
                         "(model-id v21m1/v21m2), no PDBFixer, no addHydrogens")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no OpenMM/PDBFixer import")
    a = ap.parse_args()

    rows = list(csv.DictReader((a.packet / "sample_structure_audit.csv").open()))
    if not a.all:
        want = set(a.sample)
        rows = [r for r in rows if r["peptide_name"] in want or r["sample_id"] in want]
        unknown = want - {r["peptide_name"] for r in rows} - {r["sample_id"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown samples: {sorted(unknown)}")
    tpl = read_template(a.template)
    for row in rows:
        plan = plan_sample(row, tpl, a.packet, a.model_id)
        done = (Path(plan["out"]) / "prep_record.json").exists()
        if a.dry_run:
            v2 = start_path(a, plan)
            print(json.dumps({**plan, "already_prepared": done, "seed": a.seed, "protocol": a.protocol,
                              "v2_start": str(v2) if v2 else None, "v2_start_exists": bool(v2 and v2.exists())}))
            continue
        if done and not a.force:
            print(json.dumps({"skip_prepared": plan["peptide_name"], "out": plan["out"]}), flush=True)
            continue
        v2_start = start_path(a, plan)
        if v2_start is not None and not v2_start.exists():
            raise SystemExit(f"missing {a.protocol} start {v2_start}; run the MODELLER build/finalize scripts first")
        rec = prepare(plan, a.template, tpl, a.model_id, a.seed, v2_start, a.protocol)
        print(json.dumps({k: rec[k] for k in ("peptide_name", "model_id", "n_atoms", "net_charge_complex",
                                              "net_charge_peptide", "his_states", "chirality_repair", "c2_checks", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
