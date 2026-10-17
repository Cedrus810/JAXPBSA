"""Static 3EQS input audit: altlocs, GAI and crystal-water contacts.

Reads the PDB text only (stdlib + numpy); no force field, no hydrogens, no
minimisation.  Distances are heavy-atom distances in the deposited model.
Contact = heavy-atom distance <= 4.0 A; polar contact = N/O pair <= 3.5 A
(geometric only: no H placement, so not proof of an H-bond).
"""
import argparse, csv, json, math, os
from collections import defaultdict
import numpy as np

CONTACT, POLAR = 4.0, 3.5


def parse(path):
    atoms = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            atoms.append(dict(
                rec=ln[:6].strip(), name=ln[12:16].strip(), alt=ln[16].strip(),
                res=ln[17:20].strip(), chain=ln[21], resi=int(ln[22:26]),
                xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
                occ=float(ln[54:60]), b=float(ln[60:66]),
                el=(ln[76:78].strip() or ln[12:16].strip()[0])))
    return atoms


def rid(a):
    return f"{a['chain']}:{a['res']}{a['resi']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("outdir")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    atoms = parse(a.pdb)
    heavy = [x for x in atoms if x["el"] != "H"]
    X = np.array([x["xyz"] for x in heavy])

    # altlocs
    alt = defaultdict(lambda: defaultdict(list))
    for x in atoms:
        if x["alt"]:
            alt[rid(x)][x["alt"]].append(x["occ"])
    altrows = [dict(residue=r, altlocs=";".join(sorted(v)),
                    occupancies=";".join(f"{k}={v[k][0]:.2f}" for k in sorted(v)),
                    n_atoms_per_alt=";".join(f"{k}={len(v[k])}" for k in sorted(v)))
               for r, v in alt.items()]

    poly = [i for i, x in enumerate(heavy) if x["rec"] == "ATOM"]
    rec = [i for i in poly if heavy[i]["chain"] == "A"]
    pep = [i for i in poly if heavy[i]["chain"] == "B"]

    def near(i, idx, cut):
        d = np.linalg.norm(X[idx] - X[i], axis=1)
        return [(idx[j], float(d[j])) for j in np.where(d <= cut)[0]]

    def describe(lst):
        best = {}
        for j, d in lst:
            k = f"{rid(heavy[j])}.{heavy[j]['name']}{('/' + heavy[j]['alt']) if heavy[j]['alt'] else ''}"
            best[k] = min(d, best.get(k, 9e9))
        return ";".join(f"{k}@{v:.2f}" for k, v in sorted(best.items(), key=lambda t: t[1]))

    def polar(lst):
        return [(j, d) for j, d in lst if heavy[j]["el"] in ("N", "O") and d <= POLAR]

    # GAI
    gai = [i for i, x in enumerate(heavy) if x["res"] == "GAI"]
    gc = sorted({(j, d) for i in gai for j, d in near(i, poly, CONTACT)}, key=lambda t: t[1])
    gai_out = dict(
        atoms=[heavy[i]["name"] for i in gai],
        b_factors=[heavy[i]["b"] for i in gai],
        contacts_le_4A=describe(gc),
        polar_le_3p5A=describe([(j, d) for j, d in gc if heavy[j]["el"] in ("N", "O") and d <= POLAR]),
        min_dist_to_peptide=float(min(np.linalg.norm(X[pep] - X[i], axis=1).min() for i in gai)),
        peptide_residues_le_4A=sorted({rid(heavy[j]) for j, d in gc if heavy[j]["chain"] == "B"}),
    )

    # waters
    wrows = []
    for i, x in enumerate(heavy):
        if x["res"] != "HOH":
            continue
        cr, cp = near(i, rec, CONTACT), near(i, pep, CONTACT)
        pr, pp = polar(cr), polar(cp)
        dmin_r = float(np.linalg.norm(X[rec] - X[i], axis=1).min())
        dmin_p = float(np.linalg.norm(X[pep] - X[i], axis=1).min())
        cls = ("bridge_polar_both" if pr and pp else
               "peptide_polar_only" if pp else
               "peptide_contact_nonpolar" if cp else
               "receptor_only" if cr else "no_polymer_contact_le_4A")
        wrows.append(dict(water=f"{x['chain']}:HOH{x['resi']}", alt=x["alt"], occ=x["occ"], b=x["b"],
                          dmin_receptor=round(dmin_r, 2), dmin_peptide=round(dmin_p, 2), category=cls,
                          receptor_polar_le_3p5=describe(pr), peptide_polar_le_3p5=describe(pp)))

    with open(os.path.join(a.outdir, "3eqs_water_contacts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0]))
        w.writeheader(); w.writerows(wrows)
    with open(os.path.join(a.outdir, "3eqs_altlocs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(altrows[0]))
        w.writeheader(); w.writerows(altrows)
    cnt = defaultdict(int)
    for r in wrows:
        cnt[r["category"]] += 1
    summary = dict(pdb=os.path.abspath(a.pdb), contact_cutoff_A=CONTACT, polar_cutoff_A=POLAR,
                   n_waters=len(wrows), water_categories=dict(cnt), gai=gai_out,
                   altloc_residues=[r["residue"] for r in altrows],
                   polymer_residues={c: sorted({heavy[i]["resi"] for i in poly if heavy[i]["chain"] == c})
                                     for c in "AB"},
                   note="geometric distances only; no hydrogens; not an H-bond or necessity claim")
    json.dump(summary, open(os.path.join(a.outdir, "3eqs_contact_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
