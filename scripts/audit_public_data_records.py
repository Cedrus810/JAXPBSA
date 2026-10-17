#!/usr/bin/env python3
"""Audit two existing public-data series; stdlib only, no molecular computation.

Reads retained source HTML and inventory. Writes derived audit files separately
from the historical inventory. Bcl-xL clusters are review candidates, not accepted
experimental measurements. MDMX labels are rebuilt from original Tables 2/3.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from curate_mdm2_cases import Tables, record

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/2026-10-10_peptide_samples"
OUT = ROOT / "experiments/2026-10-11_public_data_record_audit"
LI_DOI = "10.1016/j.apsb.2021.06.010"
BCL_DOIS = {"10.1110/ps.9.12.2528", "10.1017/s096183680000331x"}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def source_mdmx():
    path = INPUT / "raw/mdm2_article.html"
    html = path.read_text()
    parser = Tables()
    parser.feed(html)
    tables = [t for t in parser.tables if t and
              t[0] == ["Peptide sequence", "PMI‒MDM2", "PMI‒MDMX"]]
    t2 = next(t for t in tables if len(t[1]) == 8)
    t3 = next(t for t in tables if len(t[1]) == 4)
    points = {}
    loci = defaultdict(list)

    def add(seq, metric, cell, table, name=""):
        point = record(seq, "MDMX", metric, cell, table, name)
        # Assay assignment follows captions + Table 3 footnote a + Methods;
        # a pM value by itself is not a reason to call an assay ITC.
        point["assay"] = ("FP" if metric == "Ki" else
                          "ITC" if name == "PMI-M3" else "competition_SPR")
        key = (seq, metric)
        if key in points:
            assert point["value_nM"] == points[key]["value_nM"], key
        points[key] = point
        loci[key].append(table)

    for row in t2[2:]:
        add(row[0], "Kd", row[5], "Table 2")
        add(row[0], "Ki", row[7], "Table 2")
    current = None
    for row in t3[2:]:
        if len(row) == 6:
            current = row[0], row[1]
            add(current[1], "Kd", row[4], "Table 3", current[0])
        elif len(row) == 4:
            assert current
            add(current[1], "Ki", row[2], "Table 3", current[0])
        else:
            raise ValueError(f"Unexpected Table 3 row: {row}")
    assert len(points) == 55
    assert Counter(p["assay"] for p in points.values()) == {
        "competition_SPR": 27, "FP": 27, "ITC": 1}
    for key, p in points.items():
        p["source_loci"] = ";".join(loci[key])
        p["value_M"] = float(Decimal(str(p["value_nM"])) * Decimal("1e-9"))
        p["label_status"] = "source_table_reconciled_structure_and_chemistry_pending"
        p["uncertainty_note"] = (
            "For PMI-M3 the authors caution about ultratight ITC uncertainty and "
            "show a representative run; do not treat this as precise same-assay SPR ranking."
            if p["assay"] == "ITC" else "reported_plus_minus_definition_not_assumed")
    # Cross-check the historical extraction, without overwriting it.
    with (INPUT / "manifests/mdm2_mdmx_primary_measurements.csv").open() as f:
        previous = {(p["peptide_sequence"], p["metric"]): p
                    for p in csv.DictReader(f) if p["target"] == "MDMX"}
    assert points.keys() == previous.keys()
    for key, point in points.items():
        assert point["value_nM"] == float(previous[key]["value_nM"]), key
        assert point["assay"] == previous[key]["assay"], key
    return points, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    points, source = source_mdmx()
    bcl, mdmx = [], []
    with (INPUT / "manifests/candidate_inventory.csv").open() as f:
        for row in csv.DictReader(f):
            doi = row["DOI"].lower()
            if doi in BCL_DOIS:
                bcl.append(row)
            if doi == LI_DOI and row["Gene"] == "MDM4":
                mdmx.append(row)

    clusters = defaultdict(list)
    for row in bcl:
        # Assay is deliberately not merged: database assay labels conflict.
        # This key identifies possible repeat records pending source-locus audit.
        key = tuple(row[k] for k in ["Peptide_Sequence", "Protein_Sequence",
                    "Chemical_Modification", "Residue_Modification", "Nterminal",
                    "Cterminal", "Linear/Cyclic", "Cyclization_Method",
                    "Cyclization_Way", "assay_type", "affinity_relation"])
        key += (str(Decimal(row["affinity_M"]).normalize()),)
        clusters[key].append(row)
    bcl_rows = []
    for i, records in enumerate(clusters.values(), 1):
        for row in records:
            bcl_rows.append({
                "record_id": row["ID"], "paper_id": "PMID:11206074",
                "canonical_doi": "10.1110/ps.9.12.2528", "raw_doi": row["DOI"],
                "sequence": row["Peptide_Sequence"], "length": len(row["Peptide_Sequence"]),
                "raw_affinity": row["Affinity"], "parsed_value_M": row["affinity_M"],
                "raw_assay": row["Affinity_Assay"], "curated_assay": "",
                "n_terminus_raw": row["Nterminal"], "c_terminus_raw": row["Cterminal"],
                "chemical_modification_raw": row["Chemical_Modification"],
                "pdb_link_raw": row["PDB_ID"],
                "candidate_duplicate_cluster": f"Petros2000_candidate_{i:03d}",
                "cluster_record_count": len(records),
                "source_record_ids_in_cluster": ";".join(r["ID"] for r in records),
                "status": "hold_for_primary_table_and_assay_verification",
                "review": "same_paper_alias;possible_duplicate_not_independent_measurement",
            })

    mdmx_rows = []
    for row in mdmx:
        key = row["Peptide_Sequence"], row["assay_type"]
        point = points.get(key)
        flags = []
        if not point:
            flags.append("no_original_table_point_for_this_sequence_and_metric")
        else:
            indexed = Decimal(row["affinity_M"])
            correct = Decimal(str(point["value_nM"])) * Decimal("1e-9")
            if abs(indexed - correct) > max(abs(correct), abs(indexed)) * Decimal("1e-6"):
                flags.append("value_or_unit_conflict")
            matches_assay = {
                "ITC": "calorimetry" in row["Affinity_Assay"].lower(),
                "FP": "polarization" in row["Affinity_Assay"].lower(),
                "competition_SPR": "plasmon" in row["Affinity_Assay"].lower(),
            }[point["assay"]]
            if not matches_assay:
                flags.append("assay_conflict")
            if row["Affinity_Parameters"].lower() != point["metric"].lower():
                flags.append("normalized_metric_conflict")
        mdmx_rows.append({
            "record_id": row["ID"], "sequence": row["Peptide_Sequence"],
            "raw_affinity": row["Affinity"], "parsed_metric": row["assay_type"],
            "raw_assay": row["Affinity_Assay"], "parsed_value_M": row["affinity_M"],
            "normalized_metric_raw": row["Affinity_Parameters"],
            "normalized_value_nM_raw": row["nM"],
            "primary_sample_id": point["sample_id"] if point else "",
            "primary_metric": point["metric"] if point else "",
            "primary_value_nM": point["value_nM"] if point else "",
            "primary_assay": point["assay"] if point else "",
            "primary_loci": point["source_loci"] if point else "",
            "flags": ";".join(flags) or "matches_primary_label",
            "action": "use_unique_primary_measurement_not_database_row" if point
                      else "quarantine_unresolved_record",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8463443/",
        })

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "bclxl_candidate_duplicate_audit.csv", bcl_rows)
    write_csv(OUT / "mdmx_database_vs_primary.csv", mdmx_rows)
    for assay, filename in [("competition_SPR", "mdmx_primary_spr_kd.csv"),
                            ("FP", "mdmx_primary_fp_ki.csv"),
                            ("ITC", "mdmx_primary_itc_kd.csv")]:
        write_csv(OUT / filename, (p for p in points.values() if p["assay"] == assay))
    summary = {
        "scope": "two existing series; source-label audit only; no molecular computation",
        "bclxl": {
            "inventory_rows": len(bcl), "unique_sequences": len({r["Peptide_Sequence"] for r in bcl}),
            "candidate_duplicate_clusters_not_verified_measurements": len(clusters),
            "length_counts_unique_sequences": dict(Counter(len(s) for s in {r["Peptide_Sequence"] for r in bcl})),
            "source_method_and_values_status": "primary_PDF_methods_and_tables_pending",
        },
        "mdmx": {
            "inventory_rows": len(mdmx),
            "inventory_rows_labeled_itc": sum("calorimetry" in r["Affinity_Assay"].lower() for r in mdmx),
            "inventory_kd_rows_labeled_itc": sum(r["assay_type"] == "Kd" and "calorimetry" in r["Affinity_Assay"].lower() for r in mdmx),
            "primary_measurements_by_assay": dict(Counter(p["assay"] for p in points.values())),
            "audit_flag_counts": dict(Counter(flag for r in mdmx_rows for flag in r["flags"].split(";"))),
            "primary_source": source,
            "structure_and_chemistry_status": "pending_not_ready_for_MD",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
