#!/usr/bin/env python3
"""Extract MDM2/MDMX measurements directly from Li et al. Tables 2 and 3.

PPIKB is a discovery index; the original experimental tables supply pilot labels.
The article HTML and original PPIKB workbook are retained beside the outputs.
"""
from html.parser import HTMLParser
from pathlib import Path
import csv
import json
import re

from collect_peptide_samples import write_csv

ROOT = Path(__file__).resolve().parents[1] / "experiments/2026-10-10_peptide_samples"
DOI = "10.1016/j.apsb.2021.06.010"
URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8463443/"


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.cell = None
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.active, self.rows = True, []
        if self.active and tag == "tr":
            self.row = []
        if self.active and tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.active and tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if self.active and tag == "tr":
            self.rows.append(self.row)
        if tag == "table":
            self.active = False
            self.tables.append(self.rows)


def record(seq, target, metric, text, table, name=""):
    vals = re.findall(r"\d+(?:\.\d+)?", text)
    value, uncertainty = map(float, vals[:2])
    factor = 0.001 if "pmol" in text else 1.0
    return {
        "sample_id": f"Li2021_{target}_{seq}_{metric}",
        "target": target, "peptide_name": name, "peptide_sequence": seq,
        "metric": metric, "value_nM": value * factor,
        "reported_uncertainty_nM": uncertainty * factor,
        "uncertainty_definition": "reported_plus_minus_definition_not_assumed",
        "raw_measurement": text, "source_table": table, "source_doi": DOI,
        "source_url": URL, "assay": "FP" if metric == "Ki" else ("ITC" if "pmol" in text else "competition_SPR"),
        "temperature_K": 298.15, "pH": 7.4,
        "salt_NaCl_M": 0.15 if metric == "Kd" and "pmol" not in text else "",
        "receptor_construct": "25-109" if target == "MDM2" else "24-108",
        "n_terminus": "free_assumption_from_synthesis_and_PPIKB_pending_mass_check",
        "c_terminus": "free_assumption_from_synthesis_and_PPIKB_pending_mass_check",
        "structure_status": "template_model_required_for_most_variants",
    }


def main():
    parser = Tables()
    parser.feed((ROOT / "raw/mdm2_article.html").read_text())
    t2 = next(t for t in parser.tables if len(t) > 10 and t[0] == ["Peptide sequence", "PMI‒MDM2", "PMI‒MDMX"]
              and len(t[1]) == 8)
    t3 = next(t for t in parser.tables if len(t) > 10 and t[0] == ["Peptide sequence", "PMI‒MDM2", "PMI‒MDMX"]
              and len(t[1]) == 4)
    records = {}
    for row in t2[2:]:
        seq = row[0]
        for target, offset in [("MDM2", 1), ("MDMX", 5)]:
            for metric, column in [("Kd", offset), ("Ki", offset + 2)]:
                r = record(seq, target, metric, row[column], "Table 2")
                records[(seq, target, metric)] = r
    current = None
    for row in t3[2:]:
        if len(row) == 6:
            current = (row[0], row[1])
            for target, column in [("MDM2", 2), ("MDMX", 4)]:
                r = record(current[1], target, "Kd", row[column], "Table 3", current[0])
                records[(current[1], target, "Kd")] = r
        elif len(row) == 4:
            for target, column in [("MDM2", 0), ("MDMX", 2)]:
                r = record(current[1], target, "Ki", row[column], "Table 3", current[0])
                records[(current[1], target, "Ki")] = r
    write_csv(ROOT / "manifests/mdm2_mdmx_primary_measurements.csv", records.values())
    pilot_seqs = ["TSFAEYWNLLSP", "TSFAEYWNLASP", "LSFAEYWNLASP", "TSFLEYWNLASP",
                  "TSFAEYWALASP", "TSFAEYWNLAMP", "TSFAEYWNLASQ", "LTFLEYWNLASP"]
    pilot = [records[(s, "MDM2", "Kd")] for s in pilot_seqs]
    write_csv(ROOT / "manifests/pilot_mdm2_8.csv", pilot)
    conflicts = []
    with (ROOT / "manifests/candidate_inventory.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["DOI"] != DOI or row["Gene"] != "MDM2" or row["assay_type"] != "Kd":
                continue
            key = (row["Peptide_Sequence"], "MDM2", "Kd")
            if key not in records or not row["affinity_M"]:
                continue
            original = records[key]
            indexed = float(row["affinity_M"]) * 1e9
            if abs(indexed - original["value_nM"]) > 1e-6 * max(indexed, original["value_nM"]):
                conflicts.append({"ppikb_id": row["ID"], "sequence": key[0],
                                  "ppikb_affinity_raw": row["Affinity"],
                                  "primary_kd_nM": original["value_nM"],
                                  "primary_table": original["source_table"],
                                  "review": "database_condition_or_value_mismatch_use_primary_table"})
    write_csv(ROOT / "manifests/mdm2_ppikb_source_conflicts.csv", conflicts)
    print(json.dumps({"primary_measurements": len(records),
                      "sequences_per_target": len({k[0] for k in records}),
                      "pilot_candidates": len(pilot), "database_rows_needing_source_review": len(conflicts)}, indent=2))


if __name__ == "__main__":
    main()
