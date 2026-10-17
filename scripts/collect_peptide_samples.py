#!/usr/bin/env python3
"""Convert public PPIKB workbooks into traceable peptide inventories.

Uses the standard library: no changes to the production Python environment.
Input workbooks remain untouched. Lowercase sequence letters are preserved
(they may describe D residues); no assay values or missing conditions are inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AA = set("ACDEFGHIKLMNPQRSTVWY")
AFFINITY = re.compile(
    r"^\s*(Kd|KD|K_D|Ki|KI|IC50|IC₅₀)\s*([<>=~≈]+)?\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\s*"
    r"(?:[±]\s*[0-9.]+\s*)?(pM|nM|uM|μM|µM|mM|M)\s*$"
)


def workbook_rows(path: Path):
    """Read XLSX values, resolving sparse cells, shared strings and inline text."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall("s:si", NS)]
        wb = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {node.get("Id"): node.get("Target") for node in rels}
        for sheet in wb.findall("s:sheets/s:sheet", NS):
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = targets[rid]
            filename = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = ET.fromstring(archive.read(filename)).findall(".//s:row", NS)
            headers = None
            for node in rows:
                cells = {}
                for cell in node.findall("s:c", NS):
                    col = re.match(r"[A-Z]+", cell.get("r")).group()
                    index = 0
                    for ch in col:
                        index = index * 26 + ord(ch) - ord("A") + 1
                    value = cell.find("s:v", NS)
                    text = value.text or "" if value is not None else ""
                    if cell.get("t") == "s":
                        text = shared[int(text)]
                    elif cell.get("t") == "inlineStr":
                        text = "".join(cell.find("s:is", NS).itertext())
                    cells[index - 1] = text
                if not any(cells.values()):
                    continue
                values = [cells.get(i, "") for i in range(max(cells) + 1)]
                if headers is None:
                    headers = values
                    continue
                record = {h: cells.get(i, "") for i, h in enumerate(headers) if h}
                record["source_sheet"] = sheet.get("name")
                record["source_row"] = node.get("r")
                yield record


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate(row):
    record = dict(row)
    record["N-terminal_Modification"] = row.get("N-terminal_Modification", row.get("Nterminal", ""))
    record["C-terminal_Modification"] = row.get("C-terminal_Modification", row.get("Cterminal", ""))
    seq = row.get("Peptide_Sequence", "").strip()
    issues = []
    if not seq or not set(seq) <= AA:
        issues.append("noncanonical_or_lowercase_sequence")
    if row.get("Linear/Cyclic") != "Linear":
        issues.append("cyclic_or_unspecified")
    for key in ["Chemical_Modification", "Residue_Modification", "Cyclization_Method"]:
        if row.get(key, "").strip().lower() not in {"none", "no", ""}:
            issues.append(key.lower())
    if record["N-terminal_Modification"].lower() != "free":
        issues.append("n_terminal_requires_review")
    if record["C-terminal_Modification"].lower() != "free":
        issues.append("c_terminal_requires_review")
    match = AFFINITY.fullmatch(row.get("Affinity", ""))
    record.update(assay_type="", affinity_relation="", affinity_M="")
    if match:
        metric, relation, value, unit = match.groups()
        factor = {"pM": 1e-12, "nM": 1e-9, "uM": 1e-6, "μM": 1e-6,
                  "µM": 1e-6, "mM": 1e-3, "M": 1}[unit]
        metric = {"kd": "Kd", "k_d": "Kd", "ki": "Ki", "ic50": "IC50",
                  "ic₅₀": "IC50"}[metric.lower()]
        record.update(assay_type=metric, affinity_relation=relation or "unspecified",
                      affinity_M=format(float(value) * factor, ".12g"))
        if relation != "=":
            issues.append("censored_or_approximate_affinity")
    else:
        issues.append("affinity_needs_manual_parse")
    data_issues = []
    if match and row.get("Affinity_Parameters") and row["Affinity_Parameters"].lower() != metric.lower():
        data_issues.append("assay_type_conflicts_with_raw_affinity")
    if match and row.get("nM"):
        try:
            declared = float(row["nM"]) * 1e-9
            parsed = float(record["affinity_M"])
            if not abs(declared - parsed) <= 1e-6 * max(abs(declared), abs(parsed)):
                data_issues.append("normalized_value_conflicts_with_raw_affinity")
        except ValueError:
            data_issues.append("normalized_value_needs_manual_parse")
    record.update(
        sequence_length_observed=len(seq),
        chemistry_review=";".join(issues),
        data_review=";".join(data_issues),
        simple_linear_free_termini=str(not any(i for i in issues if i not in {
            "affinity_needs_manual_parse", "censored_or_approximate_affinity"})).lower(),
        experimental_conditions_status="source_paper_review_pending",
        structure_mapping_status="pdb_link_only_sequence_and_construct_unverified",
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/2026-10-10_peptide_samples")
    args = parser.parse_args()
    out = args.root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ["main", "branch", "structure"]:
        path = args.root / "raw" / f"ppikb_{name}.xlsx"
        if path.exists():
            rows = list(workbook_rows(path))
            tables[name] = rows
            write_csv(out / f"ppikb_{name}.csv", rows)
    primary = "main" if "main" in tables else "branch"
    annotated = [annotate(row) for row in tables[primary]]
    branch_ids = {r.get("ID") for r in tables.get("branch", [])}
    for row in annotated:
        row["source_dataset"] = f"PPIKB_{primary}"
        row["in_structure_linked_branch"] = str(row.get("ID") in branch_ids).lower()
    write_csv(out / "candidate_inventory.csv", annotated)
    simple = [r for r in annotated if r["simple_linear_free_termini"] == "true"
              and r["affinity_relation"] == "=" and r["assay_type"] == "Kd"
              and not r["data_review"]]
    write_csv(out / "linear_kd_candidates.csv", simple)
    groups = defaultdict(list)
    for row in simple:
        key = (row.get("DOI", ""), row.get("Protein_Name", ""),
               row.get("Protein_Sequence", ""), row.get("Gene", ""),
               row.get("Affinity_Assay", ""))
        groups[key].append(row)
    summary = []
    for (doi, name, protein_seq, gene, assay), members in groups.items():
        seqs = {r["Peptide_Sequence"] for r in members}
        affinities = [float(r["affinity_M"]) for r in members if float(r["affinity_M"]) > 0]
        summary.append({
            "protein_name": name, "gene": gene, "doi": doi,
            "assay": assay,
            "protein_length": len(protein_seq), "n_records": len(members),
            "n_unique_peptide_sequences": len(seqs),
            "n_structure_linked_records": sum(r["in_structure_linked_branch"] == "true" for r in members),
            "kd_min_M": min(affinities) if affinities else "",
            "kd_max_M": max(affinities) if affinities else "",
            "record_ids": ";".join(r["ID"] for r in members),
            "pdb_ids": ";".join(sorted({r.get("PDB_ID", "") for r in members} - {""})),
            "conditions_status": "same_paper_candidate_group_assay_and_construct_review_pending",
        })
    summary.sort(key=lambda r: (-r["n_unique_peptide_sequences"], -r["n_structure_linked_records"], r["doi"]))
    write_csv(out / "ranking_groups.csv", summary)
    stats = {
        "workbook_rows": {k: len(v) for k, v in tables.items()},
        "primary_inventory": primary,
        "unique_record_ids": len({r.get("ID") for r in annotated}),
        "unique_peptide_sequences": len({r.get("Peptide_Sequence") for r in annotated}),
        "assay_types": dict(Counter(r["assay_type"] or "manual_review" for r in annotated)),
        "rows_with_raw_normalized_field_conflicts": sum(bool(r["data_review"]) for r in annotated),
        "simple_linear_free_termini_exact_kd_rows": len(simple),
        "same_paper_target_groups_with_at_least_6_sequences": sum(r["n_unique_peptide_sequences"] >= 6 for r in summary),
        "independence_note": "Rows are measurements, not independent peptides; repeated assays and constructs remain explicit.",
    }
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print("Largest candidate groups:")
    for row in summary[:15]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
