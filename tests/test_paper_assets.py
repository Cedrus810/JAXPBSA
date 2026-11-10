"""The paper must reject stale/corrupt sources instead of exporting plausible plots."""
import csv
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_assets", ROOT / "scripts/build_paper_assets.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


@pytest.fixture
def reports(tmp_path):
    for name in ["hou_ranking_20261030/samples.csv", "hou_sampling_20261030_verified/replicas.csv"] + [
        f"brd4_statistics_20261101_v2/{kind}_{grid}.csv"
        for kind in ("samples", "replicas") for grid in ("default", "h0.4")
    ]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paper.REPORTS / name, destination)
    return tmp_path


def edit(path, mutate):
    rows = paper.read_csv(path)
    mutate(rows)
    paper.write_csv(path, rows)


def test_report_means_rank_and_recovery_agree(reports):
    result = paper.check_inputs(reports)
    assert result["frames"] == 42000
    assert result["recovered_replicas"] == 3


def test_missing_replica_blocks_paper(reports):
    edit(reports / "hou_sampling_20261030_verified/replicas.csv", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="42 replicas"):
        paper.check_inputs(reports)


def test_changed_score_blocks_paper(reports):
    edit(reports / "hou_ranking_20261030/samples.csv", lambda rows: rows[0].update(score="-40"))
    with pytest.raises(ValueError, match="mean mismatch"):
        paper.check_inputs(reports)


def test_erased_historical_flags_block_paper(reports):
    edit(reports / "hou_sampling_20261030_verified/replicas.csv",
         lambda rows: [r.update(online_flagged="0") for r in rows])
    with pytest.raises(ValueError, match="flags must be retained"):
        paper.check_inputs(reports)


def test_historical_brd4_input_cannot_replace_corrected_result(reports):
    # Even when both summary and replicas are changed consistently, reject the old input.
    for kind, field in [("samples", "score"), ("replicas", "mean")]:
        edit(reports / f"brd4_statistics_20261101_v2/{kind}_default.csv",
             lambda rows: [r.update({field: "-18.09"}) for r in rows if r["sample"] == "BRD4-5"])
    with pytest.raises(ValueError, match="corrected alprazolam"):
        paper.check_inputs(reports)
