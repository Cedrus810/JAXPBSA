"""validate_mmpbsa 的逐帧验收: 坏帧不能进均值。无需 GPU / Amber。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "validate_mmpbsa", Path(__file__).resolve().parents[1] / "scripts" / "validate_mmpbsa.py")
validate_mmpbsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_mmpbsa)

FLAG = 0.8


def frame(converged=True, margin=5.0, margin_lig=2.0):
    return {"converged": np.bool_(converged), "iters": np.array([9, 9, 9]),
            "margin_A": margin, "margin_lig_A": margin_lig}


def test_good_frame_passes():
    assert validate_mmpbsa.frame_violation(frame(), FLAG) is None


def test_shared_grid_nan_ligand_margin_passes():
    assert validate_mmpbsa.frame_violation(frame(margin_lig=float("nan")), FLAG) is None


@pytest.mark.parametrize("kw, word", [
    ({"converged": False}, "未收敛"),
    ({"margin": -0.3}, "margin_A"),
    ({"margin": 0.5}, "margin_A"),          # 没丢原子但低于 flag 阈值
    ({"margin_lig": -0.3}, "margin_lig_A"),
])
def test_bad_frames_rejected(kw, word):
    why = validate_mmpbsa.frame_violation(frame(**kw), FLAG)
    assert why is not None and word in why
