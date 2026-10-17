"""Fail closed on unverifiable MPS benchmark clients, without needing a GPU."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import subprocess

import pytest

_spec = importlib.util.spec_from_file_location(
    "online_overhead", Path(__file__).resolve().parents[1] / "scripts" / "online_overhead.py")
online_overhead = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(online_overhead)


def mock_control(monkeypatch, responses):
    monkeypatch.setattr(online_overhead.shutil, "which", lambda name: "/fake/mps-control")
    calls = []

    def run(command, **kwargs):
        query = kwargs["input"].strip()
        calls.append(query)
        value = responses[query]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(returncode=0, stdout=value, stderr="")

    monkeypatch.setattr(online_overhead.subprocess, "run", run)
    return calls


def test_mps_checks_client_membership_across_servers(monkeypatch, capsys):
    calls = mock_control(monkeypatch, {
        "get_server_list": "10\n20\n", "get_client_list 10": "101\n999\n",
        "get_client_list 20": "202\n",
    })
    online_overhead._verify_mps_clients({101, 202})
    assert calls == ["get_server_list", "get_client_list 10", "get_client_list 20"]
    assert "MPS_VERIFIED servers=[10, 20] clients=[101, 202]" in capsys.readouterr().out


@pytest.mark.parametrize("clients", ["101\n", "202\n", ""])
def test_running_daemon_without_each_required_client_fails(monkeypatch, clients):
    mock_control(monkeypatch, {"get_server_list": "10", "get_client_list 10": clients})
    with pytest.raises(RuntimeError, match="PIDs.*missing"):
        online_overhead._verify_mps_clients({101, 202})


@pytest.mark.parametrize("response", ["", "MPS server not found", subprocess.TimeoutExpired("mps", 5)])
def test_mps_query_errors_and_empty_servers_fail(monkeypatch, response):
    mock_control(monkeypatch, {"get_server_list": response})
    with pytest.raises(RuntimeError, match="MPS"):
        online_overhead._verify_mps_clients({101})


def test_missing_control_fails(monkeypatch):
    monkeypatch.setattr(online_overhead.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        online_overhead._verify_mps_clients({101})
