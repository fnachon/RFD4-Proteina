import subprocess
from pathlib import Path

import pytest

from rfproteina.metrics.oracles import af3_oracle
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle, _fold_timeout_per_design_s


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("", None), ("  ", None), ("900", 900.0), ("0.5", 0.5)])
def test_timeout_per_design_parses(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AF3_FOLD_TIMEOUT_PER_DESIGN_S", raising=False)
    else:
        monkeypatch.setenv("AF3_FOLD_TIMEOUT_PER_DESIGN_S", raw)
    assert _fold_timeout_per_design_s() == expected


@pytest.mark.parametrize("raw", ["soon", "0", "-5", "inf", "nan"])
def test_timeout_per_design_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("AF3_FOLD_TIMEOUT_PER_DESIGN_S", raw)
    with pytest.raises(ValueError, match="AF3_FOLD_TIMEOUT_PER_DESIGN_S"):
        _fold_timeout_per_design_s()


def _dispatch(monkeypatch, per_design, n_designs, daemon):
    seen = {}
    monkeypatch.delenv("AF3_FOLD_TIMEOUT_PER_DESIGN_S", raising=False)
    if per_design is not None:
        monkeypatch.setenv("AF3_FOLD_TIMEOUT_PER_DESIGN_S", per_design)
    if daemon:
        monkeypatch.setenv("AF3_LAUNCHER_SOCKET_TEMPLATE", "/placeholder/af3_{rank}.sock")

        def send(sock, payload, timeout_s):
            seen["timeout"] = timeout_s
            return {"status": "ok"}

        monkeypatch.setattr(af3_oracle, "_send_to_daemon", send)
    else:
        monkeypatch.delenv("AF3_LAUNCHER_SOCKET_TEMPLATE", raising=False)
        monkeypatch.setattr(af3_oracle, "_resolve_gpu_device", lambda rank: 0)
        monkeypatch.setattr(af3_oracle, "_resolve_local_rank", lambda: 0)
        monkeypatch.setattr(af3_oracle, "build_apptainer_fold_command", lambda **kw: ["true"])

        def run(cmd, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            if kwargs.get("timeout") == 3600.0 and n_designs == 0:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(af3_oracle.subprocess, "run", run)
    AF3Oracle.__new__(AF3Oracle)._dispatch_fold(Path("/placeholder/manifest.json"), 0, n_designs)
    return seen["timeout"]


def test_daemon_timeout_defaults_to_two_minutes_per_design(monkeypatch):
    assert _dispatch(monkeypatch, None, 100, daemon=True) == 100 * 120.0


def test_daemon_timeout_follows_the_env_var(monkeypatch):
    assert _dispatch(monkeypatch, "600", 100, daemon=True) == 100 * 600.0
    assert _dispatch(monkeypatch, "600", 2, daemon=True) == 3600.0  # the floor stays


def test_direct_fold_is_unbounded_unless_configured(monkeypatch):
    assert _dispatch(monkeypatch, None, 4, daemon=False) is None
    assert _dispatch(monkeypatch, "900", 10, daemon=False) == 9000.0


def test_direct_fold_timeout_is_reported(monkeypatch):
    with pytest.raises(RuntimeError, match="timed out"):
        _dispatch(monkeypatch, "1", 0, daemon=False)
