"""The AF3 worker backend must match the runtime that launches it.

``run_alphafold_batch.py`` runs in the AF3 environment, so it is read here rather than imported.
"""

import ast
from pathlib import Path

import pytest

from rfproteina.af3 import af3_apptainer

BATCH_SCRIPT = Path(af3_apptainer.__file__).with_name("run_alphafold_batch.py")


def _worker_default_backend() -> str:
    for node in ast.walk(ast.parse(BATCH_SCRIPT.read_text())):
        is_enum = isinstance(node, ast.Call) and node.args and ast.unparse(node.func) == "flags.DEFINE_enum"
        if is_enum and ast.literal_eval(node.args[0]) == "jax_backend":
            return ast.literal_eval(node.args[1])
    raise AssertionError("--jax_backend flag not found")


def _command(monkeypatch, runtime: str) -> list[str]:
    monkeypatch.setattr(af3_apptainer, "AF3_RUNTIME", runtime)
    for name in ("AF3_IMAGE", "AF3_PYTHON", "AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR"):
        monkeypatch.setenv(name, f"/placeholder/{name}")
    monkeypatch.delenv("AF3_JAX_BACKEND", raising=False)
    return af3_apptainer.build_apptainer_fold_command("/placeholder/manifest.json", 0, False)


def test_apptainer_worker_runs_on_cuda(monkeypatch):
    cmd = _command(monkeypatch, "apptainer")
    passed = [arg.split("=", 1)[1] for arg in cmd if arg.startswith("--jax_backend=")]
    assert (passed[-1] if passed else _worker_default_backend()) == "gpu"


@pytest.mark.parametrize(("env", "expected"), [(None, "mps"), ("cpu", "cpu")])
def test_native_worker_backend_is_explicit(monkeypatch, env, expected):
    monkeypatch.setattr(af3_apptainer, "AF3_RUNTIME", "native")
    for name in ("AF3_PYTHON", "AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR"):
        monkeypatch.setenv(name, f"/placeholder/{name}")
    if env is None:
        monkeypatch.delenv("AF3_JAX_BACKEND", raising=False)
    else:
        monkeypatch.setenv("AF3_JAX_BACKEND", env)
    cmd = af3_apptainer.build_apptainer_fold_command("/placeholder/manifest.json", 0, False)
    assert f"--jax_backend={expected}" in cmd
    assert "--flash_attention_implementation=xla" in cmd


def test_mistyped_runtime_is_rejected(monkeypatch):
    monkeypatch.setattr(af3_apptainer, "AF3_RUNTIME", "natve")
    with pytest.raises(ValueError, match="AF3_RUNTIME"):
        af3_apptainer.validate_runtime()
    with pytest.raises(ValueError, match="AF3_RUNTIME"):
        _command(monkeypatch, "natve")


@pytest.mark.parametrize("runtime", ["apptainer", "native"])
def test_known_runtimes_are_accepted(monkeypatch, runtime):
    monkeypatch.setattr(af3_apptainer, "AF3_RUNTIME", runtime)
    assert af3_apptainer.validate_runtime() == runtime


def test_mistyped_native_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(af3_apptainer, "AF3_RUNTIME", "native")
    for name in ("AF3_PYTHON", "AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR"):
        monkeypatch.setenv(name, f"/placeholder/{name}")
    monkeypatch.setenv("AF3_JAX_BACKEND", "metal")
    with pytest.raises(ValueError, match="AF3_JAX_BACKEND"):
        af3_apptainer.build_apptainer_fold_command("/placeholder/manifest.json", 0, False)
