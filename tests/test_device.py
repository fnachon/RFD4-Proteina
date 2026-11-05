import pytest

from rfproteina.utils.device import prefer_eager_trunk


@pytest.mark.parametrize(
    ("device", "n", "env", "expected"),
    [
        ("mps", 512, None, False),
        ("mps", 640, None, True),
        ("mps", 640, "-1", False),
        ("mps", 256, "128", True),
        ("cuda", 1024, None, False),
        ("cpu", 1024, None, False),
    ],
)
def test_prefer_eager_trunk(monkeypatch, device, n, env, expected):
    if env is None:
        monkeypatch.delenv("RFP_MPS_EAGER_ABOVE", raising=False)
    else:
        monkeypatch.setenv("RFP_MPS_EAGER_ABOVE", env)
    assert prefer_eager_trunk(n, device) is expected
