import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()), reason="needs MPS"
)

from rfproteina.nn.modules import metal_trimul  # noqa: E402
from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (  # noqa: E402
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
@pytest.mark.parametrize(("batch", "n", "c_z", "c"), [(1, 37, 64, 64), (2, 100, 256, 256), (1, 16, 32, 32), (1, 45, 384, 256)])
def test_metal_matches_torch(cls, batch, n, c_z, c):
    torch.manual_seed(0)
    module = cls(c_z, c, bias=True, backend="torch").to("mps").eval()
    for p in module.parameters():  # the "final" init is zero; make the output non-trivial
        p.data.normal_(0, 0.1 if p.dim() > 1 else 0.05)
    z = torch.randn(batch, n, n, c_z, device="mps")
    mask = (torch.rand(batch, n, n, device="mps") > 0.1).float()
    with torch.no_grad():
        assert metal_trimul.enabled_for(z, c_z, c)
        fused = metal_trimul.tri_mult_metal(module, z, mask)
        reference = module._forward_torch(z, mask) if hasattr(module, "_forward_torch") else None
    if reference is None:
        import os

        os.environ["RFP_MPS_TRIMUL"] = "0"
        try:
            with torch.no_grad():
                reference = module(z, mask)
        finally:
            os.environ.pop("RFP_MPS_TRIMUL")
    torch.testing.assert_close(fused, reference, atol=1e-5, rtol=1e-4)


def test_disabled_with_grad_or_dtype():
    z = torch.randn(1, 8, 8, 32, device="mps")
    assert not metal_trimul.enabled_for(z, 32, 32)  # autograd on
    with torch.no_grad():
        assert not metal_trimul.enabled_for(z.half(), 32, 32)
        assert not metal_trimul.enabled_for(z, 32, 288)  # c_hidden above the 256-channel tile
