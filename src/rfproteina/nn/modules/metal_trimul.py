"""Fused Metal kernels for the triangle multiplicative update on Apple Silicon (inference only, fp32).

The vanilla path (:class:`TriangleMultiplicativeUpdate`) is a chain of memory-bound passes over ``[L, L, 4*c]``-sized
tensors around two GEMM stages. This module keeps the GEMMs on MPS (``F.linear``, ``torch.bmm``) and fuses the
elementwise work between them into three kernels, written for ``torch.mps.compile_shader``:

``tm_gate_pack``
    ``a = P[:, :c] * sigmoid(P[:, 2c:3c]) * mask`` (and ``b`` likewise), written channel-major ``[c, L*L]`` so the
    contraction is a plain batched matmul with no permute copies.
``tm_ln_out``
    Reads the channel-major matmul result, applies LayerNorm over the channel axis and writes token-major
    ``[L*L, c]`` for the output projection.
``tm_gate_mul``
    ``out = sigmoid(G) * X``.

Selected automatically on MPS under ``torch.no_grad()`` with fp32 inputs; ``RFP_MPS_TRIMUL=0`` disables it.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// P: [tok, 4H] columns [a_p | b_p | a_g | b_g]; M: [tok]; A, B: [H, tok]
// grid: x = H, y = ceil(tok/32) * 8; threadgroup 32 x 8; tile = 32 channels x 32 tokens
kernel void tm_gate_pack(
    device const float* P [[buffer(0)]],
    device const float* M [[buffer(1)]],
    device float* A [[buffer(2)]],
    device float* B [[buffer(3)]],
    constant uint& LL [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 t [[thread_position_in_threadgroup]]) {
  threadgroup float ta[32][33];
  threadgroup float tb[32][33];
  const uint c0 = tg.x * 32, t0 = tg.y * 32;
  const ulong W = 4 * (ulong)H;
  for (uint r = t.y; r < 32; r += 8) {
    const uint tok = t0 + r;
    float a = 0.0f, b = 0.0f;
    if (tok < LL) {
      device const float* row = P + (ulong)tok * W;
      const float m = M[tok];
      const uint c = c0 + t.x;
      a = row[c] * (1.0f / (1.0f + exp(-row[2 * H + c]))) * m;
      b = row[H + c] * (1.0f / (1.0f + exp(-row[3 * H + c]))) * m;
    }
    ta[r][t.x] = a;
    tb[r][t.x] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = t.y; r < 32; r += 8) {
    const uint tok = t0 + t.x;
    if (tok < LL) {
      A[(ulong)(c0 + r) * LL + tok] = ta[t.x][r];
      B[(ulong)(c0 + r) * LL + tok] = tb[t.x][r];
    }
  }
}

// X: [H, tok] channel-major; Y: [tok, H]; LayerNorm over H (H <= 256). 16 tokens per threadgroup of 256 threads;
// grid x = ceil(tok/16) * 256.
kernel void tm_ln_out(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* Bi [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& LL [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tt [[thread_position_in_threadgroup]]) {
  threadgroup float tile[256][17];
  const uint t = tt.x;
  const uint t0 = tg.x * 16;
  for (uint ch = t; ch < H; ch += 256) {
    for (uint r = 0; r < 16; ++r) {
      const uint tok = t0 + r;
      tile[ch][r] = (tok < LL) ? X[(ulong)ch * LL + tok] : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = t / 16, lane = t % 16;
  float s = 0.0f;
  for (uint ch = lane; ch < H; ch += 16) s += tile[ch][r];
  for (uint o = 8; o > 0; o >>= 1) s += simd_shuffle_xor(s, o);
  const float mean = s / (float)H;
  float v = 0.0f;
  for (uint ch = lane; ch < H; ch += 16) {
    const float d = tile[ch][r] - mean;
    v += d * d;
  }
  for (uint o = 8; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
  const float rstd = rsqrt(v / (float)H + eps);
  const uint tok = t0 + r;
  if (tok < LL) {
    for (uint ch = lane; ch < H; ch += 16)
      Y[(ulong)tok * H + ch] = (tile[ch][r] - mean) * rstd * W[ch] + Bi[ch];
  }
}

kernel void tm_gate_mul(
    device const float* G [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* O [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint3 i3 [[thread_position_in_grid]]) {
  const ulong i = i3.x;
  if (i < n) O[i] = X[i] * (1.0f / (1.0f + exp(-G[i])));
}
"""

_lib = None


def _library():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SOURCE)
    return _lib


def enabled_for(z: torch.Tensor, c_z: int, c_hidden: int) -> bool:  # noqa: ARG001 (c_z: any width works)
    """True when the fused path applies: MPS, fp32, no autograd, channel sizes the kernels support."""
    return (
        os.environ.get("RFP_MPS_TRIMUL", "1") != "0"
        and not torch.compiler.is_compiling()  # inside torch.compile, inductor fuses the torch path itself
        and z.device.type == "mps"
        and z.dtype == torch.float32
        and z.dim() == 4
        and not torch.is_grad_enabled()
        and c_hidden % 32 == 0
        and c_hidden <= 256
    )


def tri_mult_metal(module, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forward of a ``TriangleMultiplicativeUpdate`` using the fused kernels. ``mask``: ``[B, L, L]``."""
    lib = _library()
    B, L = z.shape[:2]
    LL, H = L * L, module.c_hidden
    outgoing = module._direction == "outgoing"
    mask = mask.to(z.dtype).reshape(B, LL).contiguous()

    w_in = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight, module.linear_a_g.weight, module.linear_b_g.weight])
    b_in = None
    if module.linear_a_p.bias is not None:
        b_in = torch.cat([module.linear_a_p.bias, module.linear_b_p.bias, module.linear_a_g.bias, module.linear_b_g.bias])
    ln_w = module.layer_norm_out.weight
    ln_b = module.layer_norm_out.bias if module.layer_norm_out.bias is not None else torch.zeros_like(ln_w)

    outs = []
    for bi in range(B):
        zn = module.layer_norm_in(z[bi : bi + 1])  # [1, L, L, c_z]
        P = F.linear(zn, w_in, b_in).reshape(LL, 4 * H).contiguous()
        A = torch.empty(H, LL, device=z.device, dtype=z.dtype)
        Bm = torch.empty_like(A)
        n_tiles = (LL + 31) // 32
        lib.tm_gate_pack(P, mask[bi], A, Bm, LL, H, threads=(H, n_tiles * 8), group_size=(32, 8))
        del P
        A, Bm = A.view(H, L, L), Bm.view(H, L, L)
        prod = torch.bmm(A, Bm.transpose(1, 2)) if outgoing else torch.bmm(A.transpose(1, 2), Bm)  # [H, i, j]
        del A, Bm
        prod = prod.contiguous().view(H, LL)
        Xn = torch.empty(LL, H, device=z.device, dtype=z.dtype)
        n_groups = (LL + 15) // 16
        lib.tm_ln_out(prod, ln_w, ln_b, Xn, LL, H, float(module.layer_norm_out.eps), threads=(n_groups * 256,), group_size=(256,))
        del prod
        X = module.linear_z(Xn)  # [LL, c_z]
        del Xn
        G = module.linear_g(zn).reshape(LL, -1)
        out = torch.empty_like(X)
        lib.tm_gate_mul(G, X, out, X.numel(), threads=(X.numel(),), group_size=(256,))
        outs.append(out.view(1, L, L, -1))
    return torch.cat(outs) if B > 1 else outs[0]
