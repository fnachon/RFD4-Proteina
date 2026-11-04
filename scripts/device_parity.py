#!/usr/bin/env python
"""CPU-vs-accelerator parity check for ``rfd4-generate``.

Runs the same generation twice, once per device, and compares the designed structures. Random draws are taken from
one seeded CPU generator on both runs and moved to the device, so the two runs consume identical noise; remaining
differences come from the numerics of the device (kernels, attention path, accumulation order), not from the RNG.

Example (from the repository root, with the environment active)::

    python scripts/device_parity.py --seed 0 --out-root runs/parity -- \
        inputs=data/benchmarks/monomer subset_to_examples=monomer-short num_replicates=1 \
        generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode compile_lengths=null

Everything after ``--`` is passed to ``rfd4-generate`` as Hydra overrides (``out_dir``, ``seed`` and the device are
set here). Exit status is 0 when the superimposed CA RMSD is below ``--max-rmsd`` for every design, else 1.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def _child(device: str, seed: int, hydra_args: list[str]) -> None:
    """Run generation on ``device`` with all random draws taken from a seeded CPU generator."""
    os.environ["RFP_DEVICE"] = device
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    orig_randn = torch.randn

    def _split_size(size):
        return tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)

    def randn(*size, generator=None, dtype=None, device=None, **kwargs):
        out = orig_randn(*_split_size(size), generator=gen, dtype=dtype)
        return out.to(device=device) if device is not None else out

    def randn_like(x, *, dtype=None, **kwargs):
        return orig_randn(tuple(x.shape), generator=gen, dtype=dtype or x.dtype).to(x.device)

    torch.randn, torch.randn_like = randn, randn_like

    sys.argv = ["rfd4-generate", *hydra_args]
    from script_utils.entrypoints import generate

    generate()


def _compare(path_a: str, path_b: str) -> dict:
    import numpy as np
    from biotite.structure import rmsd, superimpose
    from atomworks.io.parser import parse

    def ca_array(p):
        arr = parse(p)["asym_unit"][0]
        ca = arr[arr.atom_name == "CA"]
        return ca[~np.isnan(ca.coord).any(axis=-1)]

    a, b = ca_array(path_a), ca_array(path_b)
    if len(a) != len(b):
        return {"error": f"different residue counts: {len(a)} vs {len(b)}"}
    raw = float(np.sqrt(((a.coord - b.coord) ** 2).sum(-1).mean()))
    fitted, _ = superimpose(a, b)
    fit = float(rmsd(a, fitted))
    per_res = np.linalg.norm(fitted.coord - a.coord, axis=-1)
    return {
        "n_res": len(a),
        "rmsd_raw": raw,
        "rmsd_fit": fit,
        "max_dev": float(per_res.max()),
        "seq_identity": float((a.res_name == b.res_name).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", nargs=2, default=["cpu", "mps"], metavar=("REF", "TEST"))
    parser.add_argument("--out-root", default="runs/parity")
    parser.add_argument("--max-rmsd", type=float, default=1.0, help="pass threshold on superimposed CA RMSD (A)")
    parser.add_argument("--child", metavar="DEVICE", help=argparse.SUPPRESS)
    parser.add_argument("hydra_args", nargs="*")
    args = parser.parse_args()

    if args.child:
        _child(args.child, args.seed, args.hydra_args)
        return 0

    out = {}
    for dev in args.devices:
        out_dir = Path(args.out_root) / dev
        cmd = [sys.executable, __file__, "--child", dev, "--seed", str(args.seed), "--",
               *args.hydra_args, f"seed={args.seed}", f"out_dir={out_dir}"]  # fmt: skip
        print(f"[{dev}] {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"[{dev}] generation failed (exit {rc})")
            return 2
        out[dev] = sorted(glob.glob(str(out_dir / "inference_samples" / "**" / "*_generated.cif*"), recursive=True))

    ref, test = args.devices
    if len(out[ref]) != len(out[test]) or not out[ref]:
        print(f"output mismatch: {len(out[ref])} vs {len(out[test])} designs")
        return 2
    ok = True
    for pa, pb in zip(out[ref], out[test], strict=True):
        res = _compare(pa, pb)
        print(f"{Path(pa).parent.name}: {res}")
        ok &= "error" not in res and res["rmsd_fit"] <= args.max_rmsd
    print("PARITY", "PASS" if ok else "FAIL", f"(superimposed CA RMSD <= {args.max_rmsd} A)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
