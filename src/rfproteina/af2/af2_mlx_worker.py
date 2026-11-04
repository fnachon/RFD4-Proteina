"""AF2 worker for Apple Silicon: ColabDesign on ``jax-mps``.

Run as ``python af2_mlx_worker.py MANIFEST.json`` *inside the JAX environment* (the one with ``colabdesign``,
``alphafold-colabfold`` and ``jax-mps``). It deliberately imports nothing from ``rfproteina`` so that environment
needs no other install. The oracle in :mod:`rfproteina.metrics.oracles.af2_mlx_oracle` builds the manifest and reads
the results.

Manifest: a JSON list of jobs, each with

- ``name``: output stem.
- ``kind``: ``"monomer"`` (one chain) or ``"binder"`` (target chain(s) from ``target_pdb`` plus one binder chain).
- ``sequence``: the (binder) sequence to predict.
- ``model_name``: e.g. ``model_1_ptm`` (monomer) or ``model_1_multimer_v3`` (multimer).
- ``params_dir``, ``num_recycles``, ``seed`` (optional), ``out_dir``.
- binder only: ``target_pdb``, ``target_chains`` (e.g. ``"A,B"``).

Per job it writes ``<name>.pdb``, ``<name>.npz`` (per-residue ``plddt`` in 0..1 and ``pae`` in Angstrom, ordered
target chains then binder) and ``<name>.json`` (scalar confidences, or ``{"error": ...}``). A failed job never stops
the rest.
"""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

_MODELS: dict[tuple, object] = {}


def _get_model(kind: str, multimer: bool, params_dir: str, num_recycles: int):
    from colabdesign import mk_afdesign_model

    key = (kind, multimer, params_dir, num_recycles)
    if key not in _MODELS:
        _MODELS[key] = mk_afdesign_model(
            protocol="binder" if kind == "binder" else "hallucination",
            num_recycles=num_recycles,
            data_dir=params_dir,
            use_multimer=multimer,
        )
    return _MODELS[key]


def _scalar(x):
    return None if x is None else float(np.asarray(x))


def run_job(job: dict) -> dict:
    name, kind = job["name"], job["kind"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = job["model_name"]
    multimer = "multimer" in model_name
    model_idx = int(model_name.split("_")[1]) - 1
    n_rec = int(job.get("num_recycles", 3))
    seq = "".join(c for c in job["sequence"].upper() if c.isalpha())

    model = _get_model(kind, multimer, job["params_dir"], n_rec)
    if kind == "binder":
        model.prep_inputs(
            pdb_filename=job["target_pdb"],
            chain=job["target_chains"],
            binder_len=len(seq),
            rm_target_seq=False,
            rm_target_sc=False,
        )
    else:
        model.prep_inputs(length=len(seq))
    kwargs = {"seed": int(job["seed"])} if job.get("seed") is not None else {}
    model.predict(seq=seq, models=[model_idx], num_recycles=n_rec, verbose=False, **kwargs)

    model.save_pdb(str(out_dir / f"{name}.pdb"))
    aux = model.aux
    np.savez(out_dir / f"{name}.npz", plddt=np.asarray(aux["plddt"], dtype=np.float32), pae=np.asarray(aux["pae"], dtype=np.float32))
    log = aux["log"]
    result = {"ptm": _scalar(log.get("ptm")), "model_name": model_name, "n_residues": int(np.asarray(aux["plddt"]).shape[0])}
    if kind == "binder":
        result["i_ptm"] = _scalar(log.get("i_ptm"))
        for key in ("min_ipsae", "max_ipsae", "avg_ipsae"):
            result[key] = _scalar(log.get(key))
        result["binder_len"] = len(seq)
    (out_dir / f"{name}.json").write_text(json.dumps(result))
    return result


def main(manifest_path: str) -> int:
    jobs = json.loads(Path(manifest_path).read_text())
    n_fail = 0
    for job in jobs:
        out = Path(job["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        try:
            run_job(job)
        except Exception as e:  # keep going; the caller reads the error
            n_fail += 1
            traceback.print_exc()
            (out / f"{job['name']}.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    return 1 if n_fail == len(jobs) else 0


if __name__ == "__main__":
    code = main(sys.argv[1])
    sys.stdout.flush()
    sys.stderr.flush()
    # jax-mps intermittently double-frees in its native teardown during a normal interpreter exit (SIGABRT after all
    # results are written). Every result is already on disk, so skip the teardown.
    os._exit(code)
