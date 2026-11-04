"""AlphaFold 2 oracle for Apple Silicon: ColabDesign on ``jax-mps``.

Unlike :class:`~rfproteina.metrics.oracles.af2_oracle.AF2Oracle` (CUDA container + daemon), this runs
:mod:`rfproteina.af2.af2_mlx_worker` with the interpreter of a separate JAX environment, so it works on a Mac. Set
``AF2_PYTHON`` (that environment's ``python``) and ``AF2_DIR`` (the directory holding ``params_model_*.npz``).

What it folds:

- one protein chain: a single-chain prediction (default ``model_1_ptm``), single sequence;
- a *binder* (the chain of interest) plus one or more other protein chains: ColabDesign's ``binder`` protocol
  (default ``model_1_multimer_v3``). **The other chains are given as a backbone template** (their coordinates in the
  input), exactly as in the Proteina-Complexa AF2 evaluation and the AF2 PPI oracle's ``sa_mask('C_CRD')`` setting;
  only the binder is predicted from its sequence. The interface metrics are therefore binder-vs-templated-target.

Chains that are not protein (ligands, DNA, RNA) are not supported by AF2 and raise a ``ValueError``. There is no MSA
(single sequence), matching the existing AF2 oracle's database-free input.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import biotite.structure.io.pdb as pdbio
import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file

from rfproteina.metrics.oracles.oracle_inputs import ChainEntity, chain_entities
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

_WORKER = Path(__file__).resolve().parents[2] / "af2" / "af2_mlx_worker.py"


class AF2MLXOracle(OracleMetric):
    """Refold designs with AF2 (ColabDesign / jax-mps) and report AF3-style confidence metrics.

    Args:
        oracle_name: Label used in output filenames.
        python: Interpreter of the JAX environment (falls back to ``$AF2_PYTHON``).
        params_dir: Directory with ``params_model_*.npz`` (falls back to ``$AF2_DIR``).
        model_name: ``model_{1..5}_ptm`` or ``model_{1..5}_multimer_v3``. ``None`` picks ``model_1_ptm`` for one
            chain and ``model_1_multimer_v3`` for a binder complex.
        num_recycles: AF2 recycles.
        seed: Optional model seed.
        timeout_s: Per-batch subprocess timeout.
    """

    def __init__(
        self,
        oracle_name: str = "af2_mlx",
        python: str | None = None,
        params_dir: str | None = None,
        model_name: str | None = None,
        num_recycles: int = 3,
        seed: int | None = None,
        timeout_s: float | None = None,
        template_query: str | None = None,
        **kwargs: Any,
    ) -> None:
        if template_query is not None:
            raise ValueError(
                "AF2MLXOracle always templates the non-binder chains' backbone in complexes and takes no "
                "template_query; set it to null."
            )
        super().__init__(oracle_name=oracle_name, template_query=None, **kwargs)
        self.python = python or os.environ.get("AF2_PYTHON")
        self.params_dir = params_dir or os.environ.get("AF2_DIR")
        if not self.python:
            raise ValueError("AF2MLXOracle needs `python` or $AF2_PYTHON (the JAX environment's interpreter).")
        if not self.params_dir:
            raise ValueError("AF2MLXOracle needs `params_dir` or $AF2_DIR (directory with params_model_*.npz).")
        if model_name is not None and not (model_name.endswith("_ptm") or model_name.endswith("_multimer_v3")):
            raise ValueError(f"model_name must be model_N_ptm or model_N_multimer_v3, got {model_name!r}")
        self.model_name = model_name
        self.num_recycles = num_recycles
        self.seed = seed
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # OracleMetric interface
    # ------------------------------------------------------------------

    def fold_one(self, job: OracleJobInput) -> dict:
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"AF2MLXOracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def _plan(self, job: OracleJobInput) -> tuple[list[ChainEntity], ChainEntity | None]:
        """(protein entities in input order, binder entity or ``None`` for a single chain)."""
        entities = chain_entities(job.atom_array)
        bad = [e.chain_id for e in entities if e.kind != "protein"]
        if bad:
            raise ValueError(f"{job.example_id!r}: AF2 only models protein; chain(s) {bad} are not protein.")
        if len(entities) == 1:
            return entities, None
        coi = job.chain_of_interest
        if coi is None:
            coi, unambiguous = get_chain_of_interest(job.atom_array)
            if not unambiguous:
                coi = None
        binder = [e for e in entities if e.is_chain(coi)]
        if not binder:
            raise ValueError(f"{job.example_id!r}: {len(entities)} protein chains and no usable chain of interest.")
        return entities, binder[0]

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        names = [self.manifest_job_name(j) for j in pending]
        if len(set(names)) != len(names):
            raise ValueError(f"AF2MLXOracle: duplicate job names in one batch: {sorted({n for n in names if names.count(n) > 1})}")
        plans = [self._plan(j) for j in pending]

        batch_dir = Path(tempfile.mkdtemp(prefix="af2mlx_batch_"))
        out_root = batch_dir / "out"
        try:
            manifest = []
            for job, name, (entities, binder) in zip(pending, names, plans, strict=True):
                manifest.append(self._manifest_entry(job, name, entities, binder, batch_dir, out_root))
            manifest_path = batch_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self._run_worker(manifest_path, [out_root / f"{n}.json" for n in names])

            n_ok = 0
            for job, name, (entities, binder) in zip(pending, names, plans, strict=True):
                self._finalize(job, name, entities, binder, out_root)
                n_ok += 1
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Input / worker
    # ------------------------------------------------------------------

    def _manifest_entry(self, job, name, entities, binder, batch_dir: Path, out_root: Path) -> dict:
        entry = {
            "name": name,
            "params_dir": str(self.params_dir),
            "num_recycles": self.num_recycles,
            "seed": self.seed,
            "out_dir": str(out_root),
        }
        if binder is None:
            entry.update(kind="monomer", sequence=entities[0].sequence, model_name=self.model_name or "model_1_ptm")
            return entry
        targets = [e for e in entities if e is not binder]
        letters = ascii_uppercase[: len(targets)]
        if len(targets) > len(ascii_uppercase):
            raise ValueError(f"{job.example_id!r}: too many target chains for a PDB file")
        target_pdb = batch_dir / f"{name}_target.pdb"
        self._write_target_pdb(job.atom_array, targets, letters, target_pdb)
        entry.update(
            kind="binder",
            sequence=binder.sequence,
            model_name=self.model_name or "model_1_multimer_v3",
            target_pdb=str(target_pdb),
            target_chains=",".join(letters),
        )
        return entry

    @staticmethod
    def _write_target_pdb(atom_array, targets: list[ChainEntity], letters: str, path: Path) -> None:
        """Write the target chains (protein atoms with finite coordinates) as a PDB with one-letter chain ids."""
        parts = []
        for e, letter in zip(targets, letters, strict=True):
            chain = atom_array[(atom_array.chain_id == e.chain_id) & ~np.isnan(atom_array.coord).any(axis=-1)]
            chain = chain[chain.element != "H"].copy()
            chain.chain_id[:] = letter
            parts.append(chain)
        merged = parts[0]
        for p in parts[1:]:
            merged = merged + p
        f = pdbio.PDBFile()
        pdbio.set_structure(f, merged)
        f.write(str(path))

    def _run_worker(self, manifest_path: Path, expected: list[Path]) -> None:
        cmd = [str(self.python), str(_WORKER), str(manifest_path)]
        logger.info(f"AF2 (jax-mps) oracle: {' '.join(cmd)}")
        env = os.environ.copy()
        for var in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(var, None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            # A native crash while exiting must not discard finished folds, but never mask a real failure.
            if expected and all(p.exists() and "error" not in json.loads(p.read_text()) for p in expected):
                logger.warning(f"AF2 worker exited with {proc.returncode} after writing all results; using them.")
            else:
                raise RuntimeError(f"AF2 worker failed (exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-3000:]}")

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _finalize(self, job, name, entities, binder, out_root: Path) -> dict:
        res_path = out_root / f"{name}.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"error": "no result written"}
        if "error" in res:
            raise RuntimeError(f"AF2 failed for {job.example_id!r}: {res['error']}")
        arrays = np.load(out_root / f"{name}.npz")
        plddt, pae = arrays["plddt"].astype(float), arrays["pae"].astype(float)

        # ColabDesign orders residues: target chains (input order), then the binder.
        order = [e for e in entities if e is not binder] + ([binder] if binder is not None else [])
        bounds = np.cumsum([0, *[len(e.sequence) for e in order]])
        if plddt.shape[0] != bounds[-1] or pae.shape != (bounds[-1], bounds[-1]):
            raise RuntimeError(f"AF2 output size {plddt.shape} != expected {bounds[-1]} residues for {job.example_id!r}")
        sl = {e.chain_id: slice(bounds[i], bounds[i + 1]) for i, e in enumerate(order)}

        n = len(entities)
        idx = {e.chain_id: i for i, e in enumerate(entities)}  # position in the reported (input) chain order
        pair_pae = [[None] * n for _ in range(n)]
        pair_pae_min = [[None] * n for _ in range(n)]
        pair_iptm = [[None] * n for _ in range(n)]
        for e in entities:
            pair_pae[idx[e.chain_id]][idx[e.chain_id]] = float(pae[sl[e.chain_id], sl[e.chain_id]].mean())
        if binder is not None:
            for t in (e for e in entities if e is not binder):
                blocks = [pae[sl[binder.chain_id], sl[t.chain_id]], pae[sl[t.chain_id], sl[binder.chain_id]]]
                i, j = idx[binder.chain_id], idx[t.chain_id]
                pair_pae[i][j] = pair_pae[j][i] = float(np.mean([b.mean() for b in blocks]))
                pair_pae_min[i][j] = pair_pae_min[j][i] = float(min(b.min() for b in blocks))
                pair_iptm[i][j] = pair_iptm[j][i] = res.get("i_ptm")

        ptm = res.get("ptm")
        iptm = res.get("i_ptm") if binder is not None else None
        summary: dict[str, Any] = {
            "overall_plddt": float(plddt.mean()),
            "overall_pae": float(pae.mean()),
            "ptm": ptm,
            "iptm": iptm,
            "ranking_score": (0.8 * iptm + 0.2 * ptm) if (iptm is not None and ptm is not None) else ptm,
            "chain_plddt": [float(plddt[sl[e.chain_id]].mean()) for e in entities],
            "chain_pae": [pair_pae[i][i] for i in range(n)],
            "chain_pair_pae": pair_pae,
        }
        if binder is not None:
            summary["chain_pair_pae_min"] = pair_pae_min
            summary["chain_pair_iptm"] = pair_iptm
        summary = {k: v for k, v in summary.items() if v is not None}

        self._save_refolded_structure(job, out_root / f"{name}.pdb", order)

        chain_of_interest = binder.iid if binder is not None else None
        if chain_of_interest is None:
            chain_of_interest = job.chain_of_interest
            if chain_of_interest is None:
                chain_of_interest, unambiguous = get_chain_of_interest(job.atom_array)
                chain_of_interest = chain_of_interest if unambiguous else None
        metrics = extract_confidence_metrics(summary, chain_of_interest, [e.iid for e in entities])
        for key in ("min_ipsae", "max_ipsae", "avg_ipsae"):  # ColabDesign's ipSAE variants, binder complexes only
            if res.get(key) is not None:
                metrics[key] = float(res[key])
        metrics["chain_of_interest"] = chain_of_interest
        self._save_confidence_json(metrics, self.confidence_path(job.out_dir, job.example_id, job.save_id))
        return metrics

    def _save_refolded_structure(self, job: OracleJobInput, pdb_path: Path, order: list[ChainEntity]) -> None:
        cif_path = self.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
            predicted = as_atom_array_plus(parse(str(pdb_path), config=config)["asym_unit"][0])
            # The PDB chains come out in the same order as `order`; restore the design's own chain ids.
            out_chains = list(dict.fromkeys(predicted.chain_id.tolist()))
            if len(out_chains) == len(order):
                new_ids = predicted.chain_id.copy()
                for old, e in zip(out_chains, order, strict=True):
                    new_ids[predicted.chain_id == old] = e.chain_id
                predicted.chain_id[:] = new_ids
            to_cif_file(predicted, cif_path)
        except Exception as e:
            logger.warning(f"Failed to save AF2 refolded CIF for {job.example_id}: {e}")
