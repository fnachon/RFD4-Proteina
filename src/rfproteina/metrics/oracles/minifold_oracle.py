"""MiniFold structure prediction oracle (single-chain, ESM-2 based, no MSA).

MiniFold predicts one protein chain from its sequence. This oracle therefore folds only the chain of interest (or the
sole protein chain) of each design, so the refold holds that chain alone: an *apo*, single-sequence designability
check, not a complex prediction. Confidences are per-residue pLDDT only; there is no pTM, iPTM or PAE, so criteria on
those keys cannot be evaluated with this oracle.

Configure with ``MINIFOLD_PREDICT`` (path to MiniFold's ``predict.py``), ``MINIFOLD_PYTHON`` (the interpreter of the
environment MiniFold is installed in) and ``MINIFOLD_CACHE`` (directory holding ``minifold_<size>.ckpt``).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file

from rfproteina.metrics.oracles.oracle_inputs import ChainEntity, chain_entities
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class MiniFoldOracle(OracleMetric):
    """Refold the chain of interest with MiniFold and report pLDDT.

    Args:
        oracle_name: Label used in output filenames.
        predict_script: MiniFold ``predict.py`` (falls back to ``$MINIFOLD_PREDICT``).
        python: Interpreter to run it with (falls back to ``$MINIFOLD_PYTHON``, then the current one).
        cache_dir: Directory with the MiniFold checkpoint (falls back to ``$MINIFOLD_CACHE``).
        model_size: ``48L`` or ``12L``.
        token_per_batch: MiniFold batching limit.
        extra_args: Extra CLI arguments appended verbatim.
        timeout_s: Per-batch subprocess timeout.
    """

    def __init__(
        self,
        oracle_name: str = "minifold",
        predict_script: str | None = None,
        python: str | None = None,
        cache_dir: str | None = None,
        model_size: str = "48L",
        token_per_batch: int | None = None,
        extra_args: list[str] | None = None,
        timeout_s: float | None = None,
        template_query: str | None = None,
        **kwargs: Any,
    ) -> None:
        if template_query is not None:
            raise ValueError("MiniFoldOracle takes no templates; set template_query=null.")
        super().__init__(oracle_name=oracle_name, template_query=None, **kwargs)
        script = predict_script or os.environ.get("MINIFOLD_PREDICT")
        if not script:
            raise ValueError("MiniFoldOracle needs predict_script or $MINIFOLD_PREDICT (MiniFold's predict.py).")
        self.predict_script = Path(script)
        self.python = python or os.environ.get("MINIFOLD_PYTHON") or sys.executable
        self.cache_dir = cache_dir or os.environ.get("MINIFOLD_CACHE")
        self.model_size = model_size
        self.token_per_batch = token_per_batch
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s

    def fold_one(self, job: OracleJobInput) -> dict:
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"MiniFoldOracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def _select_chain(self, job: OracleJobInput) -> ChainEntity:
        """The protein chain to fold."""
        proteins = [e for e in chain_entities(job.atom_array) if e.kind == "protein"]
        coi = job.chain_of_interest
        if coi is None:
            coi, unambiguous = get_chain_of_interest(job.atom_array)
            if not unambiguous:
                coi = None
        if coi is not None:
            match = [e for e in proteins if e.is_chain(coi)]
            if match:
                return match[0]
            raise ValueError(f"{job.example_id!r}: chain of interest {coi!r} is not a protein chain MiniFold can fold.")
        if len(proteins) == 1:
            return proteins[0]
        raise ValueError(f"{job.example_id!r}: {len(proteins)} protein chains and no unambiguous chain of interest.")

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        names = [self.manifest_job_name(j) for j in pending]
        if len(set(names)) != len(names):
            raise ValueError(f"MiniFoldOracle: duplicate job names in one batch: {sorted({n for n in names if names.count(n) > 1})}")
        chains = [self._select_chain(j) for j in pending]

        batch_dir = Path(tempfile.mkdtemp(prefix="minifold_batch_"))
        try:
            fasta = batch_dir / "designs.fasta"
            fasta.write_text("".join(f">{n}\n{e.sequence}\n" for n, e in zip(names, chains, strict=True)))
            out_dir = batch_dir / "out"
            self._run(fasta, out_dir)
            result_dir = out_dir / f"minifold_results_{fasta.stem}"

            n_ok = 0
            for job, name, entity in zip(pending, names, chains, strict=True):
                pdb = result_dir / f"{name}.pdb"
                if not pdb.exists():
                    raise RuntimeError(f"MiniFold produced no structure for {job.example_id!r}; check its output.")
                self._finalize(job, pdb, entity)
                n_ok += 1
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _build_command(self, fasta: Path, out_dir: Path) -> list[str]:
        cmd = [self.python, str(self.predict_script), str(fasta), "--out_dir", str(out_dir), "--model_size", self.model_size]
        if self.cache_dir:
            cmd += ["--cache", str(self.cache_dir)]
        if self.token_per_batch is not None:
            cmd += ["--token_per_batch", str(self.token_per_batch)]
        return cmd + self.extra_args

    def _run(self, fasta: Path, out_dir: Path) -> None:
        cmd = self._build_command(fasta, out_dir)
        logger.info(f"MiniFold oracle: {' '.join(cmd)}")
        env = os.environ.copy()
        for var in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(var, None)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=self.timeout_s, cwd=self.predict_script.parent, check=False
        )
        # predict.py swallows per-batch exceptions and exits 0, so surface them here.
        if proc.returncode != 0 or "Failed to predict batch" in proc.stdout:
            raise RuntimeError(f"MiniFold failed (exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-3000:]}")

    def _finalize(self, job: OracleJobInput, pdb_path: Path, entity: ChainEntity) -> None:
        config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
        predicted = as_atom_array_plus(parse(str(pdb_path), config=config)["asym_unit"][0])
        # MiniFold writes pLDDT either in 0..1 or 0..100 depending on version; normalise to 0..100.
        if float(np.max(predicted.b_factor)) <= 1.0:
            predicted.set_annotation("b_factor", predicted.b_factor * 100.0)
        plddt = float(np.mean(predicted.b_factor[predicted.atom_name == "CA"]))
        predicted.chain_id[:] = entity.chain_id

        cif_path = self.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        to_cif_file(predicted, cif_path)

        metrics = {"overall_plddt": plddt, "plddt_coi": plddt, "chain_of_interest": entity.iid}
        self._save_confidence_json(metrics, self.confidence_path(job.out_dir, job.example_id, job.save_id))
