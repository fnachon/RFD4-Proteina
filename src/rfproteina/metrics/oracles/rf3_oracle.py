"""RoseTTAFold3 (RF3, Foundry) structure prediction oracle.

RF3 lives in its own Python environment (Foundry), so folds are dispatched to the ``rf3 fold`` CLI as one
subprocess per batch. This works on Apple Silicon (MPS), CUDA and CPU: RF3 selects its own accelerator.

Point the oracle at the Foundry install with ``rf3_executable`` or the ``RF3_BIN`` environment variable
(default: ``rf3`` on ``PATH``), and optionally at a checkpoint with ``ckpt_path`` / ``RF3_CKPT_PATH``.

Limitations: structural templates and MSAs are not forwarded (``rf3 fold`` takes them from a JSON spec, not from
CIF inputs), so ``template_query`` must be unset and designs are folded single-sequence.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file

from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class RF3Oracle(OracleMetric):
    """Refold designs with RoseTTAFold3 and report AF3-style confidence metrics.

    Args:
        oracle_name: Label used in output filenames.
        rf3_executable: Path to the ``rf3`` CLI (falls back to ``$RF3_BIN``, then ``rf3``).
        ckpt_path: RF3 checkpoint (falls back to ``$RF3_CKPT_PATH``, then RF3's own default).
        n_recycles: Trunk recycles (RF3 default 10 when ``None``).
        diffusion_batch_size: Samples per design; the best by ``ranking_score`` is kept.
        num_steps: Diffusion steps (RF3 default 200 when ``None``).
        seed: Model seed (RF3 default when ``None``).
        extra_rf3_args: Extra Hydra ``key=value`` overrides appended to the command.
        timeout_s: Per-batch subprocess timeout.
    """

    def __init__(
        self,
        oracle_name: str = "rf3",
        rf3_executable: str | None = None,
        ckpt_path: str | None = None,
        n_recycles: int | None = None,
        diffusion_batch_size: int | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        extra_rf3_args: list[str] | None = None,
        timeout_s: float | None = None,
        template_query: str | None = None,
        **kwargs: Any,
    ) -> None:
        if template_query is not None:
            raise ValueError(
                "RF3Oracle does not forward structural templates; set template_query=null "
                "(use the AF3 oracle for template-conditioned refolding)."
            )
        super().__init__(oracle_name=oracle_name, template_query=None, **kwargs)
        self.rf3_executable = rf3_executable or os.environ.get("RF3_BIN") or "rf3"
        self.ckpt_path = ckpt_path or os.environ.get("RF3_CKPT_PATH")
        self.n_recycles = n_recycles
        self.diffusion_batch_size = diffusion_batch_size
        self.num_steps = num_steps
        self.seed = seed
        self.extra_rf3_args = list(extra_rf3_args or [])
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # OracleMetric interface
    # ------------------------------------------------------------------

    def fold_one(self, job: OracleJobInput) -> dict:
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"RF3Oracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        names = [self.manifest_job_name(j) for j in pending]
        if len(set(names)) != len(names):
            raise ValueError(f"RF3Oracle: duplicate job names in one batch: {sorted(n for n in names if names.count(n) > 1)}")

        batch_dir = Path(tempfile.mkdtemp(prefix="rf3_batch_"))
        in_dir, out_root = batch_dir / "inputs", batch_dir / "outputs"
        in_dir.mkdir()
        out_root.mkdir()
        try:
            for job, name in zip(pending, names, strict=True):
                to_cif_file(job.atom_array, in_dir / f"{name}.cif")
            self._run_rf3(in_dir, out_root)

            n_ok = 0
            for job, name in zip(pending, names, strict=True):
                if self._finalize(job, out_root / name, name) is not None:
                    n_ok += 1
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_command(self, in_dir: Path, out_dir: Path) -> list[str]:
        cmd = [self.rf3_executable, "fold", f"inputs={in_dir}", f"out_dir={out_dir}"]
        for key, value in (
            ("ckpt_path", self.ckpt_path),
            ("n_recycles", self.n_recycles),
            ("diffusion_batch_size", self.diffusion_batch_size),
            ("num_steps", self.num_steps),
            ("seed", self.seed),
        ):
            if value is not None:
                cmd.append(f"{key}={value}")
        return cmd + self.extra_rf3_args

    def _run_rf3(self, in_dir: Path, out_dir: Path) -> None:
        cmd = self._build_command(in_dir, out_dir)
        logger.info(f"RF3 oracle: {' '.join(cmd)}")
        env = os.environ.copy()
        # Run RF3 in its own environment, not ours.
        for var in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(var, None)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"rf3 fold failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")

    def _finalize(self, job: OracleJobInput, rf3_dir: Path, name: str) -> dict | None:
        """Turn RF3's best-ranked sample for one design into the canonical CIF + confidence JSON."""
        summary_path = rf3_dir / f"{name}_summary_confidences.json"
        detail_path = rf3_dir / f"{name}_confidences.json"
        model_path = rf3_dir / f"{name}_model.cif"
        if not (summary_path.exists() and detail_path.exists() and model_path.exists()):
            raise RuntimeError(f"RF3 produced no outputs for {job.example_id!r} under {rf3_dir}; check rf3 stderr.")

        summary = json.loads(summary_path.read_text())
        detail = json.loads(detail_path.read_text())

        # RF3 gives only a per-chain pTM; derive the per-chain pLDDT from per-atom values (0..1).
        atom_chains = np.asarray(detail["atom_chain_ids"])
        atom_plddts = np.asarray(detail["atom_plddts"], dtype=float)
        # token_chain_ids carry an assembly suffix ("A_1"); atom_chain_ids are the input chain ids.
        chains_in_order = list(dict.fromkeys(atom_chains.tolist()))
        summary["chain_plddt"] = [float(atom_plddts[atom_chains == c].mean()) for c in chains_in_order]

        self._save_refolded_structure(job, model_path, atom_plddts)

        chain_of_interest = job.chain_of_interest
        if chain_of_interest is None:
            chain_of_interest, unambiguous = get_chain_of_interest(job.atom_array)
            if not unambiguous:
                chain_of_interest = None

        metrics = extract_confidence_metrics(summary, chain_of_interest, chains_in_order)
        metrics["chain_of_interest"] = chain_of_interest
        self._save_confidence_json(metrics, self.confidence_path(job.out_dir, job.example_id, job.save_id))
        return metrics

    def _save_refolded_structure(self, job: OracleJobInput, model_path: Path, atom_plddts: np.ndarray) -> None:
        cif_path = self.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
            predicted = as_atom_array_plus(parse(str(model_path), config=config)["asym_unit"][0])
            if predicted.array_length() == len(atom_plddts):
                predicted.set_annotation("b_factor", atom_plddts * 100.0)
            to_cif_file(predicted, cif_path)
        except Exception as e:
            logger.warning(f"Failed to save RF3 refolded CIF for {job.example_id}: {e}")
