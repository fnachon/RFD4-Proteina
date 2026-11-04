"""Boltz-2 structure prediction oracle.

Boltz lives in its own Python environment, so folds are dispatched to the ``boltz predict`` CLI as one subprocess per
batch. On Apple Silicon Boltz runs on MPS. Set ``BOLTZ_BIN`` (or ``boltz_executable``) to the CLI and, if needed,
``BOLTZ_CACHE`` to the weights directory.

MSAs: ``msa_mode="none"`` (default) folds every protein chain single-sequence (``msa: empty``). ``"target"`` searches
every protein chain except the chain of interest and ``"all"`` searches all of them, using a local MMseqs2
(see :mod:`rfproteina.utils.mmseqs_msa`). Structural templates are not forwarded, so ``template_query`` must be unset.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file

from rfproteina.metrics.oracles.oracle_inputs import ChainEntity, MSAMode, chain_entities, resolve_chain_msas
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mmseqs_msa import MMseqsMSABuilder

logger = RankedLogger(__name__, rank_zero_only=True)


class Boltz2Oracle(OracleMetric):
    """Refold designs with Boltz-2 and report AF3-style confidence metrics.

    Args:
        oracle_name: Label used in output filenames.
        boltz_executable: ``boltz`` CLI (falls back to ``$BOLTZ_BIN``, then ``boltz``).
        cache_dir: Boltz weights directory (falls back to ``$BOLTZ_CACHE``, then Boltz's default).
        accelerator: ``gpu`` (resolved to MPS or CUDA by Boltz), ``mps``, ``cpu``.
        recycling_steps / sampling_steps / diffusion_samples: Boltz sampling options (Boltz defaults when ``None``).
            The best sample by ``confidence_score`` is kept.
        use_potentials: Enable Boltz inference-time potentials.
        msa_mode: ``none`` | ``target`` | ``all``; see module docstring.
        msa_builder: MMseqs2 MSA builder; created from the environment when an MSA mode needs one and none is given.
        extra_boltz_args: Extra CLI arguments appended verbatim.
        timeout_s: Per-batch subprocess timeout.
    """

    def __init__(
        self,
        oracle_name: str = "boltz2",
        boltz_executable: str | None = None,
        cache_dir: str | None = None,
        accelerator: str = "gpu",
        recycling_steps: int | None = None,
        sampling_steps: int | None = None,
        diffusion_samples: int | None = None,
        use_potentials: bool = False,
        msa_mode: MSAMode = "none",
        msa_builder: MMseqsMSABuilder | None = None,
        extra_boltz_args: list[str] | None = None,
        timeout_s: float | None = None,
        template_query: str | None = None,
        **kwargs: Any,
    ) -> None:
        if template_query is not None:
            raise ValueError("Boltz2Oracle does not forward structural templates; set template_query=null.")
        if msa_mode not in ("none", "target", "all"):
            raise ValueError(f"msa_mode must be none, target or all, got {msa_mode!r}")
        super().__init__(oracle_name=oracle_name, template_query=None, **kwargs)
        self.boltz_executable = boltz_executable or os.environ.get("BOLTZ_BIN") or "boltz"
        self.cache_dir = cache_dir or os.environ.get("BOLTZ_CACHE")
        self.accelerator = accelerator
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.use_potentials = use_potentials
        self.msa_mode = msa_mode
        self.msa_builder = msa_builder
        self.extra_boltz_args = list(extra_boltz_args or [])
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # OracleMetric interface
    # ------------------------------------------------------------------

    def fold_one(self, job: OracleJobInput) -> dict:
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"Boltz2Oracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        names = [self.manifest_job_name(j) for j in pending]
        if len(set(names)) != len(names):
            raise ValueError(f"Boltz2Oracle: duplicate job names in one batch: {sorted({n for n in names if names.count(n) > 1})}")

        entities = [chain_entities(j.atom_array) for j in pending]
        if self.msa_mode != "none" and self.msa_builder is None:
            self.msa_builder = MMseqsMSABuilder()
        msas = resolve_chain_msas(pending, entities, self.msa_builder, self.msa_mode)

        batch_dir = Path(tempfile.mkdtemp(prefix="boltz_batch_"))
        in_dir, out_root = batch_dir / "inputs", batch_dir / "outputs"
        in_dir.mkdir()
        out_root.mkdir()
        try:
            for name, ents, chain_msas in zip(names, entities, msas, strict=True):
                (in_dir / f"{name}.yaml").write_text(yaml.safe_dump(self._build_input(ents, chain_msas), sort_keys=False))
            self._run_boltz(in_dir, out_root)

            n_ok = 0
            for job, name, ents in zip(pending, names, entities, strict=True):
                if self._finalize(job, ents, out_root, name) is not None:
                    n_ok += 1
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Input / command
    # ------------------------------------------------------------------

    @staticmethod
    def _build_input(entities: list[ChainEntity], chain_msas: dict[str, Path]) -> dict:
        sequences = []
        for e in entities:
            if e.kind == "ligand":
                sequences.append({"ligand": {"id": e.chain_id, "ccd": e.ccd}})
            elif e.kind == "protein":
                msa = str(Path(chain_msas[e.chain_id]).resolve()) if e.chain_id in chain_msas else "empty"
                sequences.append({"protein": {"id": e.chain_id, "sequence": e.sequence, "msa": msa}})
            else:
                sequences.append({e.kind: {"id": e.chain_id, "sequence": e.sequence}})
        return {"version": 1, "sequences": sequences}

    def _build_command(self, in_dir: Path, out_dir: Path) -> list[str]:
        cmd = [self.boltz_executable, "predict", str(in_dir), "--out_dir", str(out_dir), "--accelerator", self.accelerator]
        for flag, value in (
            ("--recycling_steps", self.recycling_steps),
            ("--sampling_steps", self.sampling_steps),
            ("--diffusion_samples", self.diffusion_samples),
            ("--cache", self.cache_dir),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        if self.use_potentials:
            cmd.append("--use_potentials")
        if sys.platform == "darwin":
            cmd.append("--no_kernels")  # the trifast kernels are CUDA-only
        return cmd + self.extra_boltz_args

    def _run_boltz(self, in_dir: Path, out_dir: Path) -> None:
        cmd = self._build_command(in_dir, out_dir)
        logger.info(f"Boltz-2 oracle: {' '.join(cmd)}")
        env = os.environ.copy()
        for var in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(var, None)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"boltz predict failed (exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-3000:]}")

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_prediction_dir(out_root: Path, name: str) -> Path | None:
        hits = sorted(out_root.glob(f"**/predictions/{name}"))
        return hits[0] if hits else None

    @staticmethod
    def _tokens_per_chain(entities: list[ChainEntity], atom_array) -> list[int]:
        """Boltz tokenises polymers per residue and CCD ligands per heavy atom."""
        counts = []
        for e in entities:
            if e.kind == "ligand":
                chain = atom_array[atom_array.chain_id == e.chain_id]
                counts.append(int(np.sum(chain.element != "H")))
            else:
                counts.append(len(e.sequence))
        return counts

    def _finalize(self, job: OracleJobInput, entities: list[ChainEntity], out_root: Path, name: str) -> dict | None:
        pred_dir = self._find_prediction_dir(out_root, name)
        conf_path = pred_dir / f"confidence_{name}_model_0.json" if pred_dir else None
        model_path = pred_dir / f"{name}_model_0.cif" if pred_dir else None
        if not (conf_path and conf_path.exists() and model_path.exists()):
            raise RuntimeError(f"Boltz-2 produced no outputs for {job.example_id!r} under {out_root}; check boltz stderr.")
        raw = json.loads(conf_path.read_text())
        n = len(entities)

        def matrix(key: str):
            d = raw.get(key)
            return None if d is None else [[d.get(str(i), {}).get(str(j)) for j in range(n)] for i in range(n)]

        pair_pae = matrix("pair_chains_pae")
        summary: dict[str, Any] = {
            "overall_plddt": raw.get("complex_plddt"),
            "overall_pae": raw.get("complex_pae"),
            "ptm": raw.get("ptm"),
            "iptm": raw.get("iptm"),
            "ranking_score": raw.get("confidence_score"),
            "chain_ptm": [raw.get("chains_ptm", {}).get(str(i)) for i in range(n)],
            "chain_pair_iptm": matrix("pair_chains_iptm"),
            "chain_pair_pae": pair_pae,
            "chain_pae": [pair_pae[i][i] for i in range(n)] if pair_pae else None,
        }
        summary = {k: v for k, v in summary.items() if v is not None}

        # Per-chain pLDDT and the tightest inter-chain PAE need the token-level arrays.
        try:
            bounds = np.cumsum([0, *self._tokens_per_chain(entities, job.atom_array)])
            plddt = np.load(pred_dir / f"plddt_{name}_model_0.npz")["plddt"]
            pae = np.load(pred_dir / f"pae_{name}_model_0.npz")["pae"]
            if plddt.shape[0] != bounds[-1] or pae.shape[0] != bounds[-1]:
                raise ValueError(f"token count mismatch: expected {bounds[-1]}, got {plddt.shape[0]} / {pae.shape[0]}")
            sl = [slice(bounds[i], bounds[i + 1]) for i in range(n)]
            summary["chain_plddt"] = [float(plddt[s].mean()) for s in sl]
            pae_min = [[None] * n for _ in range(n)]
            for i, j in combinations(range(n), 2):
                pae_min[i][j] = float(pae[sl[i], sl[j]].min())
                pae_min[j][i] = float(pae[sl[j], sl[i]].min())
            summary["chain_pair_pae_min"] = pae_min
        except Exception as e:
            logger.warning(f"Boltz-2 token-level metrics unavailable for {job.example_id}: {e}")

        self._save_refolded_structure(job, model_path)

        chain_of_interest = job.chain_of_interest
        if chain_of_interest is None:
            chain_of_interest, unambiguous = get_chain_of_interest(job.atom_array)
            if not unambiguous:
                chain_of_interest = None
        metrics = extract_confidence_metrics(summary, chain_of_interest, [e.iid for e in entities])
        metrics["chain_of_interest"] = chain_of_interest
        self._save_confidence_json(metrics, self.confidence_path(job.out_dir, job.example_id, job.save_id))
        return metrics

    def _save_refolded_structure(self, job: OracleJobInput, model_path: Path) -> None:
        cif_path = self.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
            predicted = as_atom_array_plus(parse(str(model_path), config=config)["asym_unit"][0])
            to_cif_file(predicted, cif_path)
        except Exception as e:
            logger.warning(f"Failed to save Boltz-2 refolded CIF for {job.example_id}: {e}")
