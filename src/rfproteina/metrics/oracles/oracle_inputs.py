"""Chain-level inputs shared by external-oracle wrappers (Boltz-2, MiniFold, ...).

``chain_entities`` reads an AtomArray into polymer / ligand entities. ``resolve_chain_msas`` fills in A3M paths for
protein chains using :class:`~rfproteina.utils.mmseqs_msa.MMseqsMSABuilder`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from biotite.structure import AtomArray

from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.mmseqs_msa import MMseqsMSABuilder

_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X",
}  # fmt: skip
_DNA = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DN": "N"}
_RNA = {"A": "A", "C": "C", "G": "G", "U": "U", "N": "N"}
_SKIP_RESIDUES = {"HOH", "DOD"}

MSAMode = Literal["none", "target", "all"]


@dataclass(slots=True)
class ChainEntity:
    chain_id: str  # what the external tool is given
    kind: Literal["protein", "dna", "rna", "ligand"]
    sequence: str | None = None  # polymers only
    ccd: str | None = None  # ligands only
    iid: str = ""  # chain instance id ("A_1"), the label the pipeline uses for chain_of_interest and metric keys

    def __post_init__(self) -> None:
        self.iid = self.iid or self.chain_id

    def is_chain(self, label: str | None) -> bool:
        """True if ``label`` (a chain id or chain instance id) names this chain."""
        return label is not None and str(label) in (self.chain_id, self.iid)


def chain_entities(atom_array: AtomArray) -> list[ChainEntity]:
    """One entity per chain, in order of first appearance. Water is ignored.

    Raises:
        ValueError: for chains that mix residue kinds or hold several different ligands, which no external
            oracle input format can express faithfully.
    """
    entities: list[ChainEntity] = []
    for chain_id in dict.fromkeys(atom_array.chain_id.tolist()):
        chain = atom_array[atom_array.chain_id == chain_id]
        iid = str(chain.chain_iid[0]) if "chain_iid" in chain.get_annotation_categories() else chain_id
        # residues in order, one entry per (res_id, ins_code)
        keys = list(dict.fromkeys(zip(chain.res_id.tolist(), chain.ins_code.tolist(), chain.res_name.tolist(), strict=True)))
        names = [k[2] for k in keys if k[2] not in _SKIP_RESIDUES]
        if not names:
            continue
        if all(n in _AA3TO1 for n in names):
            entities.append(ChainEntity(chain_id, "protein", "".join(_AA3TO1[n] for n in names), iid=iid))
        elif all(n in _DNA for n in names):
            entities.append(ChainEntity(chain_id, "dna", "".join(_DNA[n] for n in names), iid=iid))
        elif all(n in _RNA for n in names):
            entities.append(ChainEntity(chain_id, "rna", "".join(_RNA[n] for n in names), iid=iid))
        elif len(set(names)) == 1 and len(names) == 1:
            entities.append(ChainEntity(chain_id, "ligand", ccd=names[0], iid=iid))
        else:
            raise ValueError(
                f"Chain {chain_id!r} mixes residue kinds or holds several ligands ({sorted(set(names))[:6]}); "
                "unsupported by this oracle's input format."
            )
    return entities


def resolve_chain_msas(
    jobs: list[OracleJobInput],
    entities_by_job: list[list[ChainEntity]],
    builder: MMseqsMSABuilder | None,
    mode: MSAMode,
) -> list[dict[str, Path]]:
    """Per job, ``{chain_id: a3m_path}`` for the protein chains that should get an MSA.

    ``mode``: ``none`` (single sequence), ``target`` (every protein chain except the chain of interest, so a
    designed binder is folded without an MSA while its target gets one) or ``all``. MSAs supplied on the job
    (``job.msa_paths_by_chain_id``) always win over searching.
    """
    result: list[dict[str, Path]] = [{} for _ in jobs]
    wanted: list[tuple[int, str, str]] = []
    for ji, (job, ents) in enumerate(zip(jobs, entities_by_job, strict=True)):
        provided = {k: Path(v) for k, v in (job.msa_paths_by_chain_id or {}).items()}
        result[ji].update(provided)
        if mode == "none":
            continue
        for e in ents:
            if e.kind != "protein" or e.chain_id in provided or e.iid in provided:
                continue
            if mode == "target" and e.is_chain(job.chain_of_interest):
                continue
            wanted.append((ji, e.chain_id, e.sequence))
    if wanted:
        if builder is None:
            raise ValueError(f"msa_mode={mode!r} needs an MMseqs2 builder (set MMSEQS_BIN/MMSEQS_DB or pass msa_builder).")
        paths = builder.build(seq for _, _, seq in wanted)
        for ji, chain_id, seq in wanted:
            result[ji][chain_id] = paths[seq]
    return result


def chain_ids_array(atom_array: AtomArray) -> np.ndarray:
    return np.asarray(list(dict.fromkeys(atom_array.chain_id.tolist())))
