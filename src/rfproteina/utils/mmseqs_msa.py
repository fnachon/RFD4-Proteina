"""Build protein MSAs (A3M) with a local MMseqs2 install, cached per sequence.

Configure through arguments or environment variables:

- ``MMSEQS_BIN``: the ``mmseqs`` executable (use a build with the Metal/CUDA GPU backend for ``gpu=True``).
- ``MMSEQS_DB``: the target database, e.g. a ``makepaddedseqdb`` UniRef90 (required for ``gpu=True``).
- ``RFP_MSA_CACHE``: cache directory (default ``~/.cache/rfproteina/msa``).

The search follows the usual GPU recipe: ``search --gpu 1 -s 7.5 -a 1 -e 1e-3`` then ``result2msa`` in A3M mode.
Many sequences are searched in a single MMseqs2 run, because loading a large database dominates the cost.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["MMseqsMSABuilder"]


class MMseqsMSABuilder:
    def __init__(
        self,
        mmseqs_bin: str | None = None,
        database: str | None = None,
        cache_dir: str | Path | None = None,
        gpu: bool = True,
        sensitivity: float = 7.5,
        evalue: float = 1e-3,
        max_seqs: int | None = None,
        threads: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.mmseqs_bin = mmseqs_bin or os.environ.get("MMSEQS_BIN") or "mmseqs"
        database = database or os.environ.get("MMSEQS_DB")
        if not database:
            raise ValueError("MMseqsMSABuilder needs a database: pass `database` or set MMSEQS_DB.")
        self.database = str(database)
        self.cache_dir = Path(cache_dir or os.environ.get("RFP_MSA_CACHE") or Path.home() / ".cache/rfproteina/msa")
        self.gpu = gpu
        self.sensitivity = sensitivity
        self.evalue = evalue
        self.max_seqs = max_seqs
        self.threads = threads
        self.timeout_s = timeout_s

    # Anything that changes the A3M must change the cache key.
    def _settings_tag(self) -> str:
        raw = f"{Path(self.database).name}|{self.sensitivity}|{self.evalue}|{self.max_seqs}"
        return hashlib.sha1(raw.encode()).hexdigest()[:8]

    def cache_path(self, sequence: str) -> Path:
        digest = hashlib.sha1(sequence.encode()).hexdigest()[:20]
        return self.cache_dir / self._settings_tag() / f"{digest}.a3m"

    def build(self, sequences: Iterable[str]) -> dict[str, Path]:
        """Return ``{sequence: a3m_path}``, searching only sequences not already cached."""
        unique = list(dict.fromkeys(s for s in sequences if s))
        paths = {s: self.cache_path(s) for s in unique}
        missing = [s for s in unique if not paths[s].exists()]
        if missing:
            logger.info(f"MMseqs2 MSA: searching {len(missing)} new sequence(s) ({len(unique) - len(missing)} cached)")
            self._search(missing, paths)
        return paths

    def _run(self, args: list[str]) -> None:
        proc = subprocess.run(
            [self.mmseqs_bin, *args], capture_output=True, text=True, timeout=self.timeout_s, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mmseqs {args[0]} failed (exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-3000:]}")

    def _search(self, sequences: list[str], paths: dict[str, Path]) -> None:
        work = Path(tempfile.mkdtemp(prefix="rfp_mmseqs_"))
        try:
            fasta = work / "queries.fa"
            fasta.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(sequences)))
            thr = ["--threads", str(self.threads)] if self.threads else []
            search_extra = ["--max-seqs", str(self.max_seqs)] if self.max_seqs else []
            self._run(["createdb", str(fasta), str(work / "qdb"), "-v", "1"])
            self._run(
                [
                    "search", str(work / "qdb"), self.database, str(work / "res"), str(work / "tmp"),
                    "--gpu", "1" if self.gpu else "0",
                    "-s", str(self.sensitivity), "-a", "1", "-e", str(self.evalue), "-v", "1",
                    *search_extra, *thr,
                ]
            )  # fmt: skip
            self._run(
                ["result2msa", str(work / "qdb"), self.database, str(work / "res"), str(work / "msa"),
                 "--msa-format-mode", "5", "-v", "1", *thr]
            )  # fmt: skip
            out = work / "a3m"
            self._run(["unpackdb", str(work / "msa"), str(out), "--unpack-suffix", ".a3m", "--unpack-name-mode", "1", "-v", "1"])

            for i, seq in enumerate(sequences):
                src = out / f"{i}.a3m"
                dest = paths[seq]
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".a3m.tmp")
                if src.exists() and src.stat().st_size > 0:
                    shutil.copyfile(src, tmp)
                else:  # no hits: single-sequence MSA holding just the query
                    tmp.write_text(f">query\n{seq}\n")
                tmp.replace(dest)  # atomic, so concurrent ranks never read a partial file
        finally:
            shutil.rmtree(work, ignore_errors=True)
