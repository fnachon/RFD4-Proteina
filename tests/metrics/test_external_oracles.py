from pathlib import Path

import pytest
from biotite.structure import Atom, array

from rfproteina.metrics.oracles.boltz2_oracle import Boltz2Oracle
from rfproteina.metrics.oracles.minifold_oracle import MiniFoldOracle
from rfproteina.metrics.oracles.oracle_inputs import ChainEntity, chain_entities, resolve_chain_msas
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.mmseqs_msa import MMseqsMSABuilder


def _atoms(chain_id, res_name, res_id, element="C", name="CA"):
    return Atom([0.0, 0.0, 0.0], chain_id=chain_id, res_id=res_id, ins_code="", res_name=res_name, hetero=False,
                atom_name=name, element=element)  # fmt: skip


@pytest.fixture
def complex_array():
    return array(
        [_atoms("A", "ALA", 1), _atoms("A", "GLY", 2), _atoms("B", "LYS", 1), _atoms("L", "ATP", 1, "P", "PG")]
    )


def test_chain_entities(complex_array):
    ents = chain_entities(complex_array)
    assert [(e.chain_id, e.kind) for e in ents] == [("A", "protein"), ("B", "protein"), ("L", "ligand")]
    assert ents[0].sequence == "AG" and ents[2].ccd == "ATP"


def test_chain_entities_rejects_mixed_chain():
    mixed = array([_atoms("A", "ALA", 1), _atoms("A", "DA", 2)])
    with pytest.raises(ValueError, match="mixes"):
        chain_entities(mixed)


class _FakeBuilder:
    def __init__(self):
        self.seen = []

    def build(self, seqs):
        self.seen = list(seqs)
        return {s: Path(f"/msa/{s}.a3m") for s in self.seen}


def test_msa_modes(complex_array):
    ents = chain_entities(complex_array)
    job = OracleJobInput(atom_array=complex_array, example_id="d", out_dir=Path("."), chain_of_interest="A")
    assert resolve_chain_msas([job], [ents], None, "none") == [{}]
    fake = _FakeBuilder()
    assert resolve_chain_msas([job], [ents], fake, "target") == [{"B": Path("/msa/K.a3m")}]
    fake = _FakeBuilder()
    assert set(resolve_chain_msas([job], [ents], fake, "all")[0]) == {"A", "B"}
    with pytest.raises(ValueError, match="builder"):
        resolve_chain_msas([job], [ents], None, "all")


def test_provided_msas_win(complex_array):
    ents = chain_entities(complex_array)
    job = OracleJobInput(atom_array=complex_array, example_id="d", out_dir=Path("."), msa_paths_by_chain_id={"B": "/mine.a3m"})
    fake = _FakeBuilder()
    out = resolve_chain_msas([job], [ents], fake, "all")[0]
    assert out["B"] == Path("/mine.a3m") and fake.seen == ["AG"]


def test_boltz_input_yaml_shape():
    ents = [ChainEntity("A", "protein", "AG"), ChainEntity("L", "ligand", ccd="ATP")]
    data = Boltz2Oracle._build_input(ents, {})
    assert data["sequences"][0]["protein"]["msa"] == "empty"
    assert data["sequences"][1] == {"ligand": {"id": "L", "ccd": "ATP"}}


def test_boltz_rejects_templates_and_bad_mode():
    with pytest.raises(ValueError, match="template"):
        Boltz2Oracle(template_query="x")
    with pytest.raises(ValueError, match="msa_mode"):
        Boltz2Oracle(msa_mode="paired")


def test_minifold_needs_script(monkeypatch):
    monkeypatch.delenv("MINIFOLD_PREDICT", raising=False)
    with pytest.raises(ValueError, match="predict"):
        MiniFoldOracle()


def test_minifold_picks_coi(complex_array, monkeypatch):
    monkeypatch.setenv("MINIFOLD_PREDICT", "/x/predict.py")
    o = MiniFoldOracle()
    job = OracleJobInput(atom_array=complex_array, example_id="d", out_dir=Path("."), chain_of_interest="B")
    assert (o._select_chain(job).chain_id, o._select_chain(job).sequence) == ("B", "K")
    ambiguous = OracleJobInput(atom_array=complex_array, example_id="d", out_dir=Path("."), chain_of_interest="L")
    with pytest.raises(ValueError):
        o._select_chain(ambiguous)


def test_msa_builder_cache_key_depends_on_settings(tmp_path):
    a = MMseqsMSABuilder(database="/db/uniref90", cache_dir=tmp_path)
    b = MMseqsMSABuilder(database="/db/uniref90", cache_dir=tmp_path, sensitivity=5.7)
    assert a.cache_path("AAAA") != b.cache_path("AAAA")
    assert a.cache_path("AAAA") == a.cache_path("AAAA")


def test_msa_builder_needs_database(tmp_path, monkeypatch):
    monkeypatch.delenv("MMSEQS_DB", raising=False)
    with pytest.raises(ValueError, match="database"):
        MMseqsMSABuilder(cache_dir=tmp_path)


def test_chain_iid_is_used_for_labels():
    arr = array([_atoms("A", "ALA", 1), _atoms("A", "GLY", 2)])
    arr.add_annotation("chain_iid", "U8")
    arr.chain_iid[:] = "A_1"
    (e,) = chain_entities(arr)
    assert e.chain_id == "A" and e.iid == "A_1" and e.is_chain("A_1") and e.is_chain("A") and not e.is_chain("B")
