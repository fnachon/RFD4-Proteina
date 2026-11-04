from pathlib import Path

import pytest

from rfproteina.metrics.oracles.rf3_oracle import RF3Oracle


def test_rejects_template_query():
    with pytest.raises(ValueError, match="template"):
        RF3Oracle(template_query="sa_mask('C_CRD')")


def test_build_command_forwards_only_set_options():
    oracle = RF3Oracle(rf3_executable="/x/rf3", n_recycles=3, seed=7, extra_rf3_args=["num_steps=50"])
    cmd = oracle._build_command(Path("in"), Path("out"))
    assert cmd[:2] == ["/x/rf3", "fold"]
    assert "inputs=in" in cmd and "out_dir=out" in cmd
    assert "n_recycles=3" in cmd and "seed=7" in cmd and "num_steps=50" in cmd
    assert not any(c.startswith(("ckpt_path=", "diffusion_batch_size=")) for c in cmd)


def test_output_paths_use_oracle_name(tmp_path):
    oracle = RF3Oracle(oracle_name="rf3_ppi")
    assert oracle.confidence_path(tmp_path, "d1").name == "d1_rf3_ppi_confidences.json"
