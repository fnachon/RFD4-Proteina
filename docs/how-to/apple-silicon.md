# Running on Apple Silicon (MPS)

RFD4-Proteina runs on Apple Silicon through PyTorch's MPS backend. This is an inference path: generation and
refolding metrics work; training and the BioIR backend are untested on MPS. It is slower than a CUDA GPU and needs
unified memory (about 8 GB for a ~300-token PPI design and 15 GB for ~550 tokens at batch size 1).

## Install

```bash
bash setup.sh --native      # on an arm64 Mac this installs the `macos` extra (tmol, posebusters, ...)
source .venv/bin/activate
bash skills/rfd4-proteina-setup/scripts/preflight.sh
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=1 out_dir=./runs/smoke
```

The device is chosen as CUDA, then MPS, then CPU. Force one with `RFP_DEVICE=cuda|mps|cpu`.

## What differs from CUDA

| Area | On MPS |
|---|---|
| Precision | `bf16-mixed` runs as `32-true` (`RFP_MPS_ALLOW_BF16=1` keeps bf16). |
| cuEquivariance | Not available. Attention uses `scaled_dot_product_attention`; triangle multiplication uses fused Metal kernels (`rfproteina/nn/modules/metal_trimul.py`, disable with `RFP_MPS_TRIMUL=0`). |
| `torch.compile` | Used up to 512 tokens. Longer inputs run eagerly and pad to a multiple of 16 instead of the next bucket (`RFP_MPS_EAGER_ABOVE=<tokens>`, `-1` never forces eager). |
| Unsupported ops | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so they run on the CPU. |

Measured on one machine, 38_TNFalpha (556 tokens), `medium/front_loaded`, batch size 1: about 400 s and 14-16 GB.
Run-to-run timing varies by roughly 10%.

## Check a result against the CPU

```bash
python scripts/device_parity.py --seed 0 --out-root runs/parity -- \
    inputs=data/benchmarks/monomer subset_to_examples=monomer-short num_replicates=1 \
    generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode compile_lengths=null
```

It runs both devices with identical CPU-drawn noise and compares the superimposed CA RMSD. `--devices mps mps
--tokenwise-noise --ref-extra compile_lengths=null` compares eager against compiled on the same device.

## Refolding oracles

AF3 and OpenFold3 need CUDA. These oracles run on a Mac, each in its own environment and selected with
`metrics/refolding_oracles=<name>`:

| Oracle | Config | Environment variables |
|---|---|---|
| RoseTTAFold3 (Foundry) | `rf3` | `RF3_BIN` |
| Boltz-2 | `boltz2` | `BOLTZ_BIN` |
| MiniFold (single chain, pLDDT only) | `minifold` | `MINIFOLD_PREDICT`, `MINIFOLD_PYTHON`, `MINIFOLD_CACHE` |
| AlphaFold 2 (ColabDesign on jax-mps) | `af2_mlx` | `AF2_PYTHON`, `AF2_DIR` |

None of them forwards templates, so pass rates are not comparable to AF3's. MSAs come from a local MMseqs2
(`MMSEQS_BIN`, `MMSEQS_DB`; built with the Metal GPU backend it runs on the Mac GPU), cached by sequence in
`RFP_MSA_CACHE`. Use `msa_mode` in the oracle config to choose none, target-only or all chains.

### Where each oracle runs

| Oracle | Device on a Mac |
|---|---|
| `rf3` | RF3 selects its own accelerator (MPS on Apple Silicon). |
| `boltz2` | `accelerator: gpu` (the default) resolves to MPS. The oracle adds `--no_kernels` on macOS because Boltz's trifast kernels are CUDA-only. |
| `minifold` | MiniFold's `predict.py` picks CUDA, then MPS, then CPU, and runs its model in bf16 autocast. |
| `af2_mlx` | jax-mps (Metal). |

### Tested on, and known limits

- The four oracles were exercised end to end through `rfd4-generate` on a monomer and a protein-protein binder
  example. Small-molecule binders were run through RF3 and Boltz-2 only (below). Multi-chain complexes beyond a
  binder and its target, antibodies, and DNA were not run through any of them.
- MiniFold folds a single sequence with no MSA and returns pLDDT only, so criteria on pTM, ipTM or PAE cannot be
  evaluated with it. It is an apo, single-chain designability check, not a complex prediction.
- On a repetitive designed monomer, MiniFold predicted an extended helix far from the design (superimposed RMSD
  13.6-22 Å) while RF3, Boltz-2 and AF2 agreed with it. Compare its pLDDT with the other oracles' before relying on it
  alone.
- AF2 on jax-mps can abort inside the native library when its process exits, after all results are written. The
  oracle accepts a non-zero exit only if every result file is present and none reports an error.
- All four forward no templates, so refolds of template-conditioned tasks are not comparable to AF3's.

### Small-molecule binders (RF3 and Boltz-2)

Designs for the four `small_molecule_binder/sm_binder` ligands (`medium/midpeak`, `metrics_tags=[sm_binder_refolding]`,
no MSA, `seed=0`), refolded with each oracle on the same Mac. Medians per target; "recovered" counts designs whose
refolded backbone and ligand are both within 2 Å RMSD of the design (a threshold chosen for this table, not one of
the repository's pass criteria).

| Target | Designs | Oracle | pLDDT | Backbone RMSD (Å) | Ligand RMSD (Å) | Recovered |
|---|---|---|---|---|---|---|
| SAM | 2 | RF3 / Boltz-2 | 88 / 94 | 0.4 / 0.6 | 12.1 / 13.2 | 0 / 0 |
| FAD | 4 | RF3 / Boltz-2 | 86 / 89 | 0.6 / 0.8 | 4.7 / 6.9 | 2 / 1 |
| IAI | 4 | RF3 / Boltz-2 | 84 / 88 | 1.3 / 0.5 | 10.8 / 9.8 | 0 / 1 |
| OQO | 4 | RF3 / Boltz-2 | 85 / 95 | 0.6 / 0.8 | 1.6 / 1.1 | 3 / 3 |

- Every run completed and produced the ligand metrics. The refolded ligand has the same atoms and names as the
  design in all 28 refolds, so the ligand RMSD compares like with like.
- Most backbones refold well (22 of 28 within 2 Å). The ligand pose is recovered much less often (10 of 28, counting
  both oracles) and varies by target: OQO is recovered for 3 of 4 designs by both oracles, SAM for none.
- The two oracles mostly agree on which designs recover (OQO and FAD design 3 for both), so the spread across
  targets reflects the designs, not one oracle.
- Six of the 28 refolds have a backbone RMSD above 2 Å (up to 14.9 Å); keep such designs in pass-rate denominators.
- RF3 does not report `pae_coi` or `iptm_coi`; Boltz-2 does. A run of 4 designs takes 4-6 minutes including generation
  (FAD with Boltz-2 took 340 s, peak memory 11.5 GB).
