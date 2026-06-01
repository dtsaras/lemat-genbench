# Evaluating a generative model (e.g. a finetuned MatterGen)

This guide walks through evaluating crystal structures from a generative model
end-to-end: from generated structures → CIFs → metrics. It covers the standard
LeMat-GenBench benchmarks (validity, S.U.N., stability, …) **plus the added
functional-property metrics**:

- **Band gap** (eV) — via a pluggable backend: `hamgnn` (ab-initio Hamiltonian,
  PBE-level) or `alignn` (fast MBJ surrogate).
- **Ion migration barrier** (eV) — BVSE percolation barriers via `bvlain`.

The whole flow is: **structures (CIF dir) → `lemat-genbench <input> <benchmark>` → `results.yaml`**.

---

## TL;DR

```bash
# 0. one-time: install the benchmark + property extras
uv sync --extra properties           # core deps + bvlain (migration barrier)

# 1. generate structures with your finetuned MatterGen, export CIFs to ./gen_cifs/ (see Step 2)

# 2. run the property evaluation (migration barrier is fast, no extra services)
uv run lemat-genbench ./gen_cifs migration_barrier -o results/migration.yaml

# 3. band gap (needs a backend — ALIGNN is in-env; HamGNN runs out-of-process, see Step 1c)
uv run lemat-genbench ./gen_cifs band_gap -o results/band_gap.yaml

# or both at once:
uv run lemat-genbench ./gen_cifs property -o results/property.yaml
```

The input can be a **directory of `.cif` files** or a **`.jsonl`** of pymatgen
`Structure` dicts (one per line).

---

## Step 1 — Environment setup

### 1a. Core benchmark environment

```bash
git clone https://github.com/dtsaras/lemat-genbench && cd lemat-genbench
uv sync                       # core: pymatgen, torch, the MLIP stack, CLI
uv sync --extra migration     # adds bvlain (ion migration-barrier metric)
# or: uv sync --extra properties  (== migration today; HamGNN is configured separately)
```

You can run the CLI with `uv run lemat-genbench …`.

### 1b. Band-gap backend — option A: ALIGNN (fast, in-env)

ALIGNN (`jv_mbj_bandgap`) is a structure→gap surrogate trained on MBJ gaps
(closer to experiment). It needs the ALIGNN stack (installed manually — `dgl` is
not on PyPI and must match your torch build) and a checkpoint dir:

```bash
uv pip install alignn jarvis-tools
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/repo.html   # match your torch/CUDA
export ALIGNN_MODELS_DIR=/path/to/alignn_models/alignn   # dir with jv_mbj_bandgap_alignn.zip
```

Then select it per run with `--config` or a config file setting `backend: alignn`
(see Step 3).

### 1c. Band-gap backend — option B: HamGNN (accurate, out-of-process)

HamGNN predicts the ab-initio Hamiltonian; it is **not** a structure→gap model —
each structure is preprocessed with OpenMX, so it runs in its **own pinned env**
and is invoked via subprocess. One-time setup:

```bash
bash scripts/hamgnn/setup_hamgnn.sh           # builds the hamgnn env + Intel-MPI runtime + checkpoint
# then download OpenMX DFT_DATA19 and export the HAMGNN_* vars it prints:
#   export LD_LIBRARY_PATH=<impi_rt>/lib:$LD_LIBRARY_PATH
#   export HAMGNN_ENV_BIN=...  HAMGNN_OPENMX_POSTPROCESS=...  HAMGNN_READ_OPENMX=...
#   export HAMGNN_MODEL_PKL=...  HAMGNN_PREDICTOR_SCRIPT=...  HAMGNN_DFT_DATA=...
python scripts/hamgnn/smoke_test.py           # validates one structure -> gap (Si ~0.8 eV)
```

Full details + the verified config: **`scripts/hamgnn/README.md`**.

> **Cost / coverage:** HamGNN is ~minutes/structure and limited to OpenMX's 77
> elements (H→Bi, no actinides). Use it for small/medium cells; use `alignn`
> (or fewer structures) for fast or large sweeps.

---

## Step 2 — Generate structures from your finetuned MatterGen

Run generation with your finetuned checkpoint, then export the structures as a
CIF directory (or `.jsonl`). With the MatterGen CLI:

```bash
# Unconditional generation from a finetuned checkpoint
mattergen-generate ./gen_out \
  --model_path=/path/to/your_finetuned_checkpoint \
  --batch_size=64 --num_batches=16            # -> ~1024 structures

# MatterGen writes generated_crystals_cif.zip (per-structure CIFs) and
# generated_crystals.extxyz (multi-frame). Unzip the CIFs for the benchmark:
mkdir -p gen_cifs && unzip -o ./gen_out/generated_crystals_cif.zip -d gen_cifs
```

If you only have the `.extxyz` (or any ASE-readable bundle), convert it to a CIF dir:

```python
# extxyz_to_cifs.py
import sys
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import os
atoms = read(sys.argv[1], index=":")          # all frames
os.makedirs(sys.argv[2], exist_ok=True)
for i, a in enumerate(atoms):
    AseAtomsAdaptor.get_structure(a).to(filename=f"{sys.argv[2]}/gen_{i:05d}.cif")
print(f"wrote {len(atoms)} CIFs")
```
```bash
uv run python extxyz_to_cifs.py ./gen_out/generated_crystals.extxyz gen_cifs
```

> If you finetuned MatterGen **conditionally** (e.g. on a target band gap or
> chemistry), generate with `--properties_to_condition_on=...` per the MatterGen
> docs; the evaluation below then tells you whether the *generated* structures
> actually hit those properties.

---

## Step 3 — Run the evaluation

There are two entry points:

- **`scripts/run_benchmarks.py`** — the comprehensive runner (recommended for
  evaluating a model). It loads a CIF directory, **always runs validity first and
  filters to valid structures**, then runs the benchmark *families* you request
  (incl. the new `migration_barrier` / `band_gap` / `property` families). Results
  → `results_final/<name>_<config>_<timestamp>.json`.
- **`lemat-genbench <input> <benchmark>`** — a lighter CLI that runs a single
  benchmark and writes a YAML. Good for quick, one-off property runs.

### 3a. Comprehensive run (recommended)

```bash
# Full model evaluation: validity (mandatory) + S.U.N. + stability + properties.
# 'migration_barrier' is cheap; 'band_gap' (HamGNN) is slow — include deliberately.
uv run scripts/run_benchmarks.py \
  --cifs ./gen_cifs \
  --config comprehensive \
  --name finetuned_mattergen_v1 \
  --families validity sun stability migration_barrier band_gap
```

- `--cifs` accepts a **directory of `.cif`** (scanned recursively) or a **text file
  of CIF paths** (one per line). `--csv` is also supported (a `structure`/`cif_string`
  column).
- `--families` selects what to run (validity always runs). Omit it for the default
  comprehensive set (distribution, diversity, novelty, uniqueness, hhi, sun,
  stability) — add `migration_barrier` / `band_gap` / `property` to include the
  functional-property metrics.
- Per-family settings can be added to the `--config` YAML under
  `migration_barrier_settings:` / `band_gap_settings:` / `property_settings:`
  (otherwise sensible defaults are used).

For the **band_gap** family with HamGNN, export the `HAMGNN_*` vars (Step 1c)
before launching. To use ALIGNN instead, set `band_gap_settings: {backend: alignn}`
in the config.

### 3b. Single benchmark (lighter CLI)

The CLI is `lemat-genbench <INPUT> <BENCHMARK> -o <OUTPUT>`. `BENCHMARK` is a
config name (auto-created with defaults on first use) or a path to a YAML config.

#### Ion migration barrier (fast — no GPU/services)

```bash
uv run lemat-genbench ./gen_cifs migration_barrier -o results/migration.yaml
```
Defaults: `mobile_ion: Li1+`, `dimensionality: 3d`, `fast_threshold: 0.6` eV.
To screen layered cathodes by their easiest-direction barrier, use a config with
`dimensionality: min` (a layered conductor reports its real in-plane barrier
instead of the 3D no-percolation sentinel).

#### Band gap

```bash
# HamGNN backend (export the HAMGNN_* vars from Step 1c first):
uv run lemat-genbench ./gen_cifs band_gap -o results/band_gap.yaml

# ALIGNN backend instead — make a config and point at it:
cat > bandgap_alignn.yaml <<'YAML'
type: band_gap
backend: alignn
preprocess: true
metal_threshold: 0.1
insulator_threshold: 3.0
n_jobs: 4
YAML
uv run lemat-genbench ./gen_cifs ./bandgap_alignn.yaml -o results/band_gap_alignn.yaml
```

#### Combined property benchmark (band gap + migration barrier)

```bash
uv run lemat-genbench ./gen_cifs property -o results/property.yaml
```

> **Note:** the lighter `lemat-genbench` CLI fully supports the **self-contained**
> benchmarks — `validity`, `migration_barrier`, `band_gap`, `property` (these
> preprocess internally). The **S.U.N. / stability / novelty / distribution**
> benchmarks need MLIP energies + fingerprints attached first, so run those via
> the comprehensive runner (3a), which performs that preprocessing.

A typical finetuned-MatterGen evaluation (via 3a) runs **validity + S.U.N. +
stability + the property metrics** together to answer: *are the generated crystals
valid, stable, novel, AND do they have the functional properties I finetuned for?*

---

## Step 4 — Reading the results

Each run prints a summary and writes `final_scores` to the YAML.

**Migration barrier** (`migration_barrier_*`):
- `fraction_fast_ion_conductors` — fraction of Li-bearing structures with barrier
  < `fast_threshold` (the headline score).
- `fraction_with_mobile_ion` — fraction of generated structures containing the ion.
- `fraction_percolating` — of ion-bearing structures, fraction with a connected path.
- `mean_barrier` / `median_barrier` — over the conducting subset (eV).

**Band gap** (`band_gap_*`):
- `mean_band_gap`, `min/max/std` (eV); metals are reported as 0.
- `metallic_ratio` / `semiconductor_ratio` / `insulator_ratio`.
- `fraction_in_target_window` — set `target_min`/`target_max` in the config to
  score "fraction of generated structures in my desired gap range."

**Worked sanity check** (real MP structures, verified):

| structure | HamGNN gap | BVlain barrier (min) |
|---|---|---|
| LiAlO₂ (main-group insulator) | 5.37 eV (insulator) | — |
| LiTiO₂ / LiVO₂ (TM-oxide cathodes) | 0 eV (metallic) | ~1.1 eV, percolating |
| LiF / Si (toy) | LiF 0 / Si 0.82 eV | LiF 0.54 eV / Si: no Li |

---

## Configuration reference

`migration_barrier.yaml` keys: `mobile_ion`, `dimensionality` (`1d`/`2d`/`3d`/`min`),
`r_cut`, `resolution`, `k`, `encut`, `fast_threshold`, `n_jobs`, `timeout`.

`band_gap.yaml` keys: `backend` (`hamgnn`/`alignn`), `backend_kwargs`, `preprocess`,
`metal_threshold`, `insulator_threshold`, `target_min`, `target_max`, `n_jobs`, `timeout`.

`property.yaml` keys: `include_band_gap`, `include_migration_barrier`,
`band_gap_backend`, plus the band-gap and migration keys above.

Configs live in `src/lemat_genbench/config/` (auto-created with defaults), or pass
any YAML path as the `<BENCHMARK>` argument.

---

## Accuracy caveats (important for interpretation)

- **HamGNN band gaps are PBE-level.** They are reliable for main-group / d⁰
  insulators (e.g. LiAlO₂ ≈ 5.4 eV) but **predict metallic (0 eV) for correlated
  transition-metal-oxide cathodes** (LiTiO₂, LiVO₂, LiCoO₂ …) because PBE closes
  their gap. For experiment-relevant gaps on those, use the `alignn` (MBJ) backend
  or a hybrid/GW reference. The gap is read along a symmetry k-path, so strongly
  indirect gaps may be approximate.
- **BVSE migration barriers are a relative screen.** They run on the
  fully-lithiated structure, so absolute values are systematically **higher** than
  vacancy-mediated NEB/experimental barriers (~1.1 eV here vs ~0.3–0.8 eV
  experimentally for layered oxides). Use them to **rank** generated structures,
  not as absolute activation energies.

---

## Troubleshooting

- **"Found 0 files"** — the input dir has no `.cif`/`.jsonl`. Point at the unzipped
  CIF directory (Step 2).
- **Band gap: `HamGNNNotConfigured`** — the `HAMGNN_*` env vars aren't set; run
  `scripts/hamgnn/setup_hamgnn.sh` and export the printed vars (or use `backend: alignn`).
- **`band gap = -X eV` clamped to 0** — that's a metallic prediction (overlapping
  bands), not an error; see the caveat above.
- **ALIGNN import errors** — `dgl` must match your torch build; if it won't resolve
  against torch 2.6, prefer the HamGNN backend or pin a compatible dgl.
- **Slow band-gap runs** — HamGNN is minutes/structure. Sweep with `alignn`, or
  evaluate a representative sample, then spot-check with HamGNN.
```
