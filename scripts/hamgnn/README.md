# HamGNN band-gap backend — environment bring-up

The `hamgnn` band-gap backend (`lemat_genbench.properties.band_gap_backends.HamGNNBandGapBackend`)
predicts an electronic band gap by running HamGNN's documented pipeline as a
**subprocess** into a separate, pinned Python environment:

```
structure ──▶ poscar2openmx (×2: non-SOC + SOC)
          ──▶ openmx_postprocess   (non-SCF → overlap.scfout = S + H0)
          ──▶ graph_data_gen (×2 → two graph_data.npz)
          ──▶ Uni-HamiltonianPredictor.py  (→ hamiltonian.npy)
          ──▶ band_cal             (diagonalise H(k)/S(k) → "band gap = X eV")
```

HamGNN pins `pytorch-lightning 1.5.10` / torch 2.5, which **cannot** share
lemat-genbench's torch-2.6 env — hence the isolated env + subprocess design. The
lemat-genbench metric itself still runs in-process and is called like any other
benchmark; only the heavy/conflicting work is shelled out.

> **Accuracy note.** HamGNN reproduces the PBE-level DFT Hamiltonian, so the gap
> is PBE-quality (systematically underestimated vs. experiment) and is read along
> a symmetry k-path (`auto_mode`), which can miss strongly indirect gaps. Its
> advantage over a direct surrogate is the full, physically-grounded electronic
> structure. The `alignn` backend (`jv_mbj_bandgap`) targets near-experiment MBJ
> gaps and is a useful cross-check.

## Prerequisites (verified)

| Component | How |
|---|---|
| Intel-MPI runtime for the prebuilt `openmx_postprocess` | `conda create -p <dir> -c conda-forge impi_rt` — supplies `libmpi.so.12` / `libmpifort.so.12`. **Confirmed**: with these on `LD_LIBRARY_PATH`, the prebuilt `openmx_postprocess` runs (no recompile/HPC needed). |
| HamGNN env (python 3.9, torch 2.5, lightning 1.5.10, e3nn 0.5.0) | `environment.yml` here (CPU-default; standard channels) |
| HamGNN console scripts (`poscar2openmx`, `graph_data_gen`, `band_cal`) | `pip install -e <HamGNN repo>` into that env |
| OpenMX `DFT_DATA19` pseudopotentials | from the OpenMX source tarball (see below) |
| Uni-HamGNN checkpoint (`*.pkl`, ~204 MB) | Zenodo record **17239078** |

## One-shot setup

```bash
# From the lemat-genbench repo root:
bash scripts/hamgnn/setup_hamgnn.sh
```

This clones HamGNN, builds the `hamgnn` conda env + the `impi_rt` runtime,
installs the console scripts, and fetches the Uni-HamGNN checkpoint. It then
prints the `export HAMGNN_*` lines to add to your shell. `DFT_DATA19` is large
and licensed by OpenMX, so the script points you at the download rather than
fetching it automatically:

```bash
# DFT_DATA19 ships inside the OpenMX source tree:
wget https://www.openmx-square.org/openmx3.9.tar.gz
tar xzf openmx3.9.tar.gz   # → openmx3.9/DFT_DATA19/
export HAMGNN_DFT_DATA=$PWD/openmx3.9/DFT_DATA19
```

## Configuration (env vars read by the backend)

| Env var | Meaning |
|---|---|
| `HAMGNN_ENV_BIN` | `<conda>/envs/hamgnn/bin` (holds `poscar2openmx`, `graph_data_gen`, `band_cal`, `python`) |
| `HAMGNN_OPENMX_POSTPROCESS` | path to the `openmx_postprocess` binary |
| `HAMGNN_READ_OPENMX` | path to the `read_openmx` binary |
| `HAMGNN_MODEL_PKL` | Uni-HamGNN checkpoint `.pkl` |
| `HAMGNN_PREDICTOR_SCRIPT` | `<HamGNN repo>/Uni-HamGNN/Uni-HamiltonianPredictor.py` |
| `HAMGNN_DFT_DATA` | OpenMX `DFT_DATA19` dir |
| `HAMGNN_MPIRUN` / `HAMGNN_MPI_RANKS` | `mpirun` and rank count for `openmx_postprocess` (default `mpirun`, 1) |
| `HAMGNN_MPIRUN_EXTRA` | extra `mpirun` flags (e.g. `--allow-run-as-root`) |
| `HAMGNN_DEVICE` | `cpu` or `cuda` (default `cpu`) |
| `HAMGNN_NAO_MAX` | basis size (default `26`, the Uni-HamGNN value) |
| `HAMGNN_ENERGYCUTOFF` / `HAMGNN_KGRID` / `HAMGNN_NK` | OpenMX/band-cal knobs |
| `HAMGNN_TIMEOUT` | per-subprocess-step timeout (s, default 1800) |

When you also put the `impi_rt` libs on `LD_LIBRARY_PATH` (the setup script
prints the exact line), `openmx_postprocess` resolves its MPI libs.

## Smoke test (one structure → gap)

```bash
# Run from the lemat-genbench env (not the hamgnn env); the backend shells out.
python scripts/hamgnn/smoke_test.py     # builds Si, prints the predicted gap
```

If it prints a gap (Si ≈ 0.6–1.1 eV at PBE), the backend is wired correctly and
`lemat-genbench <cifs> band_gap` (with `backend: hamgnn`) will work. If a step
fails, the error includes the failing subprocess's stdout/stderr; common fixes:
set `HAMGNN_MPIRUN_EXTRA=--allow-run-as-root`, point `HAMGNN_DFT_DATA` at a valid
`DFT_DATA19`, or add `nequip` to the env if the predictor imports it.

> The subprocess wiring follows HamGNN's documented CLIs/configs. If a step's
> output filename/layout differs on your HamGNN version, adjust the small
> `_write_*_yaml` / `_locate_dat` helpers in `band_gap_backends.py` — the smoke
> test is the place to validate this end-to-end.

## Verified end-to-end (reference)

This pipeline has been run end-to-end on diamond-cubic **Si → band gap ≈ 0.82 eV**
(PBE-level, as expected) with the Uni-HamGNN checkpoint. Notes captured from that
run, in case they help on a new machine:

- **No `mpirun` needed for single-rank.** With `HAMGNN_MPI_RANKS=1` the backend
  invokes `openmx_postprocess` directly; only the Intel-MPI runtime libs need to
  be on `LD_LIBRARY_PATH` (the `impi_rt` conda env supplies `libmpi.so.12`). This
  sidesteps needing a working `mpiexec.hydra` launcher.
- **`band_cal` `auto_mode` needs `seekpath`** — included in `environment.yml`.
- The Uni-HamGNN model uses `nao_max=26` and requires **both** a non-SOC and a
  SOC `graph_data.npz` (two `openmx_postprocess` passes), which the backend
  generates automatically.
- Set `HAMGNN_KEEP_WORKDIR=1` to retain the per-structure temp dir
  (`overlap.scfout`, `graph_data.npz`, `hamiltonian.npy`, band output) for
  debugging.
