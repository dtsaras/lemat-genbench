#!/usr/bin/env bash
# Stand up the isolated HamGNN inference environment for the `hamgnn` band-gap
# backend. Idempotent-ish: skips steps whose outputs already exist.
#
# Usage:  bash scripts/hamgnn/setup_hamgnn.sh
# Override locations via env vars before running (see defaults below).
set -euo pipefail

HAMGNN_REPO="${HAMGNN_REPO:-$HOME/HamGNN}"
HAMGNN_ENV_NAME="${HAMGNN_ENV_NAME:-hamgnn}"
IMPI_PREFIX="${IMPI_PREFIX:-$HOME/.openmx_impi_rt}"
CKPT_DIR="${CKPT_DIR:-$HOME/hamgnn_models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 1/5  Clone HamGNN -> $HAMGNN_REPO"
if [ ! -d "$HAMGNN_REPO/.git" ]; then
  git clone https://github.com/QuantumLab-ZY/HamGNN "$HAMGNN_REPO"
else
  echo "    already present, skipping"
fi

echo "==> 2/5  Intel-MPI runtime (for the prebuilt openmx_postprocess) -> $IMPI_PREFIX"
if [ ! -e "$IMPI_PREFIX/lib/libmpi.so.12" ]; then
  conda create -y -p "$IMPI_PREFIX" -c conda-forge impi_rt
else
  echo "    already present, skipping"
fi

echo "==> 3/5  HamGNN conda env ($HAMGNN_ENV_NAME) + console scripts"
if ! conda env list | grep -qE "/${HAMGNN_ENV_NAME}$|^${HAMGNN_ENV_NAME}\s"; then
  conda env create -f "$SCRIPT_DIR/environment.yml" -n "$HAMGNN_ENV_NAME"
else
  echo "    env exists, skipping create"
fi
conda run -n "$HAMGNN_ENV_NAME" pip install -e "$HAMGNN_REPO"

echo "==> 4/5  Uni-HamGNN checkpoint (Zenodo 17239078) -> $CKPT_DIR"
mkdir -p "$CKPT_DIR"
if ! ls "$CKPT_DIR"/*.pkl >/dev/null 2>&1; then
  # Asset name on the record; adjust if Zenodo renames it.
  ( cd "$CKPT_DIR" && \
    wget -c "https://zenodo.org/records/17239078/files/uni-hamgnn_2_1.pkl.zip" && \
    unzip -o uni-hamgnn_2_1.pkl.zip )
else
  echo "    checkpoint present, skipping"
fi

ENV_BIN="$(conda run -n "$HAMGNN_ENV_NAME" python -c 'import sys,os;print(os.path.dirname(sys.executable))')"
OMX_DIR="$HAMGNN_REPO/DFT_interfaces/openmx/openmx_postprocess"
CKPT="$(ls "$CKPT_DIR"/*.pkl 2>/dev/null | head -1 || true)"

cat <<EOF

==> 5/5  DONE. Add these to your shell (and download DFT_DATA19 — see README):

  export LD_LIBRARY_PATH="$IMPI_PREFIX/lib:\$LD_LIBRARY_PATH"
  export HAMGNN_ENV_BIN="$ENV_BIN"
  export HAMGNN_OPENMX_POSTPROCESS="$OMX_DIR/openmx_postprocess"
  export HAMGNN_READ_OPENMX="$OMX_DIR/read_openmx"
  export HAMGNN_PREDICTOR_SCRIPT="$HAMGNN_REPO/Uni-HamGNN/Uni-HamiltonianPredictor.py"
  export HAMGNN_MODEL_PKL="${CKPT:-<path-to-uni-hamgnn>.pkl}"
  export HAMGNN_DFT_DATA="<path-to>/openmx3.9/DFT_DATA19"   # see README
  # export HAMGNN_MPIRUN_EXTRA="--allow-run-as-root"        # if running as root

Then validate end-to-end:
  python scripts/hamgnn/smoke_test.py
EOF
