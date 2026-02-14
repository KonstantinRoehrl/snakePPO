#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

echo "Activating snakePPO and running: $*"
# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snakePPO

# If user wants the unsafe workaround, they can pass --allow-duplicate-omp
ALLOW_DUPLICATE=false
if [ "${1-}" = "--allow-duplicate-omp" ]; then
  ALLOW_DUPLICATE=true
  shift
fi

if [ "$ALLOW_DUPLICATE" = true ]; then
  export KMP_DUPLICATE_LIB_OK=TRUE
  echo "Warning: KMP_DUPLICATE_LIB_OK=TRUE set (unsafe workaround)."
fi

if [ "$#" -eq 0 ]; then
  python -u src/train.py
else
  python -u "$@"
fi
