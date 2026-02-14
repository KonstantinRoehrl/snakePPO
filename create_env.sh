#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first: https://docs.conda.io/"
  exit 1
fi

echo "Creating conda environment 'snakePPO' from environment.yml..."
conda env create -f environment.yml

echo "Done. Activate with: conda activate snakePPO"
