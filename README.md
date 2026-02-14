# snakePPO environment

This workspace includes an environment recipe and a helper script to create a conda environment named `snakePPO` with the latest compatible Python, NumPy, Pygame, and PyTorch.

Quick create (recommended):

```bash
cd /Users/konstantin/Documents/Hobby_Programming/SnakePPO
bash create_env.sh
conda activate snakePPO
```

Alternative (manual) commands:

```bash
# create with a recent Python (adjust version if needed)
conda create -n snakePPO python=3.11 numpy pygame pip -y
conda activate snakePPO
# install PyTorch latest via pip (CPU/GPU selection handled by pip wheels)
pip install --upgrade pip
pip install torch torchvision torchaudio
```

Files added:

- [environment.yml](environment.yml)
- [create_env.sh](create_env.sh)
