from pathlib import Path

import torch

# Get the directory of this config file
_CONFIG_DIR = Path(__file__).parent

GAME_SIZE = 16
EPOCHS = 50
NUM_ENVS = 64
ROLLOUT_STEPS = 4096
# During training, evaluation is stopped if no apple is eaton for EVAL_STEPS steps
EVAL_STEPS = 512
PPO_EPOCHS = 4
MINIBATCH_SIZE = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 2.5e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
MODEL_DIR = str(_CONFIG_DIR.parent / "models")
LOG_DIR = str(_CONFIG_DIR.parent / "logs")
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"