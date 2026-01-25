import torch

GAME_SIZE = 16
NUM_ENVS = 64
ROLLOUT_STEPS = 4096
PPO_EPOCHS = 4
MINIBATCH_SIZE = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 2.5e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
MODEL_DIR = "../models"
LOG_DIR = "../logs"
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"