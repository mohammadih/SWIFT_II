import os

# Data configuration
DATA_PATH = os.path.join("data", "iq_samples.h5")  # Update extension if needed
SAMPLE_RATE = 15.36e6  # Hz, typical 5G NR sampling rate example
N_FFT = 4096
N_SUBBANDS = 16

# RL / DDQN hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEM_CAPACITY = 100_000
TARGET_UPDATE = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5e-5

# Reward weighting factors
ALPHA_RATE = 1.0  # Weight for communication performance (throughput/SINR)
BETA_INT = 2.0    # Weight for passive sensor interference penalty

# Training schedule
EPISODES = 400
STEPS_PER_EPISODE = 256
