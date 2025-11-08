# SWIFT II - Deep Reinforcement Learning-Assisted Spectrum Selection

## Project Overview

This repository implements a **Deep Reinforcement Learning (DRL)–assisted dynamic spectrum selection framework** that enables coexistence between 5G active systems and passive RF sensors within a shared spectrum environment. The implementation is based on the research paper:

**"SWIFT II – Deep Reinforcement Learning-Assisted Spectrum Selection for RFI-Resilient Passive Sensing in Shared 5G Environments"**

### System Architecture

The system consists of three main interacting components:

1. **5G Transceiver (gNB/UE Pair):** Generates and transmits 5G NR-compliant signals within a shared band.
2. **Passive Sensor Node:** Continuously senses the spectral environment and provides **Power Spectral Density (PSD)** feedback.
3. **DRL-based Spectrum Agent:** Acts as the decision-making controller that dynamically allocates or avoids frequency channels to protect passive sensors.

## Repository Structure

```
/workspace/
├── Codes/
│   ├── Python/
│   │   ├── ddqn_spectrum_selection.py    # Enhanced DDQN agent implementation
│   │   ├── spectrum_environment.py      # Environment class (MDP)
│   │   ├── train_ddqn_spectrum.py       # Main training script
│   │   ├── matlab_interface.py          # MATLAB-Python integration
│   │   ├── Channel.py                   # Channel modeling (Sionna)
│   │   ├── DDQN.py                      # Original DDQN implementation
│   │   ├── README.md                    # Python code documentation
│   │   └── ALGORITHM.md                 # Detailed algorithmic workflow
│   │
│   └── Matlab/
│       ├── my_NRChannel_modified.m      # 3GPP CDL channel models
│       ├── Passive_Signal.m             # Passive sensor signal generation
│       ├── my_pspectrum.m               # Power spectral density computation
│       ├── ActivePassiveAI_Sensing.m    # Main MATLAB simulation script
│       ├── SIC1_DL_Reg.m                # Signal interference cancellation
│       └── SIC2_DL_Reg.m                # Additional SIC methods
│
├── Figures/
│   ├── Channel Selection Frequency.png
│   ├── DDQN Loss Convergence.png
│   ├── Exploration Rate Decay.png
│   ├── HFS_Channel_2D.png
│   ├── HFS_Channel_3D.png
│   ├── Overal Concept.png
│   ├── Passive Signal_Spectrum.png
│   ├── Received Signal Spectrum.png
│   ├── Selected_Spectrum.png
│   └── SIC-system_model_architecture.png
│
└── README.md                            # This file
```

## Quick Start

### Python Implementation

1. **Install Dependencies:**
   ```bash
   cd Codes/Python
   pip install tensorflow numpy matplotlib
   ```

2. **Run Training:**
   ```bash
   python train_ddqn_spectrum.py
   ```

3. **View Results:**
   - Training curves are saved in `./checkpoints/ddqn_training_<timestamp>/`
   - Model checkpoints and statistics are saved automatically

### MATLAB Integration (Optional)

For full MATLAB-Python co-simulation:

1. **Install MATLAB Engine API for Python:**
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```

2. **Update `train_ddqn_spectrum.py` to enable MATLAB:**
   ```python
   env = SpectrumEnvironment(use_matlab=True, matlab_engine=matlab_iface)
   ```

## Key Features

### 1. Double Deep Q-Network (DDQN)

- Addresses Q-value overestimation in standard DQN
- Uses separate online and target networks
- CNN-based feature extraction from spectral states
- Experience replay for stable learning

### 2. State Representation

The state combines:
- **PSD (Power Spectral Density):** Normalized per sub-band
- **CQI (Channel Quality Indicator):** Derived from SINR estimates
- **Interference Ratio:** Normalized interference power

### 3. Reward Function

Balances 5G throughput and interference mitigation:

```
r_t = α * log₂(1 + throughput) - β * interference_ratio
```

### 4. MATLAB-Python Co-simulation

- Realistic 5G NR physical-layer modeling
- CDL channel models (HFS, MFS, FF)
- OFDM waveform generation
- Spectrum analysis and PSD extraction

## Algorithmic Workflow

The training follows this workflow:

```
1. Initialize environment and DDQN networks
2. For each episode:
   a. Reset environment → s₀
   b. For each time step:
      - Select action using ε-greedy policy
      - Apply action in PHY model
      - Compute reward (throughput - interference)
      - Observe new state
      - Store experience in replay buffer
      - Update Q-network using DDQN
      - Update target network periodically
   c. Decay exploration rate
3. Evaluate performance metrics
```

See `Codes/Python/ALGORITHM.md` for detailed pseudocode and implementation details.

## Performance Metrics

The system evaluates:

- **Average throughput improvement:** `ΔR = (R_DRL - R_baseline) / R_baseline × 100%`
- **Interference power reduction:** `ΔP_int = (P_baseline - P_DRL) / P_baseline × 100%`
- **Learning stability:** Q-value variance across episodes
- **Convergence rate:** Episode index at 90% stable reward

### Expected Results

- **40–50% interference mitigation** with minimal throughput degradation
- Optimal avoidance behavior without explicit coordination
- Outperforms heuristic or fixed-threshold selection schemes

## Documentation

- **Python Code:** See `Codes/Python/README.md`
- **Algorithm Details:** See `Codes/Python/ALGORITHM.md`
- **MATLAB Functions:** See `Codes/Matlab/readme.md`

## Visualization

Training generates several plots:

1. Episode Rewards (with moving average)
2. Average Throughput per Episode
3. Average Interference per Episode
4. DDQN Loss Convergence
5. Exploration Rate Decay
6. Episode Lengths

All plots are automatically saved in the checkpoint directory.

## Configuration

Key parameters can be adjusted in `train_ddqn_spectrum.py`:

```python
# Environment
num_subbands = 7
transmit_power = 5e-3  # 5 mW
noise_power = 1e-3

# Reward weights
alpha = 1.0  # Throughput weight
beta = 1.0   # Interference weight

# Agent
learning_rate = 0.0005
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

## References

- **Sutton & Barto (2018):** Reinforcement Learning: An Introduction
- **Haykin (2008):** Adaptive Dynamic Systems
- **Hong et al. (2021):** Federated edge-based training extensions

## Citation

If you use this code, please cite:

```
SWIFT II – Deep Reinforcement Learning-Assisted Spectrum Selection 
for RFI-Resilient Passive Sensing in Shared 5G Environments
```

## License

[Specify your license here]

## Contact

[Your contact information]
