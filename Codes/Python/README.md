# SWIFT II - Deep Reinforcement Learning-Assisted Spectrum Selection

## Overview

This repository implements a **Deep Reinforcement Learning (DRL)–assisted dynamic spectrum selection framework** that enables coexistence between 5G active systems and passive RF sensors within a shared spectrum environment. The implementation is based on the paper:

**"SWIFT II – Deep Reinforcement Learning-Assisted Spectrum Selection for RFI-Resilient Passive Sensing in Shared 5G Environments"**

### System Architecture

The system consists of three main interacting components:

1. **5G Transceiver (gNB/UE Pair):** Generates and transmits 5G NR-compliant signals within a shared band.
2. **Passive Sensor Node:** Continuously senses the spectral environment and provides **Power Spectral Density (PSD)** feedback.
3. **DRL-based Spectrum Agent:** Acts as the decision-making controller that dynamically allocates or avoids frequency channels to protect passive sensors.

## Algorithmic Framework

### Environment Definition

The environment is modeled as a **Markov Decision Process (MDP)**:

- **State (sₜ):** `[PSD, CQI, interference_ratio]` - Concatenation of spectral occupancy features, interference level estimates, and channel quality indicators
- **Action (aₜ):** Selection of sub-bands or RB groups for 5G transmission
- **Reward (rₜ):** `α * log₂(1 + throughput) - β * interference_ratio` - Jointly considers 5G throughput and interference power reduction

### Double Deep Q-Network (DDQN)

The implementation uses **DDQN** to address Q-value overestimation and convergence instability:

- **Online Network:** `Q_θ(s_t, a_t)` - Updated every step
- **Target Network:** `Q_θ⁻(s_t, a_t)` - Updated every C steps

**Update Rule:**
```
L(θ) = E[(r_t + γ Q_θ⁻(s_{t+1}, argmax_a' Q_θ(s_{t+1}, a')) - Q_θ(s_t, a_t))²]
```

### Reward Function

The reward function balances 5G throughput and interference mitigation:

```
r_t = α * log₂(1 + γ_t) - β * I_t
```

where:
- `α`: Throughput weight
- `γ_t`: SINR/throughput
- `β`: Interference weight  
- `I_t`: Interference ratio (P_int / P_th)

## File Structure

```
Python/
├── ddqn_spectrum_selection.py    # DDQN agent implementation
├── spectrum_environment.py        # Environment class (MDP)
├── train_ddqn_spectrum.py         # Main training script
├── matlab_interface.py            # MATLAB-Python integration
├── Channel.py                     # Channel modeling (Sionna)
├── DDQN.py                        # Original DDQN implementation
└── README.md                      # This file
```

## Installation

### Requirements

```bash
pip install tensorflow>=2.10.0
pip install numpy
pip install matplotlib
```

### Optional: MATLAB Integration

For MATLAB-Python co-simulation:

1. Install MATLAB Engine API for Python:
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```

2. Ensure MATLAB functions are in path:
   - `my_NRChannel_modified.m`
   - `Passive_Signal.m`
   - `my_pspectrum.m`

## Usage

### Basic Training

Run the main training script:

```bash
python train_ddqn_spectrum.py
```

This will:
1. Initialize the environment and DDQN agent
2. Train for 2000 episodes
3. Save checkpoints and training statistics
4. Generate training curves

### Custom Configuration

Modify parameters in `train_ddqn_spectrum.py`:

```python
# Environment parameters
num_subbands = 7
transmit_power = 5e-3  # 5 mW
noise_power = 1e-3

# Reward weights
alpha = 1.0  # Throughput weight
beta = 1.0   # Interference weight

# Agent parameters
learning_rate = 0.0005
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

### MATLAB Integration

To use MATLAB for PHY simulation:

```python
from matlab_interface import start_matlab_engine, MATLABInterface
from spectrum_environment import SpectrumEnvironment

# Start MATLAB engine
eng = start_matlab_engine()
matlab_iface = MATLABInterface(eng, matlab_path='../Matlab')

# Create environment with MATLAB
env = SpectrumEnvironment(
    num_subbands=7,
    use_matlab=True,
    matlab_engine=matlab_iface
)
```

## Training Workflow

The training follows this algorithmic workflow:

```python
Initialize environment E (MATLAB channel + PSD sensing model)
Initialize replay buffer D with capacity M
Initialize DDQN networks Qθ and Qθ− with random weights
Set ε = ε₀, learning rate η, discount γ, and target update rate C

for episode = 1 : N_episodes:
    Reset environment → s₀
    for t = 1 : T:
        # Step 1: Select action
        if random() < ε:
            a_t = random_action()
        else:
            a_t = argmax_a Qθ(s_t, a)
        
        # Step 2: Apply action in MATLAB PHY model
        throughput, interference, PSD_next = simulate_PHY(a_t)
        
        # Step 3: Compute reward
        r_t = α * log₂(1 + throughput) - β * interference
        
        # Step 4: Observe new state
        s_{t+1} = [PSD_next, CQI_next, interference]
        
        # Step 5: Store experience
        D.append((s_t, a_t, r_t, s_{t+1}))
        
        # Step 6: Sample minibatch and update Q-network
        if len(D) > batch_size:
            batch = random_sample(D)
            Perform DDQN update using loss L(θ)
        
        # Step 7: Update target network
        if t % C == 0:
            θ⁻ ← θ
        
        s_t ← s_{t+1}
    
    Decay ε after each episode
```

## Output Metrics

After training, the system evaluates:

- **Average throughput improvement:**
  ```
  ΔR = (R_DRL - R_baseline) / R_baseline × 100%
  ```

- **Interference power reduction:**
  ```
  ΔP_int = (P_baseline - P_DRL) / P_baseline × 100%
  ```

- **Learning stability:** Average Q-value variance across episodes

- **Convergence rate:** Episode index at 90% stable reward

## Results

The trained agent demonstrates:

- **40–50% interference mitigation** with minimal throughput degradation
- Optimal avoidance behavior without explicit coordination
- Outperforms heuristic or fixed-threshold selection schemes

## Visualization

Training generates several plots:

1. **Episode Rewards:** Shows learning progress
2. **Throughput:** Average throughput per episode
3. **Interference:** Average interference per episode
4. **DDQN Loss:** Convergence of Q-network
5. **Exploration Rate:** Epsilon decay over time
6. **Episode Lengths:** Steps per episode

All plots are saved in the checkpoint directory.

## Checkpoints

Training checkpoints are saved in `./checkpoints/ddqn_training_<timestamp>/`:

- `ddqn_model_ep<X>.h5`: Saved model at episode X
- `training_stats_ep<X>.json`: Training statistics
- `training_curves.png`: Visualization plots
- `evaluation_metrics.json`: Final evaluation results

## Loading a Trained Model

```python
from ddqn_spectrum_selection import DDQNAgent

# Create agent with same architecture
agent = DDQNAgent(state_shape=(7, 3), num_actions=7)

# Load trained weights
agent.load_model('checkpoints/ddqn_training_XXX/ddqn_model_final.h5')

# Use for inference
action = agent.select_action(state, training=False)
```

## MATLAB Functions

The MATLAB layer implements realistic 5G NR physical layer:

- **OFDM waveform generator:** `nrWaveformGenerator`
- **CDL channel models:** `nrCDLChannel` (CDL-C, CDL-D, etc.)
- **Spectrum analyzer:** Extracts `PSD(f_i)` and interference maps

Key MATLAB functions:
- `my_NRChannel_modified.m`: Applies 3GPP CDL channel models
- `Passive_Signal.m`: Generates passive sensor noise signal
- `my_pspectrum.m`: Computes power spectral density

## References

- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Haykin (2008): Adaptive Dynamic Systems
- Hong et al. (2021): Federated edge-based training extensions

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
