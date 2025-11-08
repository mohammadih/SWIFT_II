# Code Documentation

## Overview

This directory contains the implementation of the SWIFT II DRL-assisted spectrum selection framework, organized into MATLAB and Python components.

## Directory Structure

### MATLAB (`/Matlab/`)

Contains 5G NR physical layer simulation components:

- **`ActivePassiveAI_Sensing.m`**: Main simulation script that orchestrates the active-passive coexistence scenario
- **`my_NRChannel_modified.m`**: Custom 5G NR channel model supporting HFS (Highly Frequency Selective), MFS (Moderately Frequency Selective), and FF (Frequency-Flat) channel profiles
- **`my_pspectrum.m`**: Power spectral density computation utilities
- **`Passive_Signal.m`**: Passive sensor signal generation
- **`SIC1_DL_Reg.m` / `SIC2_DL_Reg.m`**: Successive Interference Cancellation (SIC) neural network models for signal separation

### Python (`/Python/`)

Contains the DRL agent implementation:

- **`DDQN.py`**: Double Deep Q-Network implementation with:
  - Experience replay buffer
  - Target network updates
  - ε-greedy exploration with decay
  - Multi-objective reward function
  - Visualization of training progress
- **`Channel.py`**: Channel modeling utilities using Sionna library for OFDM channel simulation

## Key Parameters

### Environment Parameters (from DDQN.py)

```python
N = 7              # Total number of channels (bins)
Ns = 2             # Number of sensing channels
T = 3              # Temporal memory depth
lambda_val = 10    # Reward weight
mu_th = 2          # SINR threshold
nu_th = 2          # PSD threshold
gamma = 0.4        # Discount factor
sigma_noise = 1e-3 # Noise power in mW
```

### Learning Parameters

```python
epsilon = 1.0           # Initial exploration rate
epsilon_min = 0.01      # Minimum exploration rate
decay_rate = 0.995      # Epsilon decay rate
learning_rate = 0.0005  # Learning rate
batch_size = 32         # Batch size for training
replay_buffer_size = 50 # Replay buffer capacity
```

## Usage

### Running MATLAB Simulation

```matlab
cd Matlab
ActivePassiveAI_Sensing
```

This script:
1. Loads 5G NR signal data
2. Applies channel model (`my_NRChannel_modified`)
3. Generates passive sensor signal
4. Computes PSD using `my_pspectrum`
5. Applies SIC for interference cancellation
6. Visualizes results

### Running Python DRL Training

```python
cd Python
python DDQN.py
```

This script:
1. Initializes the DDQN agent
2. Runs training episodes
3. Updates Q-networks using experience replay
4. Visualizes:
   - Accumulated rewards
   - Loss convergence
   - Exploration rate decay
   - Channel selection frequency

## Integration Notes

The current implementation uses:
- **Sionna** (in `Channel.py`) as an alternative to MATLAB for channel modeling
- **Standalone Python DRL** (in `DDQN.py`) that simulates the environment internally

For full MATLAB-Python integration as described in the paper:
1. Use MATLAB Engine API for Python to call MATLAB functions from Python
2. Implement proper state/action/reward interface between MATLAB PHY layer and Python RL layer
3. Set up real-time data exchange (e.g., via ZMQ or shared memory)

## Channel Models

### MATLAB Channel Types (`my_NRChannel_modified.m`)

- **HFS (Highly Frequency Selective)**: Multiple delayed paths with significant frequency selectivity
- **MFS (Moderately Frequency Selective)**: Moderate delay spread with LOS component
- **FF (Frequency-Flat)**: Single path, minimal frequency selectivity

### Python Channel Model (`Channel.py`)

Uses Sionna's CDL (Clustered Delay Line) model with:
- CDL-D profile
- Configurable delay spread
- Doppler effects
- OFDM resource grid mapping

## Output Files

Training generates:
- Loss convergence plots
- Reward accumulation curves
- Exploration rate decay visualization
- Channel selection frequency histograms

MATLAB simulation generates:
- PSD spectrograms
- Channel response heatmaps
- Filtered signal visualizations

## References

- See main [README.md](../README.md) for system overview
- See [DOCUMENTATION.md](../DOCUMENTATION.md) for detailed algorithmic workflow
