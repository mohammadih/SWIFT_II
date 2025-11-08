# SWIFT II Code Documentation

This directory contains the implementation code for the DRL-assisted spectrum selection framework.

## Directory Structure

### Python (`Python/`)

Contains the Deep Reinforcement Learning implementation and simulation framework.

#### Core Modules

1. **`main.py`** - Main simulation script
   - Entry point for training and evaluation
   - Command-line interface
   - Orchestrates training pipeline

2. **`config.py`** - Configuration management
   - `ConfigManager`: Centralized configuration
   - `SpectrumConfig`: Spectrum parameters
   - `ChannelConfig`: 5G channel settings
   - `DDQNConfig`: Neural network hyperparameters
   - `RewardConfig`: Reward function weights
   - `SimulationConfig`: Simulation parameters

3. **`environment.py`** - MDP environment
   - `SpectrumEnvironment`: Base environment class
   - `MATLABPhyEnvironment`: Extended with MATLAB integration
   - Implements state/action/reward formulation
   - Handles environment dynamics

4. **`ddqn_agent.py`** - DDQN agent
   - `DDQNAgent`: Double Deep Q-Network implementation
   - `ReplayBuffer`: Experience replay memory
   - `DDQNTrainer`: Training loop management
   - CNN-based Q-network architecture

5. **`metrics.py`** - Performance metrics
   - `PerformanceMetrics`: Metrics tracking and computation
   - Throughput improvement calculation
   - Interference reduction analysis
   - Visualization and plotting

6. **`matlab_interface.py`** - MATLAB integration
   - `MATLABInterface`: Real MATLAB engine interface
   - `SimulatedMATLABInterface`: Python fallback
   - Channel simulation wrapper
   - PSD calculation interface

7. **`Channel.py`** - Alternative channel model
   - Sionna-based 5G channel simulation
   - CDL channel models
   - OFDM resource grid

8. **`DDQN.py`** - Legacy implementation
   - Original simplified DDQN
   - Retained for reference

#### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.10+
- NumPy
- Matplotlib
- MATLAB Engine for Python (optional)

### MATLAB (`Matlab/`)

Contains the physical layer simulation and signal processing code.

#### Main Scripts

1. **`ActivePassiveAI_Sensing.m`**
   - Main simulation script
   - Combines 5G signals with passive sensing
   - Performs successive interference cancellation (SIC)
   - Generates training data

2. **`my_NRChannel_modified.m`**
   - 5G NR channel model
   - OFDM implementation
   - Channel response calculation
   - Configurable channel types (CDL, TDL)

3. **`Passive_Signal.m`**
   - Generates passive sensor signals
   - Thermal noise modeling
   - Configurable power levels
   - Spectrum analysis

4. **`my_pspectrum.m`**
   - Power spectral density calculation
   - Custom spectrum analyzer
   - Multiple visualization modes

5. **`SIC1_DL_Reg.m` & `SIC2_DL_Reg.m`**
   - Successive Interference Cancellation
   - Deep learning-based regression
   - Signal reconstruction

#### Usage

Run in MATLAB:
```matlab
% Add path
addpath('path/to/Matlab')

% Run main script
ActivePassiveAI_Sensing
```

## Usage Guide

### Basic Training

```bash
cd Python
python main.py --episodes 1000
```

### Training with MATLAB

```bash
python main.py --matlab --matlab-path ../Matlab --episodes 1000
```

### Configuration

Modify parameters in `config.py` or create a JSON configuration file:

```python
from config import ConfigManager

config = ConfigManager()
config.ddqn.learning_rate = 0.0001
config.simulation.num_episodes = 2000
```

### Evaluation

```bash
python main.py --eval-only --model-path ./models/ddqn_final.h5
```

## Module Details

### Configuration Module (`config.py`)

#### Key Classes

- **`ConfigManager`**: Main configuration container
  - `.spectrum`: Spectrum settings
  - `.channel`: Channel model parameters
  - `.ddqn`: DDQN hyperparameters
  - `.reward`: Reward function weights
  - `.simulation`: Simulation settings

#### Example Configuration

```python
config = ConfigManager()
config.spectrum.num_channels = 7
config.ddqn.learning_rate = 0.0001
config.reward.alpha_throughput = 1.0
config.reward.beta_interference = 0.5
```

### Environment Module (`environment.py`)

#### Key Methods

- `reset()`: Initialize environment
- `step(action)`: Execute action and return (state, reward, done, info)
- `_calculate_sinr(channel)`: Compute SINR
- `_calculate_reward(action, sinr)`: Compute reward

#### State Representation

State shape: `(num_channels, num_features, temporal_depth)`
- `num_channels`: Communication + sensing channels
- `num_features`: [channel_idx, reward, interference, jammer]
- `temporal_depth`: Time steps (default: 4)

### DDQN Agent Module (`ddqn_agent.py`)

#### Network Architecture

```
Input: (3, 4, 4) state tensor
  ↓
Conv2D: 16 filters, (2,2) kernel, ReLU
  ↓
AveragePooling2D: (2,1)
  ↓
Conv2D: 32 filters, (2,2) kernel, ReLU
  ↓
Flatten
  ↓
Dense: 128 units, ReLU
  ↓
Dropout: 0.2
  ↓
Dense: 64 units, ReLU
  ↓
Dense: 7 units (Q-values), Linear
```

#### Training Algorithm

1. Select action using ε-greedy policy
2. Execute action in environment
3. Store experience in replay buffer
4. Sample minibatch and compute DDQN loss
5. Update online network
6. Periodically update target network
7. Decay exploration rate

### Metrics Module (`metrics.py`)

#### Tracked Metrics

- Episode rewards
- Throughput (bits/s/Hz)
- Interference power (mW)
- SINR (dB)
- Training loss
- Q-values
- Exploration rate (ε)

#### Computed Metrics

- **Throughput Improvement**: `ΔR = (R_DRL - R_baseline) / R_baseline × 100%`
- **Interference Reduction**: `ΔP = (P_baseline - P_DRL) / P_baseline × 100%`
- **Learning Stability**: `σ²_Q = Var(Q(s,a))`
- **Convergence Episode**: When 90% stable reward reached

### MATLAB Interface Module (`matlab_interface.py`)

#### Key Functions

- `start_engine()`: Initialize MATLAB engine
- `simulate_channel(signal)`: Run channel simulation
- `generate_passive_signal(power, samples)`: Generate passive signal
- `calculate_psd(signal, fs)`: Compute PSD

#### Usage

```python
from matlab_interface import create_matlab_interface

# With real MATLAB
with create_matlab_interface(use_real_matlab=True) as matlab:
    signal_out, response = matlab.simulate_channel(signal_in)

# With simulation
with create_matlab_interface(use_real_matlab=False) as matlab:
    signal_out, response = matlab.simulate_channel(signal_in)
```

## Testing

Run individual module tests:

```bash
# Test configuration
python config.py

# Test environment
python environment.py

# Test DDQN agent
python ddqn_agent.py

# Test metrics
python metrics.py

# Test MATLAB interface
python matlab_interface.py
```

## Output Files

### Training Outputs

- **Models**: `./models/ddqn_ep{episode}_{timestamp}.h5`
- **Figures**: `./Figures/training_curves_{timestamp}.png`
- **Metrics**: `./results/metrics_{timestamp}.json`
- **Logs**: `./logs/` (TensorBoard logs)

### Figure Types

1. **Training Curves**
   - Accumulated reward
   - Episode reward with moving average
   - DDQN loss convergence
   - Exploration rate decay
   - Average throughput
   - Average interference

2. **Action Distribution**
   - Channel selection frequency histogram

3. **Baseline Comparison**
   - Throughput comparison bar chart
   - Interference comparison bar chart

## Advanced Usage

### Custom Reward Function

Modify reward calculation in `environment.py`:

```python
def _calculate_reward(self, action: int, sinr: float) -> float:
    # Custom reward logic
    throughput_reward = custom_throughput_function(sinr)
    interference_penalty = custom_penalty_function(action)
    return throughput_reward - interference_penalty
```

### Custom Network Architecture

Modify network in `ddqn_agent.py`:

```python
def _build_network(self):
    model = models.Sequential([
        # Custom layers
    ])
    return model
```

### Integration with External Systems

Use the environment as a standard OpenAI Gym-like interface:

```python
from environment import SpectrumEnvironment
from config import ConfigManager

config = ConfigManager()
env = SpectrumEnvironment(config)

state = env.reset()
for _ in range(100):
    action = your_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()
```

## Troubleshooting

### MATLAB Integration Issues

1. **MATLAB Engine not found**
   ```bash
   python -m pip install matlabengine
   ```

2. **MATLAB path issues**
   ```python
   matlab.engine.start_matlab("-desktop")  # Debug mode
   ```

3. **Use simulated interface**
   ```bash
   python main.py --no-matlab
   ```

### TensorFlow Issues

1. **GPU not detected**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Memory issues**
   - Reduce batch size in config
   - Reduce replay buffer size

### Training Issues

1. **Not converging**
   - Adjust learning rate
   - Increase exploration period
   - Check reward scaling

2. **Slow training**
   - Reduce episode length
   - Reduce target update frequency
   - Use GPU acceleration

## Contact

For questions or issues:
- Check documentation in `SWIFT_II_OVERVIEW.md`
- Review code comments
- Open an issue on the repository