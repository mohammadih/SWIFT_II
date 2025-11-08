# Python Code Documentation

## Overview

This directory contains Python implementations of the Deep Reinforcement Learning (DRL) agent for spectrum selection, using Double Deep Q-Network (DDQN) architecture.

## Files Description

### `DDQN.py`

Main implementation of the Double Deep Q-Network for dynamic spectrum selection.

**Key Components:**

1. **Environment Parameters**
   - `N = 7`: Total number of available channels
   - `Ns = 2`: Number of sensing channels
   - `T = 3`: Temporal memory depth for state representation
   - `lambda_val = 10`: Reward scaling factor
   - `mu_th = 2`: SINR threshold for communication reward
   - `nu_th = 2`: PSD threshold for sensing reward

2. **DDQN Class**
   - Online Q-network and target Q-network
   - CNN-based architecture for state feature extraction
   - Experience replay buffer integration
   - Target network periodic updates

3. **Reward Functions**
   - `communication_feedback()`: Multi-objective reward based on SINR, interference, and jammer power
   - `sensing_feedback()`: Binary reward based on PSD threshold

4. **State Representation**
   - Indication matrix: `[action, reward, interference_power, jammer_power]`
   - State matrix: Stacked history of `T` previous time steps
   - Shape: `(Ns+1, 4, T)` where:
     - `Ns+1`: Communication channel + sensing channels
     - `4`: Features per channel
     - `T`: Temporal depth

**Network Architecture:**
```python
Conv2D(10, (2,2), activation='relu')
↓
AveragePooling2D(pool_size=(2,1))
↓
Flatten()
↓
Dense(N, activation='linear')  # Q-values for N channels
```

**Training Process:**
1. Initialize environment and DDQN agent
2. For each episode:
   - Sample PSD values with jitter
   - Create state from history
   - Select action using ε-greedy policy
   - Compute reward
   - Store experience in replay buffer
   - Train on minibatch from replay buffer
   - Update target network periodically
   - Decay exploration rate

**Visualization:**
- Accumulated reward over iterations
- DDQN loss convergence
- Exploration rate (ε) decay
- Channel selection frequency histogram

**Usage:**
```python
python DDQN.py
```

### `Channel.py`

Channel modeling utilities using Sionna library for OFDM channel simulation.

**Key Features:**

1. **Resource Grid Configuration**
   - FFT size: 1028
   - Subcarrier spacing: 30 kHz
   - OFDM symbols: 14
   - Active subcarriers: 1024
   - Cyclic prefix: 72 samples

2. **CDL Channel Model**
   - CDL-D profile (delay spread: 30 ns)
   - Carrier frequency: 3.5 GHz
   - SISO antenna configuration
   - Doppler effects (speed: 30 m/s)

3. **Channel Response Extraction**
   - Full OFDM grid channel response
   - Active subcarrier extraction
   - Visualization of `|H(f,t)|` heatmap

**Usage:**
```python
python Channel.py
```

**Output:**
- Heatmap visualization of channel magnitude response on active subcarriers

## Dependencies

### Required Packages

```bash
pip install tensorflow>=2.8.0
pip install numpy
pip install matplotlib
pip install sionna  # For Channel.py
```

### Optional (for MATLAB integration)

```bash
pip install matlabengine  # MATLAB Engine API for Python
```

## Key Algorithms

### DDQN Update Rule

Standard Double DQN update to prevent Q-value overestimation:

```python
# Select action using online network
a_next = argmax(Q_online(s_next))

# Evaluate using target network
target = r + γ * Q_target(s_next, a_next)

# Update online network
loss = MSE(Q_online(s, a), target)
```

### ε-Greedy Exploration

```python
ε_t = max(ε_min, ε_0 * decay_rate^t)
if random() < ε_t:
    action = random_action()
else:
    action = argmax(Q(s))
```

### Experience Replay

- Buffer size: 50 experiences
- Batch size: 32
- Random sampling for decorrelation

## State-Action-Reward Design

### State Space

State matrix combines:
- **Communication channel indication**: `[action, reward, interference, jammer]`
- **Sensing channel indications**: Same format for each sensing channel
- **Temporal history**: Last `T` time steps

### Action Space

- Discrete: Select one channel from `N` available channels
- Action index: `0` to `N-1`

### Reward Function

Multi-objective reward balancing communication and sensing:

```python
r_comm = λ * max(0, SINR - μ_th)
r_penalty = -0.1 * (P_int + P_jammer)
reward = (r_comm + r_penalty) / λ
```

## Integration with MATLAB

### Current Status

The current implementation uses a **simplified environment** that simulates:
- PSD values with Gaussian jitter
- SINR computation
- Interference and jammer power

### Full Integration Approach

To integrate with MATLAB PHY layer:

1. **Setup MATLAB Engine:**
   ```python
   import matlab.engine
   eng = matlab.engine.start_matlab()
   eng.cd('path/to/matlab/code')
   ```

2. **Call MATLAB Functions:**
   ```python
   # Get PSD from MATLAB
   psd_matlab = eng.my_pspectrum(signal, fs, 'power', '')
   
   # Apply channel model
   signal_out = eng.my_NRChannel_modified(signal_in, 'HFS', False)
   
   # Convert MATLAB arrays to numpy
   psd_numpy = np.array(psd_matlab._data).reshape(psd_matlab.size)
   ```

3. **State Extraction:**
   ```python
   # Extract PSD values for each sub-band
   psd_values = extract_psd_per_channel(psd_numpy, N)
   
   # Compute CQI from SINR
   cqi = sinr_to_cqi(sinr)
   
   # Build state vector
   state = [psd_values, cqi, interference_ratio]
   ```

## Performance Metrics

The training script tracks:

1. **Accumulated Reward**: Cumulative reward over training
2. **Loss**: Q-network training loss
3. **Exploration Rate**: ε decay over time
4. **Channel Selection**: Frequency of each channel selection

## Customization

### Adjusting Parameters

Edit the parameter section at the top of `DDQN.py`:

```python
N = 10              # Increase number of channels
T = 5               # Increase temporal memory
lambda_val = 20     # Adjust reward scaling
learning_rate = 0.001  # Adjust learning rate
```

### Modifying Network Architecture

Edit the `build_network()` method:

```python
def build_network(self, input_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    # ... rest of the code
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `replay_buffer_size` or `batch_size`
2. **Slow Convergence**: Increase `learning_rate` or adjust `decay_rate`
3. **Poor Performance**: Tune reward weights (`lambda_val`, penalty coefficients)
4. **MATLAB Integration**: Ensure MATLAB Engine API is properly installed and MATLAB is in PATH

## References

- See main [README.md](../../README.md) for system overview
- See [DOCUMENTATION.md](../../DOCUMENTATION.md) for detailed algorithmic workflow
- See [Matlab/readme.md](../Matlab/readme.md) for MATLAB code documentation
