# Algorithmic Workflow: DRL-Assisted Spectrum Selection

## Overview

This document provides a detailed description of the algorithmic workflow for the **Deep Reinforcement Learning-Assisted Spectrum Selection (DRL-SS)** framework, as implemented in this codebase.

## 1. Environment Definition (Hybrid PHY + RL Layer)

The environment emulates the shared spectrum between the **5G NR transceiver** and the **passive sensor**. Each time step represents one transmission interval or sensing cycle.

### Input Signals and Parameters

- **F = {f₁, f₂, ..., fₙ}**: Set of available sub-bands / RB groups
- **P_tx**: Transmit power of the 5G node
- **H(f_i, t)**: Channel response per sub-band (from MATLAB channel model)
- **PSD(f_i, t)**: Measured power spectral density from the passive sensor
- **σ²**: Thermal noise level
- **τ**: Sensing feedback latency

### Outputs (to RL agent)

- **Spectrum occupancy map**: `S_t = [PSD(f₁,t), ..., PSD(fₙ,t)]`
- **Channel quality indicators (CQI)**: Derived from SINR estimates
- **Interference ratio**: `I_t = P_int / P_th`

## 2. State, Action, and Reward Design

| Element            | Description                                                                 | Mathematical Representation                       |
| ------------------ | --------------------------------------------------------------------------- | ------------------------------------------------- |
| **State** (s_t)    | Concatenation of spectral features and interference feedback                | `s_t = [PSD_t, CQI_t, I_t]`                      |
| **Action** (a_t)   | Select one or multiple sub-bands for 5G transmission                        | `a_t ∈ F`                                         |
| **Reward** (r_t)   | Balances 5G throughput and interference mitigation                          | `r_t = α log₂(1 + γ_t) - β I_t`                  |
| **Transition**     | Environment updates (s_{t+1}) via spectrum dynamics and channel evolution  | `s_{t+1} = f(s_t, a_t, H_t)`                      |

### State Representation

The state is a 2D matrix of shape `(num_subbands, 3)`:

```python
state = [
    [PSD_normalized, CQI_normalized, interference_ratio],  # Sub-band 1
    [PSD_normalized, CQI_normalized, interference_ratio],  # Sub-band 2
    ...
]
```

Where:
- **PSD_normalized**: Normalized power spectral density per sub-band
- **CQI_normalized**: Normalized channel quality indicator (from SINR)
- **interference_ratio**: Ratio of interference power to noise power

### Action Space

Actions are discrete sub-band indices: `a_t ∈ {0, 1, ..., N-1}`

### Reward Function

```python
r_t = α * log₂(1 + throughput) - β * interference_ratio
```

Parameters:
- **α**: Throughput weight (default: 1.0)
- **β**: Interference weight (default: 1.0)
- **throughput**: Achieved spectral efficiency (bits/s/Hz)
- **interference_ratio**: Normalized interference power

## 3. Deep RL Agent: Double Deep Q-Network (DDQN)

Two networks are maintained:

- **Online Network:** `Q_θ(s_t, a_t)` - Updated every training step
- **Target Network:** `Q_θ⁻(s_t, a_t)` - Updated every C steps (default: 100)

### Network Architecture

The Q-network uses CNN-based feature extraction:

```
Input: (num_subbands, 3, 1)  # Add channel dimension for CNN
  ↓
Conv2D(32, 3x3) + ReLU
  ↓
Conv2D(64, 3x3) + ReLU
  ↓
AveragePooling2D(2x2)
  ↓
Flatten
  ↓
Dense(128) + ReLU
  ↓
Dense(64) + ReLU
  ↓
Output: Dense(num_actions)  # Q-values for each sub-band
```

### Update Rule

**DDQN Loss Function:**

```
L(θ) = E[(r_t + γ Q_θ⁻(s_{t+1}, argmax_{a'} Q_θ(s_{t+1}, a')) - Q_θ(s_t, a_t))²]
```

**Gradient Update:**

```
θ ← θ - η ∇_θ L(θ)
```

Where:
- **γ**: Discount factor (default: 0.95)
- **η**: Learning rate (default: 0.0005)
- **C**: Target network update frequency (default: 100 steps)

### Exploration Strategy

**ε-greedy with exponential decay:**

```
ε_t = max(ε_min, ε₀ * e^(-k*t))
```

Parameters:
- **ε₀**: Initial exploration rate (default: 1.0)
- **ε_min**: Minimum exploration rate (default: 0.01)
- **k**: Decay rate (default: 0.995 per episode)

## 4. Algorithm Pseudocode

```python
# Initialize
Initialize environment E (MATLAB channel + PSD sensing model)
Initialize replay buffer D with capacity M
Initialize DDQN networks Qθ and Qθ− with random weights
Set ε = ε₀, learning rate η, discount γ, and target update rate C

# Training Loop
for episode = 1 : N_episodes:
    Reset environment → s₀
    
    for t = 1 : T:
        # Step 1: Select action using ε-greedy policy
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
        
        # Step 5: Store experience in replay buffer
        D.append((s_t, a_t, r_t, s_{t+1}, done))
        
        # Step 6: Sample minibatch and update Q-network
        if len(D) > batch_size:
            batch = random_sample(D, batch_size)
            # Compute targets using DDQN update rule
            for (s, a, r, s_next, done) in batch:
                if done:
                    target = r
                else:
                    a_next = argmax_a' Qθ(s_next, a')  # Online network selects
                    target = r + γ * Qθ⁻(s_next, a_next)  # Target network evaluates
                
                # Update Q-value
                Qθ(s, a) ← target
            
            # Perform gradient descent
            θ ← θ - η ∇_θ L(θ)
        
        # Step 7: Update target network periodically
        if t % C == 0:
            θ⁻ ← θ
        
        s_t ← s_{t+1}
    
    # Decay exploration rate
    ε ← max(ε_min, ε * decay_rate)
```

## 5. Simulation Loop and MATLAB Interface

### MATLAB Layer

Implements realistic 5G NR physical layer:

- **OFDM waveform generator:** `nrWaveformGenerator`
- **CDL channel models:** `nrCDLChannel` (CDL-C, CDL-D, etc.)
- **Spectrum analyzer:** Extracts `PSD(f_i)` and interference maps

**MATLAB Function Interface:**

```matlab
[throughput, PSD_next, CQI_next, interference] = PHY_simulate(action, params);
```

### Python Layer (TensorFlow 2.x)

- Handles DDQN learning, replay memory, model storage
- TensorBoard for convergence monitoring (optional)
- Visualization of training curves

## 6. Output Metrics and Analysis

After convergence, the system evaluates:

### Performance Metrics

1. **Average throughput improvement:**
   ```
   ΔR = (R_DRL - R_baseline) / R_baseline × 100%
   ```

2. **Interference power reduction:**
   ```
   ΔP_int = (P_baseline - P_DRL) / P_baseline × 100%
   ```

3. **Learning stability:**
   ```
   σ²_Q = Var(Q_values) across episodes
   ```

4. **Convergence rate:**
   ```
   Episode index at 90% stable reward
   ```

### Expected Results

- **40–50% interference mitigation** with minimal throughput degradation
- Optimal avoidance behavior without explicit coordination
- Outperforms heuristic or fixed-threshold selection schemes

## 7. Block Diagram Flow

```
┌───────────────────────────────┐
│     Spectrum Environment      │
│  (5G TX + Passive Sensor)     │
│                               │
│  - PSD measurement            │
│  - Channel estimation         │
│  - Interference computation   │
└──────────────┬────────────────┘
               │ PSD + CQI feedback
               │
┌──────────────▼───────────────┐
│    DRL Agent (DDQN)          │
│                               │
│  State → Action (sub-band)   │
│  Reward ← Interference & R5G │
│                               │
│  - Online Q-network          │
│  - Target Q-network          │
│  - Experience replay          │
└──────────────┬───────────────┘
               │ Action (a_t)
               │
┌──────────────▼───────────────┐
│ MATLAB PHY Simulation Layer  │
│                               │
│  - OFDM waveform generation  │
│  - CDL channel model         │
│  - RFI model                 │
│  - Spectrum analysis         │
└──────────────┬───────────────┘
               │ Updated PSD, CQI
        ←──────┘
```

## 8. Implementation Details

### State Computation

```python
def _compute_state(self) -> np.ndarray:
    # Normalize PSD values
    psd_normalized = self.psd_values / np.max(self.psd_values)
    
    # Compute CQI from SINR
    sinr_per_subband = self._compute_sinr()
    cqi = np.log2(1 + sinr_per_subband)
    cqi_normalized = cqi / np.max(cqi)
    
    # Interference ratio
    interference_ratio = self.interference_power / self.noise_power
    
    # Stack into state matrix: (num_subbands, 3)
    state = np.column_stack([psd_normalized, cqi_normalized, interference_ratio])
    return state
```

### Reward Computation

```python
def _compute_reward(self, throughput: float, interference: float) -> float:
    # Throughput component
    throughput_reward = self.alpha * throughput
    
    # Interference penalty (normalized)
    interference_ratio = interference / self.noise_power
    interference_penalty = self.beta * interference_ratio
    
    # Combined reward
    reward = throughput_reward - interference_penalty
    return reward
```

### DDQN Update

```python
def train_step(self):
    # Sample minibatch
    batch = random.sample(self.replay_buffer, self.batch_size)
    
    # Current Q-values
    current_q = self.q_network.predict(states)
    
    # Next actions using online network
    next_q_online = self.q_network.predict(next_states)
    next_actions = np.argmax(next_q_online, axis=1)
    
    # Target Q-values using target network
    next_q_target = self.target_network.predict(next_states)
    
    # Compute targets
    targets = current_q.copy()
    for i in range(batch_size):
        if dones[i]:
            targets[i, actions[i]] = rewards[i]
        else:
            targets[i, actions[i]] = rewards[i] + gamma * next_q_target[i, next_actions[i]]
    
    # Train network
    self.q_network.fit(states, targets, epochs=1, verbose=0)
```

## 9. Concluding Note

This integrated DRL–PHY co-simulation model represents a **closed-loop intelligent spectrum management system** capable of balancing **communication efficiency and RFI protection**. It aligns with:

- **Sutton & Barto (2018)**: Policy learning foundations
- **Haykin (2008)**: Adaptive dynamic systems
- **Hong et al. (2021)**: Federated edge-based training extensions

The framework enables autonomous spectrum selection that adapts to time-varying interference conditions while maintaining 5G communication performance.
