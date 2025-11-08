# SWIFT II - Deep Reinforcement Learning-Assisted Spectrum Selection

## Overview

This project implements a **Deep Reinforcement Learning (DRL)–assisted dynamic spectrum selection framework** that enables **coexistence between 5G active systems and passive RF sensors** within a shared spectrum environment, based on the **SWIFT-II NSF project** paper: *"Deep Reinforcement Learning-Assisted Spectrum Selection for RFI-Resilient Passive Sensing in Shared 5G Environments"*.

## System Architecture

The system consists of three main interacting components:

### 1. 5G Transceiver (gNB/UE Pair)
- Generates and transmits 5G NR-compliant signals within a shared band
- Implemented in MATLAB using 5G Toolbox
- Uses OFDM modulation with configurable resource blocks

### 2. Passive Sensor Node
- Continuously senses the spectral environment
- Provides **Power Spectral Density (PSD)** feedback
- Measures interference levels and spectrum occupancy

### 3. DRL-based Spectrum Agent
- Acts as the decision-making controller
- Dynamically allocates or avoids frequency channels to protect passive sensors
- Implements Double Deep Q-Network (DDQN) for robust policy optimization

## Markov Decision Process (MDP) Formulation

### State Space (sₜ)
Combines spectral occupancy features, interference level estimates, and channel quality indicators:
```
sₜ = [PSD(f₁,t), ..., PSD(fₙ,t), CQI_t, Iₜ]
```
where:
- `PSD(fᵢ,t)`: Power Spectral Density in sub-band i at time t
- `CQI_t`: Channel Quality Indicator from SINR estimates
- `Iₜ`: Interference ratio (P_int/P_th)

### Action Space (aₜ)
Selection of sub-bands or resource block (RB) groups for 5G transmission:
```
aₜ ∈ {f₁, f₂, ..., fₙ}
```

### Reward Function (rₜ)
Multi-objective reward balancing 5G throughput and interference mitigation:
```
rₜ = α · log₂(1 + γₜ) - β · Iₜ
```
where:
- `α`: Weight for throughput reward
- `β`: Weight for interference penalty
- `γₜ`: SINR at time t
- `Iₜ`: Interference power at passive sensor

### Objective Function
The RL agent maximizes expected cumulative reward:
```
max_π E[∑ᵗ₌₀ᵀ γᵗ(R₅G(aₜ,sₜ) - λ·P_int(aₜ,sₜ))]
```
subject to spectrum coexistence and latency constraints.

## AI and Learning Framework

### Double Deep Q-Network (DDQN)

Addresses Q-value overestimation and convergence instability found in standard DQN.

**Architecture:**
- **Online Network**: Q_θ(sₜ,aₜ) - Updated every step
- **Target Network**: Q_θ⁻(sₜ,aₜ) - Updated every C steps

**Update Rule:**
```
L(θ) = E[(rₜ + γ·Q_θ⁻(sₜ₊₁, argmax_a' Q_θ(sₜ₊₁,a')) - Q_θ(sₜ,aₜ))²]
```

**Key Features:**
- CNN-based feature extraction for spectral state representation
- Experience replay buffer for training stability
- ε-greedy exploration with exponential decay
- Target network updates for convergence stability

## Algorithmic Workflow

### 1. Environment Definition (Hybrid PHY + RL Layer)

**Input Signals and Parameters:**
- `F = {f₁, f₂, ..., fₙ}`: Set of available sub-bands/RB groups
- `P_tx`: Transmit power of 5G node
- `H(fᵢ,t)`: Channel response per sub-band (from MATLAB channel model)
- `PSD(fᵢ,t)`: Measured power spectral density from passive sensor
- `σ²`: Thermal noise level
- `τ`: Sensing feedback latency

**Outputs (to RL agent):**
- Spectrum occupancy map: `Sₜ = [PSD(f₁,t), ..., PSD(fₙ,t)]`
- Channel quality indicators (CQI) derived from SINR estimates
- Interference ratio: `Iₜ = P_int/P_th`

### 2. State, Action, and Reward Design

| Element | Description | Mathematical Representation |
|---------|-------------|----------------------------|
| **State** sₜ | Concatenation of spectral features and interference feedback | `sₜ = [PSD_t, CQI_t, Iₜ]` |
| **Action** aₜ | Select one or multiple sub-bands for 5G transmission | `aₜ ∈ F` |
| **Reward** rₜ | Balances 5G throughput and interference mitigation | `rₜ = α·log₂(1 + γₜ) - β·Iₜ` |
| **Transition** | Environment updates sₜ₊₁ via spectrum dynamics | `sₜ₊₁ = f(sₜ, aₜ, Hₜ)` |

### 3. Training Algorithm Pseudocode

```python
Initialize environment E (MATLAB channel + PSD sensing model)
Initialize replay buffer D with capacity M
Initialize DDQN networks Q_θ and Q_θ⁻ with random weights
Set ε = ε₀, learning rate η, discount γ, and target update rate C

for episode = 1 to N_episodes:
    Reset environment → s₀
    for t = 1 to T:
        # Step 1: Select action
        if random() < ε:
            aₜ = random_action()
        else:
            aₜ = argmax_a Q_θ(sₜ, a)
        
        # Step 2: Apply action in MATLAB PHY model
        throughput, interference, PSD_next = simulate_PHY(aₜ)
        
        # Step 3: Compute reward
        rₜ = α · log₂(1 + throughput) - β · interference
        
        # Step 4: Observe new state
        sₜ₊₁ = [PSD_next, CQI_next, interference]
        
        # Step 5: Store experience
        D.append((sₜ, aₜ, rₜ, sₜ₊₁))
        
        # Step 6: Sample minibatch and update Q-network
        if len(D) > batch_size:
            batch = random_sample(D)
            Perform DDQN update using loss L(θ)
        
        # Step 7: Update target network
        if t % C == 0:
            θ⁻ ← θ
        
        sₜ ← sₜ₊₁
    
    Decay ε after each episode
```

## Simulation Environment

### MATLAB Layer (Physical Layer)
Implements realistic 5G NR physical layer:
- OFDM waveform generator (`nrWaveformGenerator`)
- CDL channel models (`nrCDLChannel`, CDL-C/D)
- Spectrum analyzer to extract `PSD(fᵢ)` and interference maps
- Returns key KPIs to Python RL layer via interface:
  ```matlab
  [throughput, PSD_next, CQI_next, interference] = PHY_simulate(action, params);
  ```

### Python Layer (RL Training)
Handles DDQN learning using TensorFlow 2.x:
- Experience replay memory management
- Neural network training and optimization
- Model checkpointing and storage
- TensorBoard for convergence monitoring

## Performance Metrics

The system evaluates the following key performance indicators:

### 1. Average Throughput Improvement
```
ΔR = (R_DRL - R_baseline) / R_baseline × 100%
```

### 2. Interference Power Reduction
```
ΔP_int = (P_baseline - P_DRL) / P_baseline × 100%
```

### 3. Learning Stability
Average Q-value variance across episodes:
```
σ²_Q = Var(Q(sₜ,aₜ))
```

### 4. Convergence Rate
Episode index at which 90% stable reward is achieved.

## Expected Results

Based on the paper, the DRL agent is expected to:
- Learn optimal avoidance behavior without explicit coordination
- Achieve **40–50% interference mitigation** with minimal throughput degradation
- Outperform heuristic or fixed-threshold selection schemes in both convergence rate and sensing protection
- Demonstrate robust performance under variable interference levels

## System Block Diagram

```
┌───────────────────────────────┐
│     Spectrum Environment      │
│  (5G TX + Passive Sensor)     │
└──────────────┬────────────────┘
               │ PSD + CQI feedback
┌──────────────▼───────────────┐
│    DRL Agent (DDQN)          │
│  State → Action (sub-band)   │
│  Reward ← Interference & R5G │
└──────────────┬───────────────┘
               │ Action
┌──────────────▼───────────────┐
│ MATLAB PHY Simulation Layer  │
│  OFDM + Channel + RFI model  │
└──────────────┬───────────────┘
               │ Updated PSD
        ←──────┘
```

## Implementation Structure

```
/workspace/
├── Codes/
│   ├── Python/
│   │   ├── DDQN.py              # DDQN agent implementation
│   │   ├── Channel.py           # 5G channel modeling (Sionna)
│   │   ├── environment.py       # MDP environment wrapper
│   │   ├── config.py            # Simulation parameters
│   │   └── metrics.py           # Performance evaluation
│   ├── Matlab/
│   │   ├── ActivePassiveAI_Sensing.m  # Main simulation script
│   │   ├── my_NRChannel_modified.m    # 5G NR channel model
│   │   ├── Passive_Signal.m           # Passive sensor signal generation
│   │   └── SIC*.m                     # Interference cancellation
│   └── readme.md
├── Figures/                     # Visualization results
├── README.md                    # Project overview
└── SWIFT_II_OVERVIEW.md        # This file
```

## References

This implementation is based on theoretical foundations from:
- **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction
- **Haykin (2008)**: Cognitive Radio: Brain-Empowered Wireless Communications
- **Hong et al. (2021)**: Federated edge-based training extensions
- **SWIFT-II NSF Project**: Spectrum coexistence research

## Notes

This integrated DRL–PHY co-simulation model represents a **closed-loop intelligent spectrum management system** capable of balancing **communication efficiency and RFI protection** in shared spectrum environments.
