# SWIFT II – Deep Reinforcement Learning-Assisted Spectrum Selection for RFI-Resilient Passive Sensing in Shared 5G Environments

## Overview

This repository implements a **Deep Reinforcement Learning (DRL)–assisted dynamic spectrum selection framework** that enables **coexistence between 5G active systems and passive RF sensors** within a shared spectrum environment. The model is designed within the context of the **SWIFT-II NSF project**, focusing on **AI-driven spectrum management** for **Remote Sensing (RS) protection** and **interference-resilient 5G communications**.

## System Architecture

The system consists of three main interacting components:

1. **5G Transceiver (gNB/UE Pair):** Generates and transmits 5G NR-compliant signals within a shared band.
2. **Passive Sensor Node:** Continuously senses the spectral environment and provides **Power Spectral Density (PSD)** feedback.
3. **DRL-based Spectrum Agent:** Acts as the decision-making controller that dynamically allocates or avoids frequency channels to protect passive sensors.

The environment is modeled as a **Markov Decision Process (MDP)** where:

* **State (sₜ):** Combines spectral occupancy features, interference level estimates, and channel quality indicators.
* **Action (aₜ):** Selection of sub-bands or RB groups for 5G transmission.
* **Reward (rₜ):** Jointly considers 5G throughput, interference power reduction at the passive sensor, and sensing accuracy preservation.

## AI and Learning Framework

A **Double Deep Q-Network (DDQN)** is implemented for robust policy optimization, addressing Q-value overestimation and convergence instability found in standard DQN. The agent observes time-varying spectrum states through CNN-based feature extraction, learns optimal spectrum access policies, and adapts to RFI variations using experience replay and target network updates.

The RL agent's objective is formalized as:
```
max_π E[∑(t=0 to T) γ^t (R_5G(a_t,s_t) - λ P_int(a_t,s_t))]
```
subject to spectrum coexistence and latency constraints, where `R_5G` denotes throughput gain and `P_int` represents interference at the sensor.

## Simulation and Evaluation

The simulation environment integrates **MATLAB for 5G physical-layer modeling** and **Python/TensorFlow for DRL training**. The model emulates shared-spectrum dynamics through **multi-band PSD data**, **SINR mapping**, and **channel fading models (CDL/TDL)**. Performance is evaluated under variable interference levels and training episodes, showing that the DRL agent:

* Learns optimal avoidance behavior without explicit coordination;
* Achieves up to **40–50% interference mitigation** with minimal throughput degradation;
* Outperforms heuristic or fixed-threshold selection schemes in both convergence rate and sensing protection.

## Repository Structure

```
/workspace/
├── Codes/
│   ├── Matlab/          # 5G NR physical layer simulation
│   │   ├── ActivePassiveAI_Sensing.m    # Main simulation script
│   │   ├── my_NRChannel_modified.m      # CDL channel modeling
│   │   ├── my_pspectrum.m               # PSD computation
│   │   ├── Passive_Signal.m             # Passive sensor signal generation
│   │   └── SIC1_DL_Reg.m / SIC2_DL_Reg.m  # Successive Interference Cancellation
│   └── Python/          # DRL agent implementation
│       ├── DDQN.py      # Double Deep Q-Network implementation
│       └── Channel.py   # Channel modeling utilities (Sionna-based)
├── Figures/             # Simulation results and visualizations
└── README.md           # This file
```

## Quick Start

### Prerequisites

- **MATLAB** (R2020b or later) with 5G Toolbox
- **Python** 3.8+ with TensorFlow 2.x
- Required Python packages: `tensorflow`, `numpy`, `matplotlib`, `sionna`

### Running the Simulation

1. **MATLAB Layer (5G PHY Simulation):**
   ```matlab
   cd Codes/Matlab
   ActivePassiveAI_Sensing
   ```

2. **Python Layer (DRL Training):**
   ```python
   cd Codes/Python
   python DDQN.py
   ```

## Key Features

- **Hybrid Simulation:** MATLAB handles realistic 5G NR physical layer (OFDM, CDL channels), while Python manages DRL learning
- **DDQN Implementation:** Double Deep Q-Network with experience replay and target network updates
- **Dynamic Spectrum Selection:** Agent learns to avoid interference-prone channels while maintaining 5G throughput
- **PSD-based Feedback:** Passive sensor provides real-time spectral occupancy information

## Documentation

For detailed algorithmic workflow and implementation details, see:
- [Algorithmic Workflow Documentation](DOCUMENTATION.md)
- [Code Documentation](Codes/readme.md)

## References

- Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- Haykin (2008) - Adaptive Dynamic Systems
- Hong et al. (2021) - Federated edge-based training extensions

## License

[Add your license information here]

## Contact

[Add contact information here]
