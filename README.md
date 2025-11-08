# SWIFT-II Spectrum Coexistence Framework

This repository contains the MATLAB and Python co-simulation environment developed for the **SWIFT-II** project, enabling deep reinforcement learning (DRL) assisted spectrum sharing between 5G New Radio (NR) systems and passive RF sensors.

## Overview of Proposed Structure and Model

The framework implements a **Deep Reinforcement Learning–assisted dynamic spectrum selection** pipeline that supports coexistence between 5G active systems and passive RF sensors within a shared spectrum band. The solution targets **AI-driven spectrum management** for **remote sensing (RS) protection** and **interference-resilient 5G communications**.

### System Architecture

Three primary components interact in the environment:

1. **5G Transceiver (gNB/UE Pair):** Generates and transmits 5G NR-compliant waveforms within the shared band.
2. **Passive Sensor Node:** Continuously senses the spectral environment and reports **power spectral density (PSD)** measurements.
3. **DRL-Based Spectrum Agent:** Serves as the decision-making controller that dynamically allocates or avoids frequency channels to protect passive sensors.

The environment is modeled as a **Markov Decision Process (MDP)** with:

- **State (`s_t`):** Spectral occupancy features, interference-level estimates, and channel quality indicators.
- **Action (`a_t`):** Selection of sub-bands or resource block (RB) groups for 5G transmission.
- **Reward (`r_t`):** Balances 5G throughput, interference reduction at the passive sensor, and sensing accuracy preservation.

### AI and Learning Framework

A **Double Deep Q-Network (DDQN)** is employed to mitigate Q-value overestimation and stabilize convergence. The agent observes time-varying spectrum states via CNN-based feature extraction, learns optimal access policies through experience replay, and leverages target network updates for robustness. The objective is:

\\[
\max_{\pi} \mathbb{E} \left[\sum_{t=0}^{T} \gamma^t \left(R_{5G}(a_t,s_t) - \lambda P_{\text{int}}(a_t,s_t)\right) \right]
\\]

subject to spectrum coexistence and latency constraints, where `R_{5G}` denotes throughput gain and `P_{int}` represents interference at the passive sensor.

### Simulation and Evaluation

The co-simulation couples **MATLAB** (5G physical-layer modeling) with **Python/TensorFlow** (DRL training). Shared-spectrum dynamics are emulated using multi-band PSD data, SINR mapping, and CDL/TDL fading models. Results demonstrate:

- Adaptive avoidance behavior without explicit coordination;
- Up to **40–50% interference mitigation** with minimal throughput degradation;
- Faster convergence and better sensing protection than heuristic baselines.

## Algorithmic Workflow: DRL-Assisted Spectrum Selection

### 1. Environment Definition (Hybrid PHY + RL Layer)

Each time step represents a transmission interval or sensing cycle within the shared spectrum.

- `𝔽 = {f₁, f₂, …, f_N}` — available sub-bands / RB groups  
- `P_tx` — 5G transmit power  
- `H(f_i, t)` — sub-band channel response (from MATLAB channel model)  
- `PSD(f_i, t)` — passive sensor power spectral density measurements  
- `σ²` — thermal noise level  
- `τ` — sensing feedback latency  

**RL Observations**

- Spectrum occupancy map `S_t = [PSD(f₁,t), …, PSD(f_N,t)]`  
- CQI derived from SINR estimates  
- Interference ratio `I_t = P_int / P_th`

### 2. State, Action, and Reward Design

| Element            | Description                                                                 | Expression                                      |
| ------------------ | --------------------------------------------------------------------------- | ----------------------------------------------- |
| **State** `s_t`    | Concatenated spectral and interference features                             | `s_t = [PSD_t, CQI_t, I_t]`                      |
| **Action** `a_t`   | Selects sub-band(s) for 5G transmission                                     | `a_t ∈ 𝔽`                                       |
| **Reward** `r_t`   | Balances throughput against interference                                    | `r_t = α log₂(1 + γ_t) - β I_t`                 |
| **Transition**     | Environment evolves according to spectrum and channel dynamics              | `s_{t+1} = f(s_t, a_t, H_t)`                     |

### 3. Deep RL Agent: Double Deep Q-Network

Two networks are trained:

- **Online:** `Q_θ(s_t, a_t)`  
- **Target:** `Q_{θ⁻}(s_t, a_t)` updated every `C` steps

Loss function:

\\[
L(\theta) = \mathbb{E} \left[ \left( r_t + \gamma Q_{\theta^-}\left(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1},a')\right) - Q_{\theta}(s_t,a_t) \right)^2 \right]
\\]

Gradient descent update: `θ ← θ - η ∇_θ L(θ)`

Exploration: ε-greedy with exponential decay `ε_t = max(ε_min, ε_0 e^{-k t})` (optional Boltzmann exploration).

### 4. Algorithm Pseudocode

```python
Initialize environment E (MATLAB channel + PSD sensing)
Initialize replay buffer D with capacity M
Initialize DDQN networks Qθ, Qθ− with random weights
Set ε = ε0, learning rate η, discount γ, target update C

for episode in range(N_episodes):
    s = E.reset()
    for t in range(T):
        if random() < ε:
            a = random_action()
        else:
            a = argmax_a Qθ(s, a)

        throughput, interference, PSD_next = simulate_PHY(a)
        r = α * log2(1 + throughput) - β * interference
        s_next = concat(PSD_next, CQI_next, interference)
        D.append((s, a, r, s_next))

        if len(D) > batch_size:
            batch = random_sample(D)
            update_Q_networks(batch)

        if t % C == 0:
            θ− ← θ

        s = s_next

    ε = decay(ε)
```

### 5. Simulation Loop and MATLAB Interface

- **MATLAB Layer:** Implements the 5G NR PHY (OFDM waveform generation, CDL channel models, spectrum analysis) and returns key KPIs:

  ```matlab
  [throughput, PSD_next, CQI_next, interference] = PHY_simulate(action, params);
  ```

- **Python Layer:** Handles DDQN learning, replay memory, model persistence, and TensorBoard logging.

### 6. Output Metrics and Analysis

- Throughput gain `ΔR = ((R_DRL - R_baseline) / R_baseline) × 100%`
- Interference reduction `ΔP_int = ((P_baseline - P_DRL) / P_baseline) × 100%`
- Learning stability via average Q-value variance `σ_Q²`
- Convergence rate: episode index achieving 90% of peak reward

### 7. Conceptual Block Diagram

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

### 8. References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2018  
- Haykin, *Cognitive Dynamic Systems: Perception-Action Cycle*, 2008  
- Hong et al., *Federated Edge Learning for 6G*, 2021
