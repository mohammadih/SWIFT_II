# SWIFT_II

SWIFT_II_Codes and Simulation

## Overview of Proposed Structure and Model

The paper proposes a Deep Reinforcement Learning (DRL)-assisted dynamic spectrum selection framework that enables coexistence between 5G active systems and passive RF sensors within a shared spectrum environment. The model is designed within the context of the SWIFT-II NSF project, focusing on AI-driven spectrum management for Remote Sensing (RS) protection and interference-resilient 5G communications.

### System Architecture

The system consists of three main interacting components:

1. 5G transceiver (gNB/UE pair): generates and transmits 5G NR-compliant signals within a shared band.
2. Passive sensor node: continuously senses the spectral environment and provides Power Spectral Density (PSD) feedback.
3. DRL-based spectrum agent: acts as the decision-making controller that dynamically allocates or avoids frequency channels to protect passive sensors.

The environment is modeled as a Markov Decision Process (MDP) where:

- **State** \(s_t\): combines spectral occupancy features, interference level estimates, and channel quality indicators.
- **Action** \(a_t\): selection of sub-bands or RB groups for 5G transmission.
- **Reward** \(r_t\): jointly considers 5G throughput, interference power reduction at the passive sensor, and sensing accuracy preservation.

### AI and Learning Framework

A Double Deep Q-Network (DDQN) is implemented for robust policy optimization, addressing Q-value overestimation and convergence instability found in standard DQN. The agent observes time-varying spectrum states through CNN-based feature extraction, learns optimal spectrum access policies, and adapts to RFI variations using experience replay and target network updates.

The RL agent objective is formalized as:

\[
\max_{\pi} \mathbb{E} \left[\sum_{t=0}^{T} \gamma^t \left(R_{5G}(a_t, s_t) - \lambda P_{\text{int}}(a_t, s_t)\right) \right]
\]

subject to spectrum coexistence and latency constraints, where \(R_{5G}\) denotes throughput gain and \(P_{\text{int}}\) represents interference at the sensor.

### Simulation and Evaluation

The simulation environment integrates MATLAB for 5G physical-layer modeling and Python/TensorFlow for DRL training. The model emulates shared-spectrum dynamics through multi-band PSD data, SINR mapping, and channel fading models (CDL/TDL). Performance is evaluated under variable interference levels and training episodes, showing that the DRL agent:

- learns optimal avoidance behavior without explicit coordination,
- achieves up to 40-50% interference mitigation with minimal throughput degradation,
- outperforms heuristic or fixed-threshold selection schemes in both convergence rate and sensing protection.

## Algorithmic Workflow: DRL-Assisted Spectrum Selection for RFI-Resilient 5G-Passive Sensing Coexistence

### 1. Environment Definition (Hybrid PHY + RL Layer)

The environment emulates the shared spectrum between the 5G NR transceiver and the passive sensor. Each time step represents one transmission interval or sensing cycle.

**Input signals and parameters**

- \(\mathcal{F} = \{f_1, f_2, \dots, f_N\}\): set of available sub-bands or RB groups.
- \(P_{tx}\): transmit power of the 5G node.
- \(H(f_i, t)\): channel response per sub-band (from MATLAB channel model).
- \(\text{PSD}(f_i, t)\): measured power spectral density from the passive sensor.
- \(\sigma^2\): thermal noise level.
- \(\tau\): sensing feedback latency.

**Outputs to the RL agent**

- Spectrum occupancy map \(S_t = [\text{PSD}(f_1, t), \ldots, \text{PSD}(f_N, t)]\).
- Channel quality indicators derived from SINR estimates.
- Interference ratio \(I_t = P_{\text{int}} / P_{th}\).

### 2. State, Action, and Reward Design

| Element | Description | Mathematical representation |
| --- | --- | --- |
| State \(s_t\) | Concatenation of spectral features and interference feedback | \(s_t = [\text{PSD}_t, \text{CQI}_t, I_t]\) |
| Action \(a_t\) | Select one or multiple sub-bands for 5G transmission | \(a_t \in \mathcal{F}\) |
| Reward \(r_t\) | Balances 5G throughput and interference mitigation | \(r_t = \alpha \log_2(1 + \gamma_t) - \beta I_t\) |
| Transition | Environment updates \(s_{t+1}\) via spectrum dynamics and channel evolution | \(s_{t+1} = f(s_t, a_t, H_t)\) |

### 3. Deep RL Agent: Double Deep Q-Network (DDQN)

Two networks are maintained:

- Online network \(Q_{\theta}(s_t, a_t)\).
- Target network \(Q_{\theta^{-}}(s_t, a_t)\) (updated every \(C\) steps).

Update rule:

\[
L(\theta) = \mathbb{E}\left[ \left( r_t + \gamma Q_{\theta^{-}}\left(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a')\right) - Q_{\theta}(s_t, a_t) \right)^2 \right]
\]

with gradient descent updates \(\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta)\).

**Exploration strategy**

- \(\epsilon\)-greedy with decay: \(\epsilon_t = \max(\epsilon_{\text{min}}, \epsilon_0 e^{-k t})\).
- Optional Boltzmann policy for continuous exploration.

### 4. Algorithm Pseudocode

```python
Initialize environment E (MATLAB channel + PSD sensing model)
Initialize replay buffer D with capacity M
Initialize DDQN networks Q_theta and Q_theta_target with random weights
Set epsilon = epsilon_0, learning rate eta, discount gamma, and target update rate C

for episode in range(N_episodes):
    s_t = reset_environment()
    for t in range(T):
        # Step 1: Select action
        if random() < epsilon:
            a_t = random_action()
        else:
            a_t = argmax_a Q_theta(s_t, a)
        
        # Step 2: Apply action in MATLAB PHY model
        throughput, interference, psd_next, cqi_next = simulate_PHY(a_t)
        
        # Step 3: Compute reward
        r_t = alpha * log2(1 + throughput) - beta * interference
        
        # Step 4: Observe new state
        s_next = build_state(psd_next, cqi_next, interference)
        
        # Step 5: Store experience
        D.append((s_t, a_t, r_t, s_next))
        
        # Step 6: Sample minibatch and update Q-network
        if len(D) > batch_size:
            batch = random_sample(D)
            perform_DDQN_update(batch)
        
        # Step 7: Update target network
        if t % C == 0:
            Q_theta_target.load_state_dict(Q_theta.state_dict())
        
        s_t = s_next
    
    epsilon = decay_epsilon(epsilon)
```

### 5. Simulation Loop and MATLAB Interface

**MATLAB layer**

Implements a realistic 5G NR physical layer:

- OFDM waveform generator (`nrWaveformGenerator`).
- CDL channel models (`nrCDLChannel`, CDL-C, etc.).
- Spectrum analyzer to extract `PSD(f_i)` and interference maps.
- Returns key KPIs to the Python RL layer via ZeroMQ or `matlab.engine` interface:

```matlab
[throughput, psd_next, cqi_next, interference] = PHY_simulate(action, params);
```

**Python layer (TensorFlow 2.x)**

- Handles DDQN learning, replay memory, and model storage.
- TensorBoard for convergence monitoring.

### 6. Output Metrics and Analysis

After convergence, the system evaluates:

- Average throughput improvement: \(\Delta R = \frac{R_{\text{DRL}} - R_{\text{baseline}}}{R_{\text{baseline}}} \times 100\%\).
- Interference power reduction: \(\Delta P_{\text{int}} = \frac{P_{\text{baseline}} - P_{\text{DRL}}}{P_{\text{baseline}}} \times 100\%\).
- Learning stability: average Q-value variance \(\sigma_Q^2\) across episodes.
- Convergence rate: episode index at 90% stable reward.

### 7. Block Diagram Flow (conceptual for figure)

```
+-------------------------------+
|     Spectrum Environment      |
|  (5G TX + Passive Sensor)     |
+---------------+---------------+
                | PSD + CQI feedback
+---------------v---------------+
|    DRL Agent (DDQN)           |
|  State -> Action (sub-band)   |
|  Reward <- Interference + R5G |
+---------------+---------------+
                | Action
+---------------v---------------+
| MATLAB PHY Simulation Layer   |
|  OFDM + Channel + RFI model   |
+---------------+---------------+
                | Updated PSD
        <-------+
```

### 8. Concluding Note

This integrated DRL-PHY co-simulation model represents a closed-loop intelligent spectrum management system capable of balancing communication efficiency and RFI protection. It aligns with the theoretical foundations of Sutton and Barto (2018) for policy learning, Haykin (2008) for adaptive dynamic systems, and Hong et al. (2021) for federated edge-based training extensions.
