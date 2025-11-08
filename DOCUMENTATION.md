# Algorithmic Workflow: DRL-Assisted Spectrum Selection for RFI-Resilient 5G–Passive Sensing Coexistence

## 1. Environment Definition (Hybrid PHY + RL Layer)

The environment emulates the shared spectrum between the **5G NR transceiver** and the **passive sensor**. Each time step represents one transmission interval or sensing cycle.

### Input Signals and Parameters

* $\mathcal{F} = \{f_1, f_2, \dots, f_N\}$: set of available sub-bands / RB groups
* $P_{tx}$: transmit power of the 5G node
* $H(f_i,t)$: channel response per sub-band (from MATLAB channel model)
* $PSD(f_i,t)$: measured power spectral density from the passive sensor
* $\sigma^2$: thermal noise level
* $\tau$: sensing feedback latency

### Outputs (to RL agent)

* Spectrum occupancy map: $S_t = [PSD(f_1,t), ..., PSD(f_N,t)]$
* Channel quality indicators (CQI) derived from SINR estimates
* Interference ratio: $I_t = P_{int}/P_{th}$

---

## 2. State, Action, and Reward Design

| Element            | Description                                                                 | Mathematical Representation                       |
| ------------------ | --------------------------------------------------------------------------- | ------------------------------------------------- |
| **State** ($s_t$)  | Concatenation of spectral features and interference feedback                | $s_t = [PSD_t, CQI_t, I_t]$                       |
| **Action** ($a_t$) | Select one or multiple sub-bands for 5G transmission                        | $a_t \in \mathcal{F}$                             |
| **Reward** ($r_t$) | Balances 5G throughput and interference mitigation                          | $r_t = \alpha \log_2(1 + \gamma_t) - \beta I_t$   |
| **Transition**     | Environment updates ($s_{t+1}$) via spectrum dynamics and channel evolution | $s_{t+1} = f(s_t, a_t, H_t)$                      |

---

## 3. Deep RL Agent: Double Deep Q-Network (DDQN)

Two networks are maintained:

* **Online Network:** $Q_\theta(s_t,a_t)$
* **Target Network:** $Q_{\theta^-}(s_t,a_t)$ (updated every $C$ steps)

### Update Rule

$$L(\theta) = \mathbb{E}\left[ \left( r_t + \gamma Q_{\theta^-}\left(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1},a')\right) - Q_\theta(s_t,a_t) \right)^2 \right]$$

with gradient descent updates:
$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

### Exploration Strategy

* ε-greedy with decay: $\epsilon_t = \max(\epsilon_{min}, \epsilon_0 e^{-kt})$
* Optional **Boltzmann policy** for continuous exploration.

---

## 4. Algorithm Pseudocode

```python
Initialize environment E (MATLAB channel + PSD sensing model)
Initialize replay buffer D with capacity M
Initialize DDQN networks Qθ and Qθ− with random weights
Set ε = ε0, learning rate η, discount γ, and target update rate C

for episode = 1 : N_episodes:
    Reset environment → s0
    for t = 1 : T:
        # Step 1: Select action
        if random() < ε:
            a_t = random_action()
        else:
            a_t = argmax_a Qθ(s_t, a)
        
        # Step 2: Apply action in MATLAB PHY model
        throughput, interference, PSD_next = simulate_PHY(a_t)
        
        # Step 3: Compute reward
        r_t = α * log2(1 + throughput) - β * interference
        
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
            θ− ← θ
        
        s_t ← s_{t+1}
    
    Decay ε after each episode
```

---

## 5. Simulation Loop and MATLAB Interface

### MATLAB Layer

Implements realistic 5G NR physical layer:

* OFDM waveform generator (`nrWaveformGenerator`)
* CDL channel models (`nrCDLChannel`, CDL-C, etc.)
* Spectrum analyzer to extract `PSD(f_i)` and interference maps
* Returns key KPIs to Python RL layer via `matlab.engine` interface:

```matlab
[throughput, PSD_next, CQI_next, interference] = PHY_simulate(action, params);
```

### Python Layer (TensorFlow 2.x)

* Handles DDQN learning, replay memory, model storage
* TensorBoard for convergence monitoring

---

## 6. Output Metrics and Analysis

After convergence, the system evaluates:

* **Average throughput improvement:**
  $$\Delta R = \frac{R_{DRL} - R_{baseline}}{R_{baseline}} \times 100\%$$
* **Interference power reduction:**
  $$\Delta P_{int} = \frac{P_{baseline} - P_{DRL}}{P_{baseline}} \times 100\%$$
* **Learning stability:** average Q-value variance $\sigma^2_Q$ across episodes
* **Convergence rate:** episode index at 90% stable reward

---

## 7. Block Diagram Flow

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

---

## 8. Implementation Details

### State Representation

The state matrix combines:
- **Indication Matrix:** Contains communication action, SINR feedback, sensing actions, and PSD measurements
- **Temporal Memory:** Maintains history of $T$ previous time steps
- **Channel Features:** Interference power, jammer power, channel gains

### Reward Function

Multi-objective reward balancing:
- **Communication reward:** Based on SINR exceeding threshold $\mu_{th}$
- **Interference penalty:** Penalizes high interference power at selected channel
- **Jammer penalty:** Additional penalty for jammer presence

$$r_t = \frac{\lambda \max(0, SINR - \mu_{th}) - 0.1 \cdot P_{int} - 0.1 \cdot P_{jammer}}{\lambda}$$

### Network Architecture

- **Input:** State matrix of shape `(Ns+1, 4, T)` where:
  - `Ns+1`: Number of sensing channels + 1 communication channel
  - `4`: Features per channel (action, reward, interference, jammer)
  - `T`: Temporal depth
- **Convolutional Layers:** 2D convolution for spatial-temporal feature extraction
- **Output:** Q-values for each of $N$ available channels

---

## 9. Concluding Note

This integrated DRL–PHY co-simulation model represents a **closed-loop intelligent spectrum management system** capable of balancing **communication efficiency and RFI protection**. It aligns with the theoretical foundations of **Sutton & Barto (2018)** for policy learning, **Haykin (2008)** for adaptive dynamic systems, and **Hong et al. (2021)** for federated edge-based training extensions.
