# Getting Started with SWIFT II DRL-Assisted Spectrum Selection

## 🚀 Quick Start (5 Minutes)

### Step 1: Verify Installation

```bash
cd /workspace/Codes/Python
ls -l
# You should see: main.py, config.py, environment.py, ddqn_agent.py, metrics.py, matlab_interface.py
```

### Step 2: Install Dependencies

```bash
pip install tensorflow>=2.10.0 numpy matplotlib scipy pandas tqdm
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Run Your First Training

```bash
# Quick test (100 episodes, ~5 minutes)
python main.py --episodes 100 --no-plot

# Full training (1000 episodes, ~1 hour)
python main.py --episodes 1000
```

### Step 4: Check Results

```bash
ls ../../../Figures/
ls ../../../models/
```

## 📋 What Was Implemented

Based on your paper specifications, I've created a complete implementation with:

### ✅ Core Modules (All New)

1. **`config.py`** (430 lines)
   - Complete configuration management system
   - All paper parameters: spectrum, channel, DDQN, reward, simulation
   - JSON import/export
   - Easy customization

2. **`environment.py`** (490 lines)
   - Full MDP environment matching paper formulation
   - State: `[PSD(f1,t), ..., PSD(fn,t), CQI_t, I_t]`
   - Action: Channel selection
   - Reward: `r_t = α·log2(1 + SINR) - β·I_t`
   - Temporal depth support
   - MATLAB integration support

3. **`ddqn_agent.py`** (560 lines)
   - Double Deep Q-Network with paper-aligned architecture
   - CNN-based feature extraction
   - Experience replay buffer
   - Target network updates
   - ε-greedy exploration with decay
   - Training loop management

4. **`metrics.py`** (490 lines)
   - Performance tracking (throughput, interference, SINR)
   - Baseline comparison
   - Convergence detection
   - Visualization suite (6-panel plots)
   - JSON export

5. **`matlab_interface.py`** (410 lines)
   - MATLAB Engine integration
   - Simulated fallback for testing
   - Channel simulation wrapper
   - PSD calculation interface

6. **`main.py`** (370 lines)
   - Complete training pipeline
   - Command-line interface
   - Baseline comparison
   - Evaluation mode
   - Automatic checkpointing

### 📚 Documentation (All New)

1. **`SWIFT_II_OVERVIEW.md`** (9.5 KB)
   - Complete paper overview
   - System architecture
   - MDP formulation
   - Algorithm pseudocode
   - Expected results

2. **`README.md`** (8.0 KB)
   - Project overview
   - Installation guide
   - Usage examples
   - Performance metrics

3. **`Codes/readme.md`** (Complete rewrite)
   - Detailed code documentation
   - Module descriptions
   - API reference
   - Troubleshooting guide

4. **`IMPLEMENTATION_SUMMARY.md`** (13 KB)
   - Implementation checklist
   - Feature list
   - Quick reference

5. **`GETTING_STARTED.md`** (This file)
   - Quick start guide
   - Common use cases

## 🎯 Common Use Cases

### Use Case 1: Basic Training

```bash
python main.py --episodes 500
```

**What happens:**
- Trains DDQN agent for 500 episodes
- Compares with random baseline
- Saves model checkpoints every 100 episodes
- Generates training curves
- Prints performance summary

**Output:**
- Models: `./models/ddqn_ep*.h5`
- Figures: `../../../Figures/training_curves_*.png`
- Metrics: `../../../results/metrics_*.json`

### Use Case 2: Custom Configuration

```python
# Create custom_config.py
from config import ConfigManager

config = ConfigManager()
config.ddqn.learning_rate = 0.0005
config.reward.alpha_throughput = 1.5
config.reward.beta_interference = 0.3
config.simulation.num_episodes = 2000

# Save to JSON
import json
with open('custom_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)
```

Then run:
```bash
python main.py --config custom_config.json
```

### Use Case 3: Evaluation Only

```bash
# Load trained model and evaluate
python main.py --eval-only --model-path ./models/ddqn_ep1000_final_online.h5
```

**What happens:**
- Loads trained model
- Runs 50 evaluation episodes (no training)
- Generates action distribution plot
- Prints performance statistics

### Use Case 4: With MATLAB PHY Layer

```bash
# If you have MATLAB installed
python main.py --matlab --matlab-path ../Matlab --episodes 1000
```

**Requirements:**
- MATLAB R2020b+
- MATLAB Engine for Python
- 5G Toolbox

### Use Case 5: Quick Test

```bash
# Fast test run (no plots, few episodes)
python main.py --episodes 50 --no-plot
```

## 📊 Understanding the Output

### Training Progress

```
Episode 100/1000
  Reward: 52.34 (avg: 48.76)
  Epsilon: 0.6050
  Loss: 0.0234
  Q-value: 125.67
  Buffer: 6400/10000
```

- **Reward**: Current episode total reward
- **Epsilon**: Exploration rate (decays over time)
- **Loss**: DDQN training loss
- **Q-value**: Average Q-value (should increase)
- **Buffer**: Replay buffer size

### Final Summary

```
PERFORMANCE SUMMARY
==================
Training Statistics:
  Total Episodes: 1000
  Convergence Episode: 847
  Final Epsilon: 0.0100

Performance Metrics:
  Average Reward: 65.23 ± 4.12
  Average Throughput: 2.45 bits/s/Hz
  Average Interference: 1.82 mW
  Average SINR: 11.34 dB

Comparison with Baseline:
  Throughput Improvement: +22.5%
  Interference Reduction: -48.0%
```

## 🔧 Configuration Parameters

### Key Parameters to Tune

| Parameter | Default | Description | When to Modify |
|-----------|---------|-------------|----------------|
| `learning_rate` | 0.0001 | DDQN learning rate | If not converging |
| `epsilon_decay` | 0.995 | Exploration decay | If converging too fast/slow |
| `batch_size` | 64 | Training batch size | Based on memory |
| `alpha_throughput` | 1.0 | Throughput reward weight | To prioritize throughput |
| `beta_interference` | 0.5 | Interference penalty weight | To prioritize RFI mitigation |
| `num_episodes` | 1000 | Training episodes | Based on time budget |

### Quick Parameter Changes

**More exploration:**
```python
config.ddqn.epsilon_initial = 1.0
config.ddqn.epsilon_decay_rate = 0.998  # Slower decay
```

**Faster learning:**
```python
config.ddqn.learning_rate = 0.0005
config.ddqn.batch_size = 128
```

**Prioritize interference mitigation:**
```python
config.reward.beta_interference = 1.0  # Increase penalty
config.reward.alpha_throughput = 0.8   # Decrease reward
```

## 📈 Expected Performance

Based on the paper specifications:

| Metric | Expected Range |
|--------|----------------|
| Interference Mitigation | 40-50% |
| Throughput Degradation | < 5% |
| Convergence Episode | 500-1000 |
| Final Average Reward | 60-70 |
| SINR Improvement | +3-5 dB |

## 🐛 Troubleshooting

### Problem: Import errors

```bash
# Solution: Install dependencies
pip install tensorflow numpy matplotlib scipy
```

### Problem: Slow training

```bash
# Solution 1: Reduce episodes or steps
python main.py --episodes 100

# Solution 2: Increase batch size
# Modify config.py: config.ddqn.batch_size = 128
```

### Problem: Not converging

```bash
# Solution: Adjust learning rate and exploration
# Modify config.py:
config.ddqn.learning_rate = 0.0005  # Increase
config.ddqn.epsilon_decay_rate = 0.998  # Slower decay
```

### Problem: MATLAB not available

```bash
# Solution: Use simulated interface (automatic fallback)
# Just run without --matlab flag:
python main.py --episodes 1000
```

### Problem: Out of memory

```bash
# Solution: Reduce batch size and buffer
# Modify config.py:
config.ddqn.batch_size = 32
config.ddqn.replay_buffer_size = 5000
```

## 📁 File Locations

```
/workspace/
├── Codes/Python/          # Your working directory
│   ├── main.py           # Run this!
│   ├── config.py         # Modify parameters here
│   └── ...
├── models/               # Saved models appear here
├── Figures/              # Plots appear here
├── results/              # Metrics JSON appear here
└── logs/                 # TensorBoard logs (if enabled)
```

## 🎓 Learning Path

### Beginner
1. Run basic training: `python main.py --episodes 100`
2. Check output in `Figures/`
3. Read `README.md` and `SWIFT_II_OVERVIEW.md`

### Intermediate
1. Modify `config.py` parameters
2. Run custom training
3. Compare different configurations
4. Analyze `metrics.json`

### Advanced
1. Modify reward function in `environment.py`
2. Customize network in `ddqn_agent.py`
3. Integrate with external systems
4. Use MATLAB PHY layer

## 📞 Support

**Documentation:**
- `README.md` - Project overview
- `SWIFT_II_OVERVIEW.md` - Technical details
- `Codes/readme.md` - Code reference
- `IMPLEMENTATION_SUMMARY.md` - Implementation status

**Testing:**
Each module has standalone tests:
```bash
python config.py        # Test configuration
python environment.py   # Test environment
python ddqn_agent.py   # Test DDQN
python metrics.py      # Test metrics
```

## ✅ Checklist Before Starting

- [ ] Python 3.8+ installed
- [ ] TensorFlow 2.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Working directory: `/workspace/Codes/Python/`
- [ ] Reviewed configuration in `config.py`
- [ ] Understand expected runtime (~1 hour for 1000 episodes)

## 🚀 Ready to Go!

You're all set! Start with:

```bash
cd /workspace/Codes/Python
python main.py --episodes 500
```

Good luck with your SWIFT II research! 🎯
