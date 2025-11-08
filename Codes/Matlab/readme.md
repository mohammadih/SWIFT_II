# MATLAB Code Documentation

## Overview

This directory contains MATLAB implementations for 5G NR physical layer simulation, channel modeling, and passive sensor signal processing within the SWIFT II framework.

## Files Description

### `ActivePassiveAI_Sensing.m`

Main simulation script that demonstrates active-passive coexistence:

**Key Features:**
- Loads 5G NR signal data (IQ samples) from binary files
- Applies 5G NR channel model (`my_NRChannel`)
- Generates passive sensor signal
- Computes power spectral density (PSD)
- Applies Successive Interference Cancellation (SIC) using deep learning
- Visualizes received and filtered signals

**Usage:**
```matlab
ActivePassiveAI_Sensing
```

**Parameters:**
- `mu_passive = 0`: Mean of passive signal
- `sigma_passive = -90`: Noise level (dB)
- `p_n = -120`: Passive signal power (dB)
- `fs = 30.72e6`: Sampling frequency (Hz)
- `nFeatures = 4`: Number of features for neural network

### `my_NRChannel_modified.m`

Custom 5G NR channel model function supporting multiple channel profiles.

**Function Signature:**
```matlab
[signalOut, ofdmResponse, timingOffset, fs, tau_rms, Bc] = ...
    my_NRChannel_modified(signalIn, ch_type, OFDM_Response, cfgDL)
```

**Inputs:**
- `signalIn`: Complex IQ signal from `nrWaveformGenerator`
- `ch_type`: Channel type - `'HFS'`, `'MFS'`, or `'FF'`
- `OFDM_Response`: Boolean - return OFDM frequency response if true
- `cfgDL`: (Optional) `nrDLCarrierConfig` object

**Outputs:**
- `signalOut`: Filtered IQ signal after channel
- `ofdmResponse`: OFDM channel response `H(f,t)` if requested
- `timingOffset`: Timing offset from channel
- `fs`: Sample rate used
- `tau_rms`: RMS delay spread (seconds)
- `Bc`: Coherence bandwidth estimate (Hz)

**Channel Types:**

1. **HFS (Highly Frequency Selective)**
   - Path delays: [0, 0.30, 0.70, 1.20, 1.80] μs
   - NLOS (no line-of-sight)
   - High frequency selectivity

2. **MFS (Moderately Frequency Selective)**
   - Path delays: [0, 30, 65, 110, 160] ns
   - LOS with K-factor = 3 dB
   - Moderate frequency selectivity

3. **FF (Frequency-Flat)**
   - Single path (delay = 0)
   - Strong LOS (K-factor = 20 dB)
   - Minimal frequency selectivity

**Example:**
```matlab
[signalOut, H_freq] = my_NRChannel_modified(signalIn, 'HFS', true);
```

### `my_pspectrum.m`

Power spectral density computation function.

**Usage:**
```matlab
[p_spect, f_spect] = my_pspectrum(signal, fs, type, title_str)
```

**Parameters:**
- `signal`: Input signal (complex or real)
- `fs`: Sampling frequency
- `type`: `'power'`, `'spectrogram'`, or `'persistence'`
- `title_str`: Plot title

**Outputs:**
- `p_spect`: Power spectral density
- `f_spect`: Frequency vector

### `Passive_Signal.m`

Generates passive sensor signal with specified power level.

**Function Signature:**
```matlab
passive_signal = Passive_Signal(p_n, N)
```

**Parameters:**
- `p_n`: Passive signal power in dB
- `N`: Number of samples

**Output:**
- `passive_signal`: Complex passive sensor signal

### `SIC1_DL_Reg.m` / `SIC2_DL_Reg.m`

Deep learning-based Successive Interference Cancellation models.

**Purpose:**
- Separate 5G active signal from passive sensor signal
- Reduce interference at passive sensor
- Enable coexistence in shared spectrum

**Inputs:**
- Combined signal (active + passive)
- PSD features

**Outputs:**
- Reconstructed active signal
- Filtered passive signal (after SIC)

## Dependencies

- **MATLAB 5G Toolbox** (required):
  - `nrWaveformGenerator`
  - `nrCDLChannel`
  - `nrCarrierConfig`
  - `nrOFDMInfo`

- **MATLAB Signal Processing Toolbox**:
  - `pspectrum` (or custom `my_pspectrum`)

- **MATLAB Deep Learning Toolbox** (for SIC models):
  - Neural network training functions

## Channel Model Details

### CDL Configuration

The channel model uses 3GPP CDL (Clustered Delay Line) with:
- Carrier frequency: 3.5 GHz
- Configurable delay profiles
- Doppler effects (Max Doppler: 5 Hz for HFS/MFS)
- SISO (Single Input Single Output) antenna configuration

### OFDM Parameters

Default numerology:
- Subcarrier spacing: 15 kHz
- Resource blocks: 159 (or configurable via `cfgDL`)
- Cyclic prefix: 72 samples
- FFT size: 1028 (example from `Channel.py`)

## Integration with Python DRL Agent

To integrate with Python DRL training:

1. **Option 1: MATLAB Engine API**
   ```python
   import matlab.engine
   eng = matlab.engine.start_matlab()
   result = eng.my_NRChannel_modified(signal, 'HFS', True)
   ```

2. **Option 2: Shared Data Files**
   - MATLAB writes PSD data to file
   - Python reads and processes
   - Python writes action to file
   - MATLAB reads action and updates simulation

3. **Option 3: ZMQ/MATLAB Interface**
   - Real-time bidirectional communication
   - Low-latency state/action exchange

## Example Workflow

```matlab
% 1. Generate or load 5G signal
signalIn = nrWaveformGenerator(cfgDL);

% 2. Apply channel
[signalOut, H_freq] = my_NRChannel_modified(signalIn, 'HFS', true);

% 3. Add passive sensor signal
passive_signal = Passive_Signal(-120, length(signalOut));
total_signal = signalOut + passive_signal;

% 4. Compute PSD
[p_spect, f_spect] = my_pspectrum(total_signal, fs, 'power', 'Total Signal');

% 5. Apply SIC (if trained model available)
filtered_signal = SIC1_DL_Reg(total_signal, p_spect);
```

## Notes

- File paths in `ActivePassiveAI_Sensing.m` are hardcoded and need to be updated for your system
- The SIC models (`SIC1_DL_Reg.m`, `SIC2_DL_Reg.m`) require pre-trained neural networks
- Channel model parameters can be adjusted based on deployment scenario
