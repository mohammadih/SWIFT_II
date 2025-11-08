"""
Spectrum Environment for DRL-Assisted Spectrum Selection

This module implements the environment that emulates the shared spectrum
between 5G NR transceiver and passive sensor. The environment integrates
with MATLAB for realistic 5G physical-layer modeling.

The environment is modeled as a Markov Decision Process (MDP) where:
- State (s_t): Spectral occupancy features, interference estimates, CQI
- Action (a_t): Selection of sub-bands/RB groups for 5G transmission
- Reward (r_t): Jointly considers 5G throughput and interference mitigation
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


class SpectrumEnvironment:
    """
    Environment for DRL-assisted spectrum selection.
    
    Models the shared spectrum between 5G transceiver and passive sensor,
    providing state observations and rewards based on agent actions.
    """
    
    def __init__(
        self,
        num_subbands: int = 7,
        transmit_power: float = 5e-3,  # 5 mW
        noise_power: float = 1e-3,  # Thermal noise
        channel_type: str = 'HFS',  # Highly Frequency Selective
        alpha: float = 1.0,  # Throughput weight in reward
        beta: float = 1.0,  # Interference weight in reward
        interference_threshold: float = 2.0,  # P_int / P_th threshold
        use_matlab: bool = False,
        matlab_engine: Optional[object] = None
    ):
        """
        Initialize spectrum environment.
        
        Args:
            num_subbands: Number of available sub-bands/RB groups (N)
            transmit_power: 5G transmit power (Watts)
            noise_power: Thermal noise power (Watts)
            channel_type: Channel model type ('HFS', 'MFS', 'FF')
            alpha: Weight for throughput in reward function
            beta: Weight for interference in reward function
            interference_threshold: Threshold for interference ratio
            use_matlab: Whether to use MATLAB for PHY simulation
            matlab_engine: MATLAB engine instance if use_matlab=True
        """
        self.num_subbands = num_subbands
        self.transmit_power = transmit_power
        self.noise_power = noise_power
        self.channel_type = channel_type
        self.alpha = alpha
        self.beta = beta
        self.interference_threshold = interference_threshold
        self.use_matlab = use_matlab
        self.matlab_engine = matlab_engine
        
        # Initialize environment state
        self.current_state = None
        self.current_subband = None
        
        # Channel parameters (can be updated from MATLAB)
        self.channel_gains = None
        self.interference_power = None
        self.psd_values = None
        
        # Statistics
        self.episode_rewards = []
        self.episode_throughput = []
        self.episode_interference = []
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state observation: [PSD, CQI, interference_ratio]
        """
        # Initialize channel gains (simplified model, can be replaced with MATLAB)
        self.channel_gains = np.random.uniform(0.4, 0.9, self.num_subbands)
        
        # Initialize interference power per sub-band
        self.interference_power = np.random.uniform(1e-3, 5e-3, self.num_subbands)
        
        # Initialize PSD values (in dBm, converted to linear)
        base_psd_db = np.array([-54, -45, -48, -51, -50, -49, -56])[:self.num_subbands]
        jitter = np.random.normal(0, 1, self.num_subbands)
        psd_values_db = base_psd_db + jitter
        self.psd_values = 10 ** (psd_values_db / 10)  # Convert to linear scale (mW)
        
        # Compute initial state
        self.current_state = self._compute_state()
        self.current_subband = None
        
        return self.current_state
    
    def _compute_state(self) -> np.ndarray:
        """
        Compute state representation: [PSD, CQI, interference_ratio]
        
        State components:
        - PSD: Power Spectral Density per sub-band
        - CQI: Channel Quality Indicator (derived from SINR)
        - I_t: Interference ratio (P_int / P_th)
        
        Returns:
            State vector of shape (num_subbands, 3)
        """
        # Normalize PSD values
        psd_normalized = self.psd_values / np.max(self.psd_values) if np.max(self.psd_values) > 0 else self.psd_values
        
        # Compute CQI from SINR estimates
        sinr_per_subband = self._compute_sinr()
        cqi = np.log2(1 + sinr_per_subband)  # CQI as spectral efficiency
        
        # Normalize CQI
        cqi_normalized = cqi / np.max(cqi) if np.max(cqi) > 0 else cqi
        
        # Interference ratio
        interference_ratio = self.interference_power / self.noise_power
        
        # Stack into state matrix: (num_subbands, 3)
        state = np.column_stack([psd_normalized, cqi_normalized, interference_ratio])
        
        return state
    
    def _compute_sinr(self) -> np.ndarray:
        """
        Compute Signal-to-Interference-plus-Noise Ratio per sub-band.
        
        SINR = (H * P_tx) / (σ² + P_int)
        
        Returns:
            SINR values per sub-band
        """
        signal_power = self.channel_gains * self.transmit_power
        interference_plus_noise = self.noise_power + self.interference_power
        sinr = signal_power / (interference_plus_noise + 1e-10)  # Avoid division by zero
        return sinr
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Selected sub-band index for 5G transmission
            
        Returns:
            next_state: Next state observation
            reward: Reward for the action
            done: Whether episode is done
            info: Additional information (throughput, interference, etc.)
        """
        if action < 0 or action >= self.num_subbands:
            raise ValueError(f"Invalid action: {action}. Must be in [0, {self.num_subbands-1}]")
        
        self.current_subband = action
        
        # Simulate PHY layer (can use MATLAB here)
        if self.use_matlab and self.matlab_engine is not None:
            throughput, interference, psd_next = self._simulate_phy_matlab(action)
        else:
            throughput, interference, psd_next = self._simulate_phy_simplified(action)
        
        # Update environment state
        self.psd_values = psd_next
        self.interference_power[action] += 0.1 * self.transmit_power  # Increase interference on selected band
        
        # Compute reward
        reward = self._compute_reward(throughput, interference)
        
        # Update state
        next_state = self._compute_state()
        self.current_state = next_state
        
        # Episode termination (can be customized)
        done = False
        
        # Store statistics
        info = {
            'throughput': throughput,
            'interference': interference,
            'sinr': self._compute_sinr()[action],
            'selected_subband': action
        }
        
        return next_state, reward, done, info
    
    def _simulate_phy_simplified(self, action: int) -> Tuple[float, float, np.ndarray]:
        """
        Simplified PHY simulation (used when MATLAB is not available).
        
        Args:
            action: Selected sub-band
            
        Returns:
            throughput: Achieved throughput (bits/s/Hz)
            interference: Interference power at passive sensor
            psd_next: Updated PSD values
        """
        # Compute SINR for selected sub-band
        sinr = self._compute_sinr()[action]
        
        # Throughput using Shannon capacity
        throughput = np.log2(1 + sinr)
        
        # Interference at passive sensor (simplified model)
        interference = self.interference_power[action] + 0.1 * self.transmit_power * self.channel_gains[action]
        
        # Update PSD with some variation
        jitter = np.random.normal(0, 0.5, self.num_subbands)
        psd_next = self.psd_values * (1 + 0.1 * jitter)
        psd_next = np.clip(psd_next, 1e-6, 1e-3)  # Reasonable bounds
        
        return throughput, interference, psd_next
    
    def _simulate_phy_matlab(self, action: int) -> Tuple[float, float, np.ndarray]:
        """
        MATLAB-based PHY simulation (requires MATLAB engine).
        
        This method calls MATLAB functions for realistic 5G NR simulation:
        - nrWaveformGenerator for OFDM waveform
        - nrCDLChannel for channel modeling
        - Spectrum analyzer for PSD extraction
        
        Args:
            action: Selected sub-band
            
        Returns:
            throughput: Achieved throughput
            interference: Interference power
            psd_next: Updated PSD values
        """
        if self.matlab_engine is None:
            warnings.warn("MATLAB engine not available, using simplified model")
            return self._simulate_phy_simplified(action)
        
        try:
            # Call MATLAB function for PHY simulation
            # This is a placeholder - actual implementation depends on MATLAB setup
            result = self.matlab_engine.PHY_simulate(
                float(action),
                self.transmit_power,
                self.channel_type
            )
            
            throughput = float(result['throughput'])
            interference = float(result['interference'])
            psd_next = np.array(result['psd']).flatten()
            
            return throughput, interference, psd_next
            
        except Exception as e:
            warnings.warn(f"MATLAB simulation failed: {e}. Using simplified model.")
            return self._simulate_phy_simplified(action)
    
    def _compute_reward(self, throughput: float, interference: float) -> float:
        """
        Compute reward function.
        
        Reward formulation from paper:
        r_t = α * log2(1 + γ_t) - β * I_t
        
        where:
        - α: Throughput weight
        - γ_t: SINR/throughput
        - β: Interference weight
        - I_t: Interference ratio (P_int / P_th)
        
        Args:
            throughput: Achieved throughput
            interference: Interference power
            
        Returns:
            Reward value
        """
        # Throughput component
        throughput_reward = self.alpha * throughput
        
        # Interference penalty (normalized)
        interference_ratio = interference / self.noise_power
        interference_penalty = self.beta * interference_ratio
        
        # Combined reward
        reward = throughput_reward - interference_penalty
        
        return reward
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.current_state.copy() if self.current_state is not None else None
    
    def get_statistics(self) -> Dict:
        """Get environment statistics."""
        return {
            'num_subbands': self.num_subbands,
            'current_subband': self.current_subband,
            'avg_psd': np.mean(self.psd_values) if self.psd_values is not None else 0,
            'avg_interference': np.mean(self.interference_power) if self.interference_power is not None else 0
        }
