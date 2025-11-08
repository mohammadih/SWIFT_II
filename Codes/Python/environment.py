"""
Environment Module for SWIFT II DRL-Assisted Spectrum Selection
Implements the MDP (Markov Decision Process) formulation
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque
from config import ConfigManager


class SpectrumEnvironment:
    """
    Spectrum Sharing Environment for 5G-Passive Sensor Coexistence
    
    Implements the MDP formulation from the paper:
    - State: [PSD(f1,t), ..., PSD(fn,t), CQI_t, I_t]
    - Action: Selection of sub-band for 5G transmission
    - Reward: α·log2(1 + SINR) - β·I_t
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the spectrum environment
        
        Args:
            config: Configuration manager instance
        """
        if config is None:
            from config import default_config
            config = default_config
        
        self.config = config
        self.num_channels = config.spectrum.num_channels
        self.num_sensing = config.spectrum.num_sensing_channels
        self.temporal_depth = config.ddqn.temporal_depth
        
        # Physical layer parameters
        self.tx_power = config.channel.tx_power
        self.noise_power = config.channel.thermal_noise_power
        self.channel_gain_signal = config.channel.channel_gain_signal
        self.channel_gain_jammer = config.channel.channel_gain_jammer
        
        # Reward parameters
        self.alpha = config.reward.alpha_throughput
        self.beta = config.reward.beta_interference
        self.gamma = config.reward.gamma_jammer
        self.sinr_threshold = config.reward.sinr_threshold_linear
        self.reward_scale = config.reward.reward_scale
        self.reward_clip = config.reward.reward_clip_range
        
        # State history for temporal representation
        self.state_history = deque(maxlen=self.temporal_depth)
        
        # Current environment state
        self.interference_power = None
        self.jammer_power = None
        self.channel_gains = None
        self.psd_values = None
        
        # Performance tracking
        self.episode_reward = 0.0
        self.episode_interference = 0.0
        self.episode_throughput = 0.0
        self.step_count = 0
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial state observation
        """
        # Clear state history
        self.state_history.clear()
        
        # Initialize environment parameters
        self._initialize_environment()
        
        # Create initial indication matrices to build state history
        for _ in range(self.temporal_depth):
            sinr = self._calculate_sinr(0)  # Default channel 0
            sensing_actions = np.random.choice(
                self.num_channels, 
                size=self.num_sensing, 
                replace=False
            )
            psds = self.psd_values[sensing_actions]
            
            indication_matrix = self._create_indication_matrix(
                communication_action=0,
                sinr=sinr,
                sensing_actions=sensing_actions,
                psds=psds
            )
            self.state_history.append(indication_matrix)
        
        # Reset performance tracking
        self.episode_reward = 0.0
        self.episode_interference = 0.0
        self.episode_throughput = 0.0
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment
        
        Args:
            action: Channel index to use for 5G transmission
        
        Returns:
            observation: Next state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Validate action
        assert 0 <= action < self.num_channels, f"Invalid action: {action}"
        
        # Update environment dynamics
        self._update_environment()
        
        # Calculate SINR for selected channel
        sinr = self._calculate_sinr(action)
        
        # Select sensing channels (random or based on policy)
        sensing_actions = np.random.choice(
            self.num_channels,
            size=self.num_sensing,
            replace=False
        )
        psds = self.psd_values[sensing_actions]
        
        # Calculate reward
        reward = self._calculate_reward(action, sinr)
        
        # Create new indication matrix and update state
        indication_matrix = self._create_indication_matrix(
            communication_action=action,
            sinr=sinr,
            sensing_actions=sensing_actions,
            psds=psds
        )
        self.state_history.append(indication_matrix)
        
        # Get next state
        next_state = self._get_state()
        
        # Update performance metrics
        self.step_count += 1
        self.episode_reward += reward
        self.episode_interference += self.interference_power[action]
        self.episode_throughput += np.log2(1 + sinr)
        
        # Check if episode is done
        done = self.step_count >= self.config.simulation.steps_per_episode
        
        # Prepare info dictionary
        info = {
            'sinr': sinr,
            'sinr_db': 10 * np.log10(sinr + 1e-10),
            'interference': self.interference_power[action],
            'jammer_power': self.jammer_power[action],
            'throughput': np.log2(1 + sinr),
            'channel_selected': action,
            'avg_reward': self.episode_reward / self.step_count,
            'avg_interference': self.episode_interference / self.step_count,
            'avg_throughput': self.episode_throughput / self.step_count
        }
        
        return next_state, reward, done, info
    
    def _initialize_environment(self):
        """Initialize or reset environment parameters"""
        # Generate random interference and jammer power for each channel
        self.interference_power = np.random.uniform(1.0, 5.0, self.num_channels)
        self.jammer_power = np.random.uniform(1.0, 5.0, self.num_channels)
        
        # Generate channel gains
        self.channel_gains = np.random.uniform(
            *self.config.channel.channel_gain_interference_range,
            self.num_channels
        )
        
        # Generate PSD values (in dBm, then convert to linear)
        base_psd_db = np.array([-54, -45, -48, -51, -50, -49, -56])
        if len(base_psd_db) != self.num_channels:
            # If different number of channels, generate random PSD
            base_psd_db = np.random.uniform(-60, -40, self.num_channels)
        
        jitter = np.random.normal(0, 1, self.num_channels)
        psd_db = base_psd_db + jitter
        self.psd_values = 10 ** (psd_db / 10)  # Convert to linear scale (mW)
    
    def _update_environment(self):
        """Update environment dynamics (time-varying channel)"""
        # Add temporal variation to interference and jammer
        self.interference_power += np.random.normal(0, 0.1, self.num_channels)
        self.interference_power = np.clip(self.interference_power, 0.5, 10.0)
        
        self.jammer_power += np.random.normal(0, 0.1, self.num_channels)
        self.jammer_power = np.clip(self.jammer_power, 0.5, 10.0)
        
        # Update channel gains (fading)
        self.channel_gains += np.random.normal(0, 0.05, self.num_channels)
        self.channel_gains = np.clip(
            self.channel_gains,
            *self.config.channel.channel_gain_interference_range
        )
        
        # Update PSD values
        psd_db = 10 * np.log10(self.psd_values + 1e-10)
        psd_db += np.random.normal(0, 1, self.num_channels)
        self.psd_values = 10 ** (psd_db / 10)
    
    def _calculate_sinr(self, channel: int) -> float:
        """
        Calculate SINR for a given channel
        
        SINR = (h_s * P_s) / (σ² + Σ(h_i * P_i) + h_j * Σ(P_j))
        
        Args:
            channel: Channel index
        
        Returns:
            SINR value (linear scale)
        """
        signal_power = self.channel_gain_signal * self.tx_power
        
        interference_total = np.sum(self.channel_gains * self.interference_power)
        jammer_total = self.channel_gain_jammer * np.sum(self.jammer_power)
        
        noise_and_interference = self.noise_power + interference_total + jammer_total
        
        sinr = signal_power / (noise_and_interference + 1e-10)
        
        return max(sinr, 1e-6)  # Prevent numerical issues
    
    def _calculate_reward(self, action: int, sinr: float) -> float:
        """
        Calculate multi-objective reward
        
        r_t = α·log2(1 + SINR) - β·I_t - γ·J_t
        
        Args:
            action: Selected channel
            sinr: Calculated SINR
        
        Returns:
            Reward value
        """
        # Throughput component (Shannon capacity approximation)
        throughput_reward = self.alpha * np.log2(1 + sinr)
        
        # Interference penalty
        interference_penalty = self.beta * self.interference_power[action]
        
        # Jammer penalty
        jammer_penalty = self.gamma * self.jammer_power[action]
        
        # Combined reward
        reward = throughput_reward - interference_penalty - jammer_penalty
        
        # Scale and clip reward
        reward = reward * self.reward_scale
        reward = np.clip(reward, *self.reward_clip)
        
        return reward
    
    def _create_indication_matrix(
        self,
        communication_action: int,
        sinr: float,
        sensing_actions: np.ndarray,
        psds: np.ndarray
    ) -> np.ndarray:
        """
        Create indication matrix for state representation
        
        Format: [channel_idx, reward, interference, jammer]
        
        Args:
            communication_action: Selected communication channel
            sinr: Calculated SINR
            sensing_actions: Array of sensing channel indices
            psds: PSD values for sensing channels
        
        Returns:
            Indication matrix of shape (num_sensing + 1, 4)
        """
        indication_matrix = []
        
        # Communication channel entry
        comm_reward = self._calculate_reward(communication_action, sinr)
        indication_matrix.append([
            communication_action,
            comm_reward,
            self.interference_power[communication_action],
            self.jammer_power[communication_action]
        ])
        
        # Sensing channel entries
        for action, psd in zip(sensing_actions, psds):
            sensing_reward = self._sensing_feedback(psd)
            indication_matrix.append([
                action,
                sensing_reward,
                self.interference_power[action],
                self.jammer_power[action]
            ])
        
        return np.array(indication_matrix, dtype=np.float32)
    
    def _sensing_feedback(self, psd: float) -> float:
        """
        Calculate sensing reward based on PSD
        
        Args:
            psd: Power spectral density value
        
        Returns:
            Sensing reward
        """
        psd_threshold = self.config.passive_sensor.psd_threshold
        
        if psd < psd_threshold:
            return self.reward_scale  # Good - low interference
        else:
            return -self.reward_scale  # Bad - high interference
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        State is a stack of indication matrices over temporal depth:
        Shape: (num_channels, num_features, temporal_depth)
        
        Returns:
            State array
        """
        if len(self.state_history) < self.temporal_depth:
            # Pad with zeros if not enough history
            padding_needed = self.temporal_depth - len(self.state_history)
            state_matrices = [np.zeros_like(self.state_history[0])] * padding_needed
            state_matrices.extend(list(self.state_history))
        else:
            state_matrices = list(self.state_history)
        
        # Stack along temporal dimension
        state = np.stack(state_matrices, axis=-1)  # (channels, features, time)
        
        return state.astype(np.float32)
    
    def get_info(self) -> Dict:
        """Get current environment information"""
        return {
            'num_channels': self.num_channels,
            'interference_power': self.interference_power.copy(),
            'jammer_power': self.jammer_power.copy(),
            'channel_gains': self.channel_gains.copy(),
            'psd_values': self.psd_values.copy(),
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'episode_interference': self.episode_interference,
            'episode_throughput': self.episode_throughput
        }


class MATLABPhyEnvironment(SpectrumEnvironment):
    """
    Extended environment with MATLAB PHY layer integration
    
    Uses MATLAB engine to run realistic 5G NR physical layer simulation
    """
    
    def __init__(self, config: Optional[ConfigManager] = None, matlab_engine=None):
        """
        Initialize environment with MATLAB integration
        
        Args:
            config: Configuration manager
            matlab_engine: Initialized MATLAB engine instance
        """
        super().__init__(config)
        self.matlab_engine = matlab_engine
        self.use_matlab = config.simulation.use_matlab_phy if config else False
        
        if self.use_matlab and self.matlab_engine is None:
            raise ValueError("MATLAB engine must be provided when use_matlab_phy=True")
    
    def _calculate_sinr(self, channel: int) -> float:
        """
        Calculate SINR using MATLAB PHY layer if available
        
        Args:
            channel: Channel index
        
        Returns:
            SINR value
        """
        if self.use_matlab and self.matlab_engine is not None:
            try:
                # Call MATLAB function to get SINR
                # sinr = self.matlab_engine.calculate_sinr(channel, nargout=1)
                # return float(sinr)
                pass  # Placeholder for MATLAB integration
            except Exception as e:
                print(f"MATLAB SINR calculation failed: {e}. Using default calculation.")
        
        # Fall back to parent implementation
        return super()._calculate_sinr(channel)
    
    def _update_environment(self):
        """Update environment with MATLAB channel simulation"""
        if self.use_matlab and self.matlab_engine is not None:
            try:
                # Call MATLAB to update channel state
                # self.matlab_engine.update_channel_state()
                pass  # Placeholder for MATLAB integration
            except Exception as e:
                print(f"MATLAB environment update failed: {e}. Using default update.")
        
        # Fall back to parent implementation
        super()._update_environment()


if __name__ == "__main__":
    # Example usage and testing
    from config import ConfigManager
    
    config = ConfigManager()
    config.simulation.steps_per_episode = 50
    
    print("Testing SpectrumEnvironment...")
    print("-" * 80)
    
    env = SpectrumEnvironment(config)
    
    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Expected shape: {config.get_state_shape()}")
    
    # Test episode
    total_reward = 0
    for step in range(10):
        action = np.random.randint(0, env.num_channels)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  SINR: {info['sinr_db']:.2f} dB")
        print(f"  Throughput: {info['throughput']:.3f} bits/s/Hz")
        print(f"  Done: {done}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Average reward: {total_reward / (step + 1):.3f}")
