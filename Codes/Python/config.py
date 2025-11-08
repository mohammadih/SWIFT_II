"""
Configuration Module for SWIFT II DRL-Assisted Spectrum Selection
Based on paper: "Deep Reinforcement Learning-Assisted Spectrum Selection 
for RFI-Resilient Passive Sensing in Shared 5G Environments"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SpectrumConfig:
    """Spectrum and frequency configuration"""
    # Number of sub-bands or resource block groups
    num_channels: int = 7
    
    # Number of sensing channels for passive sensor
    num_sensing_channels: int = 2
    
    # Carrier frequency (Hz)
    carrier_frequency: float = 3.5e9
    
    # Bandwidth per channel (Hz)
    channel_bandwidth: float = 1.4e6
    
    # Sampling rate (Hz)
    sampling_rate: float = 30.72e6
    
    # Frequency range for each channel (Hz)
    frequency_range: Tuple[float, float] = (-15.36e6, 15.36e6)


@dataclass
class ChannelConfig:
    """5G NR Channel Model Configuration"""
    # Channel model type: "CDL-A", "CDL-B", "CDL-C", "CDL-D", "TDL-A", etc.
    channel_model: str = "CDL-D"
    
    # Delay spread (seconds)
    delay_spread: float = 30e-9
    
    # UE velocity (m/s)
    min_speed: float = 3.0
    max_speed: float = 30.0
    
    # Transmit power (Watts)
    tx_power: float = 5e-3  # 5 mW
    
    # Noise power spectral density (dBm/Hz)
    noise_psd: float = -174.0
    
    # Thermal noise power (Watts)
    thermal_noise_power: float = 1e-3
    
    # Channel gain parameters
    channel_gain_signal: float = 0.8
    channel_gain_jammer: float = 0.7
    channel_gain_interference_range: Tuple[float, float] = (0.4, 0.9)


@dataclass
class PassiveSensorConfig:
    """Passive Sensor Configuration"""
    # Passive signal power level (dBm)
    passive_signal_power_dbm: float = -120
    
    # Noise floor (dBm)
    noise_floor_dbm: float = -90
    
    # PSD threshold for sensing feedback (linear scale)
    psd_threshold: float = 2.0
    
    # Sensing feedback latency (seconds)
    sensing_latency: float = 1e-3


@dataclass
class DDQNConfig:
    """DDQN Agent Configuration"""
    # Network architecture
    conv_filters: int = 16
    conv_kernel_size: Tuple[int, int] = (2, 2)
    dense_units: List[int] = None  # Additional dense layers
    
    # Training hyperparameters
    learning_rate: float = 0.0001
    discount_factor: float = 0.95  # γ (gamma)
    batch_size: int = 64
    replay_buffer_size: int = 10000
    
    # Exploration parameters
    epsilon_initial: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay_rate: float = 0.995
    
    # Target network update
    target_update_frequency: int = 100  # Update every C steps
    
    # State representation
    temporal_depth: int = 4  # T - number of time steps in state
    state_features: int = 4  # Features per channel: [channel_idx, reward, interference, jammer]
    
    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [128, 64]


@dataclass
class RewardConfig:
    """Reward Function Configuration"""
    # Throughput reward weight (α)
    alpha_throughput: float = 1.0
    
    # Interference penalty weight (β)
    beta_interference: float = 0.5
    
    # Jammer penalty weight
    gamma_jammer: float = 0.1
    
    # SINR threshold for communication (dB)
    sinr_threshold_db: float = 10.0
    
    # SINR threshold (linear scale)
    sinr_threshold_linear: float = 10.0  # 10^(10/10)
    
    # Reward scaling factor
    reward_scale: float = 10.0
    
    # Reward clipping range
    reward_clip_range: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class SimulationConfig:
    """Overall Simulation Configuration"""
    # Number of training episodes
    num_episodes: int = 1000
    
    # Time steps per episode
    steps_per_episode: int = 200
    
    # Model checkpoint frequency (episodes)
    checkpoint_frequency: int = 100
    
    # Evaluation frequency (episodes)
    eval_frequency: int = 50
    
    # Number of evaluation episodes
    num_eval_episodes: int = 10
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Enable MATLAB integration
    use_matlab_phy: bool = False
    
    # MATLAB engine timeout (seconds)
    matlab_timeout: float = 300.0
    
    # Logging and visualization
    enable_tensorboard: bool = True
    log_directory: str = "./logs"
    model_save_directory: str = "./models"
    figure_save_directory: str = "./Figures"
    
    # Verbosity level (0: minimal, 1: normal, 2: detailed)
    verbosity: int = 1


class ConfigManager:
    """Unified configuration manager for SWIFT II simulation"""
    
    def __init__(self):
        self.spectrum = SpectrumConfig()
        self.channel = ChannelConfig()
        self.passive_sensor = PassiveSensorConfig()
        self.ddqn = DDQNConfig()
        self.reward = RewardConfig()
        self.simulation = SimulationConfig()
    
    def get_state_shape(self) -> Tuple[int, int, int]:
        """
        Calculate state shape for DDQN input
        Returns: (num_channels, num_features, temporal_depth)
        """
        num_channels = self.spectrum.num_sensing_channels + 1  # +1 for communication channel
        num_features = self.ddqn.state_features
        temporal_depth = self.ddqn.temporal_depth
        return (num_channels, num_features, temporal_depth)
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return self.spectrum.num_channels
    
    def set_random_seed(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.simulation.random_seed)
        import random
        random.seed(self.simulation.random_seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(self.simulation.random_seed)
        except ImportError:
            pass
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 80)
        print("SWIFT II - DRL-Assisted Spectrum Selection Configuration")
        print("=" * 80)
        print(f"\n[Spectrum Configuration]")
        print(f"  Number of channels: {self.spectrum.num_channels}")
        print(f"  Sensing channels: {self.spectrum.num_sensing_channels}")
        print(f"  Carrier frequency: {self.spectrum.carrier_frequency/1e9:.2f} GHz")
        print(f"  Sampling rate: {self.spectrum.sampling_rate/1e6:.2f} MHz")
        
        print(f"\n[Channel Configuration]")
        print(f"  Channel model: {self.channel.channel_model}")
        print(f"  Delay spread: {self.channel.delay_spread*1e9:.1f} ns")
        print(f"  Tx power: {self.channel.tx_power*1e3:.2f} mW")
        
        print(f"\n[DDQN Configuration]")
        print(f"  Learning rate: {self.ddqn.learning_rate}")
        print(f"  Discount factor: {self.ddqn.discount_factor}")
        print(f"  Batch size: {self.ddqn.batch_size}")
        print(f"  Replay buffer size: {self.ddqn.replay_buffer_size}")
        print(f"  State shape: {self.get_state_shape()}")
        print(f"  Action space size: {self.get_action_space_size()}")
        
        print(f"\n[Reward Configuration]")
        print(f"  Throughput weight (α): {self.reward.alpha_throughput}")
        print(f"  Interference weight (β): {self.reward.beta_interference}")
        print(f"  SINR threshold: {self.reward.sinr_threshold_db} dB")
        
        print(f"\n[Simulation Configuration]")
        print(f"  Episodes: {self.simulation.num_episodes}")
        print(f"  Steps per episode: {self.simulation.steps_per_episode}")
        print(f"  MATLAB integration: {self.simulation.use_matlab_phy}")
        print(f"  Random seed: {self.simulation.random_seed}")
        print("=" * 80 + "\n")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'spectrum': self.spectrum.__dict__,
            'channel': self.channel.__dict__,
            'passive_sensor': self.passive_sensor.__dict__,
            'ddqn': self.ddqn.__dict__,
            'reward': self.reward.__dict__,
            'simulation': self.simulation.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ConfigManager':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'spectrum' in config_dict:
            for key, value in config_dict['spectrum'].items():
                if hasattr(config.spectrum, key):
                    setattr(config.spectrum, key, value)
        
        if 'channel' in config_dict:
            for key, value in config_dict['channel'].items():
                if hasattr(config.channel, key):
                    setattr(config.channel, key, value)
        
        if 'passive_sensor' in config_dict:
            for key, value in config_dict['passive_sensor'].items():
                if hasattr(config.passive_sensor, key):
                    setattr(config.passive_sensor, key, value)
        
        if 'ddqn' in config_dict:
            for key, value in config_dict['ddqn'].items():
                if hasattr(config.ddqn, key):
                    setattr(config.ddqn, key, value)
        
        if 'reward' in config_dict:
            for key, value in config_dict['reward'].items():
                if hasattr(config.reward, key):
                    setattr(config.reward, key, value)
        
        if 'simulation' in config_dict:
            for key, value in config_dict['simulation'].items():
                if hasattr(config.simulation, key):
                    setattr(config.simulation, key, value)
        
        return config


# Create default configuration instance
default_config = ConfigManager()


if __name__ == "__main__":
    # Example usage
    config = ConfigManager()
    config.print_config()
    
    # Example: modify configuration
    config.ddqn.learning_rate = 0.0005
    config.simulation.num_episodes = 2000
    
    print("\n[Modified Configuration]")
    config.print_config()
