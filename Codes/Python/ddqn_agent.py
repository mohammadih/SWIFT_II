"""
DDQN Agent Module for SWIFT II DRL-Assisted Spectrum Selection
Implements Double Deep Q-Network with paper-aligned architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
from typing import Tuple, List, Optional
import os
from datetime import datetime

from config import ConfigManager


class ReplayBuffer:
    """
    Experience Replay Buffer for DDQN training
    
    Stores transitions (s, a, r, s', done) for experience replay
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            max_size: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, experience: Tuple):
        """
        Add experience to buffer
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            List of sampled experiences
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()


class DDQNAgent:
    """
    Double Deep Q-Network Agent
    
    Implements DDQN algorithm with:
    - Separate online and target networks
    - Experience replay
    - ε-greedy exploration with decay
    - CNN-based state feature extraction
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize DDQN agent
        
        Args:
            config: Configuration manager instance
        """
        if config is None:
            from config import default_config
            config = default_config
        
        self.config = config
        self.state_shape = config.get_state_shape()
        self.num_actions = config.get_action_space_size()
        
        # Hyperparameters
        self.learning_rate = config.ddqn.learning_rate
        self.discount_factor = config.ddqn.discount_factor
        self.batch_size = config.ddqn.batch_size
        self.target_update_freq = config.ddqn.target_update_frequency
        
        # Exploration parameters
        self.epsilon = config.ddqn.epsilon_initial
        self.epsilon_min = config.ddqn.epsilon_min
        self.epsilon_decay = config.ddqn.epsilon_decay_rate
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.ddqn.replay_buffer_size)
        
        # Training tracking
        self.training_step = 0
        self.losses = []
        self.q_values = []
        self.epsilon_history = []
        
        # Model saving
        self.model_save_dir = config.simulation.model_save_directory
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def _build_network(self) -> models.Model:
        """
        Build CNN-based Q-network
        
        Architecture:
        - Conv2D layer for spatial feature extraction
        - Pooling layer
        - Flatten
        - Dense layers
        - Output layer (Q-values for each action)
        
        Returns:
            Compiled Keras model
        """
        input_shape = self.state_shape
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Convolutional layer for feature extraction
            layers.Conv2D(
                filters=self.config.ddqn.conv_filters,
                kernel_size=self.config.ddqn.conv_kernel_size,
                activation='relu',
                padding='same',
                name='conv1'
            ),
            
            # Pooling layer
            layers.AveragePooling2D(pool_size=(2, 1), name='pool1'),
            
            # Additional conv layer for deeper feature extraction
            layers.Conv2D(
                filters=self.config.ddqn.conv_filters * 2,
                kernel_size=(2, 2),
                activation='relu',
                padding='same',
                name='conv2'
            ),
            
            # Flatten for dense layers
            layers.Flatten(name='flatten'),
            
            # Dense layers
            layers.Dense(
                self.config.ddqn.dense_units[0],
                activation='relu',
                name='dense1'
            ),
            layers.Dropout(0.2, name='dropout1'),
            
            layers.Dense(
                self.config.ddqn.dense_units[1],
                activation='relu',
                name='dense2'
            ),
            
            # Output layer (Q-values)
            layers.Dense(
                self.num_actions,
                activation='linear',
                name='q_values'
            )
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Copy weights from online network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy
        
        Args:
            state: Current state observation
            training: Whether in training mode (use epsilon-greedy)
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploit: greedy action
            q_values = self.predict(state)
            action = np.argmax(q_values)
        
        return action
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for a given state
        
        Args:
            state: State observation
        
        Returns:
            Q-values for all actions
        """
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        
        q_values = self.q_network.predict(state, verbose=0)
        return q_values[0] if q_values.shape[0] == 1 else q_values
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.add((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (DDQN update)
        
        Uses Double DQN update rule:
        Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if enough samples in buffer
        if self.replay_buffer.size() < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get current Q-values
        q_values = self.q_network.predict(states, verbose=0)
        
        # DDQN: Use online network to select action, target network to evaluate
        next_q_online = self.q_network.predict(next_states, verbose=0)
        next_q_target = self.target_network.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                # Double DQN: action selection with online, evaluation with target
                best_action = np.argmax(next_q_online[i])
                target = rewards[i] + self.discount_factor * next_q_target[i][best_action]
            
            q_values[i][actions[i]] = target
        
        # Train network
        history = self.q_network.fit(
            states,
            q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Track Q-values for monitoring
        mean_q = np.mean(q_values)
        self.q_values.append(mean_q)
        
        return loss
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def save_model(self, episode: int, suffix: str = ""):
        """
        Save model weights
        
        Args:
            episode: Current episode number
            suffix: Optional suffix for filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ddqn_ep{episode}_{timestamp}{suffix}"
        
        online_path = os.path.join(self.model_save_dir, f"{filename}_online.h5")
        target_path = os.path.join(self.model_save_dir, f"{filename}_target.h5")
        
        self.q_network.save_weights(online_path)
        self.target_network.save_weights(target_path)
        
        print(f"Model saved: {filename}")
    
    def load_model(self, filepath: str):
        """
        Load model weights
        
        Args:
            filepath: Path to model weights file
        """
        self.q_network.load_weights(filepath)
        self.target_network.load_weights(filepath)
        print(f"Model loaded from: {filepath}")
    
    def get_training_stats(self) -> dict:
        """
        Get training statistics
        
        Returns:
            Dictionary of training metrics
        """
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': self.replay_buffer.size(),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0.0,
            'total_losses': len(self.losses)
        }
    
    def reset_training_stats(self):
        """Reset training statistics"""
        self.losses = []
        self.q_values = []
        self.epsilon_history = []


class DDQNTrainer:
    """
    Trainer class for DDQN agent
    
    Manages the training loop, episode management, and logging
    """
    
    def __init__(
        self,
        agent: DDQNAgent,
        environment,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize trainer
        
        Args:
            agent: DDQN agent instance
            environment: Training environment
            config: Configuration manager
        """
        if config is None:
            from config import default_config
            config = default_config
        
        self.agent = agent
        self.env = environment
        self.config = config
        
        # Training parameters
        self.num_episodes = config.simulation.num_episodes
        self.checkpoint_freq = config.simulation.checkpoint_frequency
        self.eval_freq = config.simulation.eval_frequency
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_avg_losses = []
        self.episode_throughputs = []
        self.episode_interferences = []
    
    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("Starting DDQN Training")
        print("=" * 80)
        self.config.print_config()
        
        for episode in range(self.num_episodes):
            episode_reward, episode_info = self._run_episode(training=True)
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_info['length'])
            self.episode_throughputs.append(episode_info['avg_throughput'])
            self.episode_interferences.append(episode_info['avg_interference'])
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Logging
            if (episode + 1) % 10 == 0 or episode == 0:
                self._log_progress(episode)
            
            # Save checkpoint
            if (episode + 1) % self.checkpoint_freq == 0:
                self.agent.save_model(episode + 1, suffix="_checkpoint")
            
            # Evaluation
            if (episode + 1) % self.eval_freq == 0:
                self._evaluate(episode + 1)
        
        # Final save
        self.agent.save_model(self.num_episodes, suffix="_final")
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        
        return self.get_training_history()
    
    def _run_episode(self, training: bool = True) -> Tuple[float, dict]:
        """
        Run one episode
        
        Args:
            training: Whether to train during episode
        
        Returns:
            Total episode reward and info dictionary
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_losses = []
        step_count = 0
        
        done = False
        while not done:
            # Select action
            action = self.agent.select_action(state, training=training)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            if training:
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            step_count += 1
        
        episode_info = {
            'length': step_count,
            'avg_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'avg_throughput': info['avg_throughput'],
            'avg_interference': info['avg_interference']
        }
        
        return episode_reward, episode_info
    
    def _evaluate(self, episode: int, num_eval_episodes: int = 10):
        """
        Evaluate agent performance
        
        Args:
            episode: Current training episode
            num_eval_episodes: Number of evaluation episodes
        """
        print(f"\n[Evaluation at Episode {episode}]")
        
        eval_rewards = []
        eval_throughputs = []
        eval_interferences = []
        
        for _ in range(num_eval_episodes):
            reward, info = self._run_episode(training=False)
            eval_rewards.append(reward)
            eval_throughputs.append(info['avg_throughput'])
            eval_interferences.append(info['avg_interference'])
        
        print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Average Throughput: {np.mean(eval_throughputs):.3f}")
        print(f"  Average Interference: {np.mean(eval_interferences):.3f}")
    
    def _log_progress(self, episode: int):
        """
        Log training progress
        
        Args:
            episode: Current episode number
        """
        recent_rewards = self.episode_rewards[-10:]
        avg_reward = np.mean(recent_rewards)
        
        stats = self.agent.get_training_stats()
        
        print(f"\nEpisode {episode + 1}/{self.num_episodes}")
        print(f"  Reward: {self.episode_rewards[-1]:.2f} (avg: {avg_reward:.2f})")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Loss: {stats['avg_loss']:.4f}")
        print(f"  Q-value: {stats['avg_q_value']:.2f}")
        print(f"  Buffer: {stats['buffer_size']}/{self.agent.replay_buffer.max_size}")
    
    def get_training_history(self) -> dict:
        """Get complete training history"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_throughputs': self.episode_throughputs,
            'episode_interferences': self.episode_interferences,
            'losses': self.agent.losses,
            'q_values': self.agent.q_values,
            'epsilon_history': self.agent.epsilon_history
        }


if __name__ == "__main__":
    # Example usage
    from config import ConfigManager
    from environment import SpectrumEnvironment
    
    print("Initializing DDQN Agent...")
    
    config = ConfigManager()
    config.simulation.num_episodes = 100
    config.ddqn.batch_size = 32
    
    # Create environment and agent
    env = SpectrumEnvironment(config)
    agent = DDQNAgent(config)
    
    print(f"\nAgent initialized:")
    print(f"  State shape: {agent.state_shape}")
    print(f"  Action space: {agent.num_actions}")
    print(f"  Network parameters: {agent.q_network.count_params()}")
    
    # Test action selection
    state = env.reset()
    action = agent.select_action(state)
    print(f"\nTest action selection: {action}")
    print(f"Q-values: {agent.predict(state)}")
