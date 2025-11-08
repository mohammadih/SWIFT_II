"""
SWIFT II - Deep Reinforcement Learning-Assisted Spectrum Selection
for RFI-Resilient Passive Sensing in Shared 5G Environments

This module implements the Double Deep Q-Network (DDQN) agent for dynamic
spectrum selection in a shared 5G-passive sensing environment.

Reference: "SWIFT II – Deep Reinforcement Learning-Assisted Spectrum Selection 
for RFI-Resilient Passive Sensing in Shared 5G Environments"
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
import os


class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) agent for spectrum selection.
    
    Implements the DDQN algorithm to address Q-value overestimation and
    convergence instability found in standard DQN.
    
    The agent learns optimal spectrum access policies by observing time-varying
    spectrum states through CNN-based feature extraction.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        memory_size: int = 10000,
        batch_size: int = 32
    ):
        """
        Initialize DDQN agent.
        
        Args:
            state_shape: Shape of the state input (e.g., (N_subbands, features))
            num_actions: Number of available sub-bands/RB groups
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            target_update_freq: Frequency of target network updates
            memory_size: Replay buffer capacity
            batch_size: Batch size for training
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Build networks
        self.q_network = self._build_network(state_shape, num_actions)
        self.target_network = self._build_network(state_shape, num_actions)
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=memory_size)
        
        # Training statistics
        self.losses = []
        self.q_values_history = []
        self.update_counter = 0
        
    def _build_network(self, input_shape: Tuple[int, ...], num_actions: int) -> models.Model:
        """
        Build the Q-network with CNN-based feature extraction.
        
        The network uses convolutional layers to extract spectral features
        from the state representation (PSD, CQI, interference ratio).
        
        Args:
            input_shape: Shape of input state
            num_actions: Number of output actions
            
        Returns:
            Compiled Keras model
        """
        # Reshape input if needed for CNN
        if len(input_shape) == 2:
            # Add channel dimension for CNN: (N_subbands, features) -> (N_subbands, features, 1)
            input_layer = layers.Input(shape=(*input_shape, 1))
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.AveragePooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)
        else:
            # For 1D or already processed features
            input_layer = layers.Input(shape=input_shape)
            if len(input_shape) == 1:
                x = layers.Dense(128, activation='relu')(input_layer)
                x = layers.Dense(64, activation='relu')(x)
            else:
                # 2D input: (N_subbands, features)
                x = layers.Flatten()(input_layer)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dense(64, activation='relu')(x)
        
        # Output layer: Q-values for each action
        output = layers.Dense(num_actions, activation='linear', name='q_values')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def update_target_network(self):
        """Update target network weights from online network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action (sub-band index)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Reshape state for prediction
        state_reshaped = self._prepare_state(state)
        q_values = self.q_network.predict(state_reshaped, verbose=0)
        return np.argmax(q_values[0])
    
    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        """Prepare state for network input."""
        if len(self.state_shape) == 2 and len(state.shape) == 2:
            # Add channel dimension for CNN
            return state[np.newaxis, ..., np.newaxis]
        else:
            return state[np.newaxis, ...]
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using DDQN update rule.
        
        DDQN update rule:
        L(θ) = E[(r_t + γ Q_θ⁻(s_{t+1}, argmax_a' Q_θ(s_{t+1}, a')) - Q_θ(s_t, a_t))²]
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample minibatch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array([self._prepare_state(s)[0] for s in states])
        next_states = np.array([self._prepare_state(s)[0] for s in next_states])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values using online network for action selection
        next_q_values_online = self.q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values_online, axis=1)
        
        # Target Q-values using target network
        next_q_values_target = self.target_network.predict(next_states, verbose=0)
        
        # Compute targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q_values_target[i, next_actions[i]]
        
        # Train the network
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save the Q-network model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved Q-network model."""
        self.q_network = models.load_model(filepath)
        self.update_target_network()
        print(f"Model loaded from {filepath}")


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Implements experience replay to break correlation between consecutive
    experiences and improve learning stability.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
