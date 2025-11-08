#%% if we want to split the spectrum to find the weakest channel (subcarrier)
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Define Environment Parameters:
N = 7  # Total number of channels (bins)
Ns = 2  # Number of sensing channels
T = 3  # Temporal memory depth
lambda_val = 10  # Reward weight
mu_th = 2  # SINR threshold
nu_th = 2  # PSD threshold
gamma = 0.4  # Discount factor
sigma_noise = 1e-3  # Noise power in mW
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Slower epsilon decay
learning_rate = 0.0005  # Reduced learning rate for stability
batch_size = 32  # Batch size for training
replay_buffer_size = 50  # Replay buffer size

# Initialize Environment
def initialize_environment():
    interference_power = np.random.uniform(1, 5, N)
    jammer_power = np.random.uniform(1, 5, N)
    channel_gain_signal = 0.8  # Channel power gain for signal
    channel_gain_jammer = 0.7  # Channel power gain for jammer
    channel_gain_interference = np.random.uniform(0.4, 0.9, N)
    return (interference_power, jammer_power, channel_gain_signal,
            channel_gain_jammer, channel_gain_interference)

# Multi-Objective Reward Function
def communication_feedback(sinr, interference_power, jammer_power, communication_action):
    sinr_reward = lambda_val * max(0, sinr - mu_th)
    interference_penalty = -0.1 * interference_power[communication_action]
    jammer_penalty = -0.1 * jammer_power[communication_action]
    reward = (sinr_reward + interference_penalty + jammer_penalty) / lambda_val
    return np.clip(reward, -lambda_val, lambda_val)

def sensing_feedback(psd):
    return lambda_val if psd < nu_th else -lambda_val

# Create Indication Matrix
def create_indication_matrix(communication_action, sinr, sensing_actions, psds, interference_power, jammer_power):
    indication_matrix = []
    indication_matrix.append([communication_action,
                               communication_feedback(sinr, interference_power, jammer_power, communication_action),
                               interference_power[communication_action],
                               jammer_power[communication_action]])
    for action, psd in zip(sensing_actions, psds):
        indication_matrix.append([action, sensing_feedback(psd), interference_power[action], jammer_power[action]])
    return np.array(indication_matrix)

# Create State Matrix
def create_state_matrix(history):
    return np.stack(history[-T:], axis=-1)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define DDQN
class DDQN:
    def __init__(self, input_shape, num_actions):
        self.num_actions = num_actions
        self.q_network = self.build_network(input_shape, num_actions)
        self.target_network = self.build_network(input_shape, num_actions)
        self.update_target_network()
        self.losses = []

    def build_network(self, input_shape, num_actions):
        model = models.Sequential([
            layers.Conv2D(10, (2, 2), activation='relu', input_shape=input_shape),
            layers.AveragePooling2D(pool_size=(2, 1)),
            layers.Flatten(),
            layers.Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def predict(self, state):
        return self.q_network.predict(state[np.newaxis, ...])

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.q_network.predict(states)
        next_q_values = self.target_network.predict(next_states)
        for i in range(len(batch)):
            target = rewards[i] + (1 - dones[i]) * gamma * np.max(next_q_values[i])
            q_values[i][actions[i]] = target
        history = self.q_network.fit(states, q_values, verbose=0)
        self.losses.append(history.history['loss'][0])

# Action Selection
def select_actions(q_values, epsilon, action_space):
    if random.random() < epsilon:
        communication_action = random.choice(action_space)
        sensing_actions = random.sample(action_space, Ns)
    else:
        communication_action = np.argmax(q_values)
        sensing_actions = random.sample(action_space, Ns)
    return communication_action, sensing_actions

def simulate():
    global epsilon
    ddqn = DDQN((Ns + 1, 4, T), N)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    history = []
    accumulated_rewards = []
    total_reward = 0
    epsilon_history = []
    selected_channels = []  # Track chosen communication channels

    for t in range(int(2e3)):
        # Randomize PSD values per bin per iteration (with small jitter)
        base_psd_db = np.array([-54, -45, -48, -51, -50, -49, -56])
        jitter = np.random.normal(0, 1, N)
        psd_values_db = base_psd_db + jitter
        psd_values_mw = 10 ** (psd_values_db / 10)

        # Environment update
        interference_power, jammer_power, hs, hj, hi = initialize_environment()
        Ps = 5e-3
        sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
        sensing_actions = random.sample(list(range(N)), Ns)
        psds = psd_values_mw[sensing_actions]

        # Create indication matrix and append to history
        indication_matrix = create_indication_matrix(
            0, sinr, sensing_actions, psds, interference_power, jammer_power
        )
        history.append(indication_matrix)

        # Ensure sufficient history depth for DDQN state
        if len(history) >= T:
            state_matrix = create_state_matrix(history)

            # Get Q-values and choose actions using epsilon-greedy policy
            q_values = ddqn.predict(state_matrix)
            communication_action, sensing_actions = select_actions(q_values, epsilon, list(range(N)))
            selected_channels.append(communication_action)

            # Simulate next state
            next_psd_values_db = base_psd_db + np.random.normal(0, 1, N)
            next_psd_values_mw = 10 ** (next_psd_values_db / 10)
            next_psds = next_psd_values_mw[sensing_actions]
            next_sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
            next_indication_matrix = create_indication_matrix(
                communication_action, next_sinr, sensing_actions, next_psds,
                interference_power, jammer_power
            )
            next_state_matrix = create_state_matrix(history[-(T - 1):] + [next_indication_matrix])

            # Compute reward for selected communication action
            reward = communication_feedback(sinr, interference_power, jammer_power, communication_action)
            total_reward += reward
            accumulated_rewards.append(total_reward)

            # Store experience in replay buffer and train DDQN
            replay_buffer.add((state_matrix, communication_action, reward, next_state_matrix, t == 1999))
            if len(replay_buffer.buffer) >= batch_size:
                ddqn.train(replay_buffer.sample(batch_size))

        # Update target network every 100 steps
        if t % 100 == 0:
            ddqn.update_target_network()

        # Record and decay epsilon
        epsilon_history.append(epsilon)
        epsilon = max(epsilon_min, epsilon * decay_rate)

    # ---------- Visualization Section ----------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accumulated_rewards, label='Accumulated Reward', linewidth=1.8)
    plt.title('Accumulated Reward over Iterations')
    plt.xlabel('Iteration'); plt.ylabel('Cumulative Reward'); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ddqn.losses, color='r', label='Loss', linewidth=1.8)
    plt.title('DDQN Loss Convergence')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Epsilon decay curve
    plt.figure()
    plt.plot(epsilon_history, linewidth=2)
    plt.title('Exploration Rate Decay (ε)')
    plt.xlabel('Iteration'); plt.ylabel('Epsilon'); plt.grid(True)
    plt.show()

    # Histogram of chosen channels
    plt.figure()
    plt.hist(selected_channels, bins=N, rwidth=0.8)
    plt.title('Channel Selection Frequency')
    plt.xlabel('Channel Index'); plt.ylabel('Selection Count')
    plt.grid(True)
    plt.show()


simulate()
