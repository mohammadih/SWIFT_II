import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def create_indication_matrix(action_comm, g_comm, actions_sensing, f_sensing, Ns):
    """
    Create the indication matrix I(t) based on communication and sensing actions.
    
    Args:
    - action_comm: Communication action at time t-1.
    - g_comm: Reward (success/failure) from communication at t-1.
    - actions_sensing: List of sensing actions at t-1.
    - f_sensing: List of PSD results of sensed channels at t-1.
    - Ns: Number of sensing actions.
    
    Returns:
    - I_t: The indication matrix at time t.
    """
    I_t = np.zeros((Ns + 1, 2))
    
    # First row: Communication action and its reward
    I_t[0, 0] = action_comm
    I_t[0, 1] = g_comm
    
    # Remaining rows: Sensing actions and their availability results
    for i in range(1, Ns + 1):
        I_t[i, 0] = actions_sensing[i - 1]
        I_t[i, 1] = f_sensing[i - 1]
    
    return I_t

def build_q_network(state_shape, num_actions):
    """
    Build a CNN-based Q-network for Q-value estimation.
    
    Args:
    - state_shape: Shape of the input state (T, Ns + 1, 2).
    - num_actions: Number of possible actions (channels).
    
    Returns:
    - model: A compiled Q-network model.
    """
    model = models.Sequential()
    
    # Convolutional Layer
    model.add(layers.Conv2D(10, (2, 2), activation='relu', input_shape=state_shape))
    model.add(layers.AveragePooling2D(pool_size=(2, 1)))
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_actions))  # Output layer: Q-values for each action

    model.compile(optimizer='adam', loss='mse')
    return model

def dqn_update(state, action, reward, next_state, done, gamma, q_network):
    q_values = q_network.predict(state)
    next_q_values = q_network.predict(next_state)
    target_q = reward + (gamma * np.max(next_q_values)) if not done else reward
    q_values[0, action] = target_q
    loss = q_network.train_on_batch(state, q_values)
    return loss

def ddqn_update(state, action, reward, next_state, done, gamma, q_network, target_network):
    q_values = q_network.predict(state)
    next_q_values_online = q_network.predict(next_state)
    next_q_values_target = target_network.predict(next_state)
    best_action = np.argmax(next_q_values_online)
    target_q = reward + (gamma * next_q_values_target[0, best_action]) if not done else reward
    q_values[0, action] = target_q
    loss = q_network.train_on_batch(state, q_values)
    return loss

# Parameters from the paper
N = 6  # Number of channels (actions)
Ns = 2  # Number of sensing actions
T = 3  # Temporal memory depth
gamma = 0.3  # Discount factor
max_steps = 20000  # Max steps per episode
epsilon = 0.3  # Exploration rate
lambda_reward = 40  # Reward parameter

# Initialize Q-networks for DDQN
q_network = build_q_network((T, Ns + 1, 2), N)
target_network = build_q_network((T, Ns + 1, 2), N)
target_network.set_weights(q_network.get_weights())

# Function to simulate environment steps
def env_step(action):
    """Simulate taking an action in the environment."""
    next_state = np.random.random((1, T, Ns + 1, 2))  # Random next state
    reward = np.random.choice([lambda_reward, -lambda_reward])  # Random reward
    done = np.random.choice([True, False], p=[0.01, 0.99])  # Random termination
    return next_state, reward, done

# Track rewards for plotting
accumulated_rewards = []
cumulative_reward = 0

# Training Loop with Reward Tracking
for step in range(max_steps):
    state = np.random.random((1, T, Ns + 1, 2))  # Random initial state
    action = np.argmax(q_network.predict(state)) if np.random.random() > epsilon else np.random.randint(N)
    next_state, reward, done = env_step(action)

    # Choose between DQN and DDQN update
    if np.random.random() > 0.5:
        loss = dqn_update(state, action, reward, next_state, done, gamma, q_network)
    else:
        loss = ddqn_update(state, action, reward, next_state, done, gamma, q_network, target_network)

    # Update cumulative reward and store for plotting
    cumulative_reward += reward
    accumulated_rewards.append(cumulative_reward / ((step + 1) * lambda_reward))  # Normalize

    # Update target network periodically (for DDQN)
    if step % 100 == 0:
        target_network.set_weights(q_network.get_weights())

    if done:
        break

# Plot the Normalized Accumulated Reward (SINR)
plt.figure(figsize=(10, 5))
plt.plot(accumulated_rewards, label='Normalized Accumulated Reward (SINR)')
plt.xlabel('Steps')
plt.ylabel('Normalized Accumulated Reward')
plt.title('Normalized Accumulated Reward (SINR) over Time')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import matplotlib.pyplot as plt

# Step 1: Define Environment Parameters
N = 6  # Total number of channels
Ns = 2  # Number of sensing channels
T = 3  # Temporal memory depth
lambda_val = 10  # Reward weight
mu_th = 2  # SINR threshold
nu_th = 2  # PSD threshold
gamma = 0.4  # Discount factor
sigma_noise = 1e-3  # Noise power in mW
epsilon = 0.1  # Exploration rate
learning_rate = 0.1

# Define Environment Properties
def initialize_environment():
    interference_power = np.random.uniform(3, 6, N)
    jammer_power = np.random.uniform(3, 6, N)
    channel_gain_signal = 0.8  # Channel power gain for signal
    channel_gain_jammer = 0.7  # Channel power gain for jammer
    channel_gain_interference = np.random.uniform(0.4, 0.9, N)
    return interference_power, jammer_power, channel_gain_signal, channel_gain_jammer, channel_gain_interference

# Define Reward Feedback
def communication_feedback(sinr):
    return lambda_val if sinr > mu_th else -lambda_val

def sensing_feedback(psd):
    return lambda_val if psd < nu_th else -lambda_val

# Create Indication Matrix
def create_indication_matrix(communication_action, sinr, sensing_actions, psds):
    indication_matrix = []
    indication_matrix.append([communication_action, communication_feedback(sinr)])
    for action, psd in zip(sensing_actions, psds):
        indication_matrix.append([action, sensing_feedback(psd)])
    return np.array(indication_matrix)

# Create State Matrix
def create_state_matrix(history):
    return np.stack(history[-T:], axis=-1)

# Define the Double DQN
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
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def predict(self, state):
        return self.q_network.predict(state[np.newaxis, ...])

    def train(self, state, action, reward, next_state, done):
        q_values = self.q_network.predict(state[np.newaxis, ...])[0]
        next_q_values = self.target_network.predict(next_state[np.newaxis, ...])[0]
        target = reward + (1 - done) * gamma * np.max(next_q_values)
        q_values[action] = target
        history = self.q_network.fit(state[np.newaxis, ...], q_values[np.newaxis, ...], verbose=0)
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

# Simulation
def simulate():
    interference_power, jammer_power, hs, hj, hi = initialize_environment()
    ddqn = DDQN((3, 2, T), N)
    history = []
    accumulated_rewards = []
    total_reward = 0

    for t in range(10000):
        # Generate SINR and PSDs
        Ps = 5e-3  # Signal power in mW
        sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
        psds = np.random.uniform(0, 5, Ns)

        # Select actions
        if len(history) >= T:
            state_matrix = create_state_matrix(history)
            q_values = ddqn.predict(state_matrix)
            communication_action, sensing_actions = select_actions(q_values, epsilon, list(range(N)))
        else:
            communication_action = random.choice(range(N))
            sensing_actions = random.sample(range(N), Ns)

        # Generate indication matrix
        indication_matrix = create_indication_matrix(communication_action, sinr, sensing_actions, psds)
        history.append(indication_matrix)

        if len(history) >= T:
            state_matrix = create_state_matrix(history)

            # Simulate next state
            next_sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
            next_psds = np.random.uniform(0, 5, Ns)
            next_indication_matrix = create_indication_matrix(communication_action, next_sinr, sensing_actions, next_psds)
            next_state_matrix = create_state_matrix(history[-(T - 1):] + [next_indication_matrix])

            # Compute reward
            reward = communication_feedback(sinr)
            total_reward += reward
            accumulated_rewards.append(total_reward)

            # Train DDQN
            done = t == 9999
            ddqn.train(state_matrix, communication_action, reward, next_state_matrix, done)

        # Update target network periodically
        if t % 100 == 0:
            ddqn.update_target_network()

    # Plot Results
    plt.figure(figsize=(12, 5))

    # Plot Accumulated Reward
    plt.subplot(1, 2, 1)
    plt.plot(accumulated_rewards)
    plt.title('Accumulated Reward (SINR)')
    plt.xlabel('Iteration')
    plt.ylabel('Accumulated Reward')

    # Plot Loss Function
    plt.subplot(1, 2, 2)
    plt.plot(ddqn.losses)
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

simulate()

#%%
# revising the above code

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Define Environment Parameters:
N = 6 # Total number of channels
Ns = 2 # Number of sensing channels
T = 3 # Temporal memory depth
lambda_val = 10 # Reward weight
mu_th = 2 # SINR threshold
nu_th = 2 # PSD threshold
gamma = 0.4 # Discount factor
sigma_noise = 1e-3 # Noise power in mW
epsilon = 0.1 # Exploration rate
learning_rate = 0.1

def initialize_environment():
    interference_power = np.random.uniform(3, 6, N)
    jammer_power = np.random.uniform(3, 6 ,N)
    channel_gain_signal = 0.8 # Channel power gain for signal
    channel_gain_jammer = 0.7 # Channel power gain for jammer
    channel_gain_interference = np.random.uniform(0.4, 0.9, N)
    return (interference_power, jammer_power, channel_gain_signal,
             channel_gain_jammer, channel_gain_interference)

# Define Reward Feedback, it forces to choose the strong channel for 
# communication when forcing the agent choose the other weakest for sensing
def communication_feedback(sinr):
    return lambda_val if sinr > mu_th else -lambda_val

def sensing_feedback(psd):
    return lambda_val if psd < nu_th else -lambda_val

def create_indication_matrix(communication_action, sinr, sensing_actions,
                             psds):
    """
    Create the indication matrix I(t) based on communication and 
    sensing actions.
    
    Args:
        - action: Communication action at time t-1.
        - communication_action: Reward (success/failure) from communication at t-1.
        - action: List of sensing actions at t-1.
        - sensing_actions: List of PSD results of sensed channels at t-1.
        
    Returns:
        - I_t: The indication matrix at time t.
    """
    
    """Ns + 1: The number of rows in the array.
    The extra 1 accounts for the communication action,
    while the remaining rows correspond to sensing actions
    for the Ns channels.
    2: The number of columns in the array.
    It represents two pieces of information for each row
    This does not overwrite the first row because append()
    adds new rows to the end of the list. The communication action
    and its feedback remain as the first row.
    [
    [communication_action, communication_feedback(sinr)],  # Communication action
    [sensing_action_1,     sensing_feedback(psd_1)],       # Sensing action 1
    [sensing_action_2,     sensing_feedback(psd_2)],       # Sensing action 2
    ...
]
    zip(sensing_actions, psds) pairs each sensing action with its
    corresponding PSD value.

    """
    indication_matrix = []
    indication_matrix.append([communication_action,
                              communication_feedback(sinr)])
    for action, psd in zip(sensing_actions, psds):
        indication_matrix.append([action, sensing_feedback(psd)])
    return np.array(indication_matrix)
   
# Create State Matrix
"""stacks the selected T indication matrices along a new axis,
 creating a 3D array (tensor).
 First Dimension (N_s + 1).
 Second Dimension (2).
 Third Dimension (T).
 The slice [-T:] selects the most recent T elements from
 the history list.
 axis = -1 specifies that the new axis created by the operation
 should be added as the last dimension of the resulting array.
 axis = 0 adds the new dimension at the beginning.
 axis = 1 adds the new dimension after the first axis.
 State Matrix: (N_s + 1, 2, T)
 """
def create_state_matrix(history):
    return np.stack(history[-T:], axis = -1)

# Define the Double DQN
class DDQN:
    def __init__(self, input_shape, num_actions):
        self.num_actions = num_actions
        self.q_network = self.build_network(input_shape, num_actions)
        self.target_network = self.build.network(input_shape, num_actions)
        self.update_target_network()
        self.losses = []
    
    # num_actions in layers.Dense() means the number of output neurons.
    def build_network(self, input_shape, num_actions):
        model = models.Sequential([
            layers.Conv2D(10, [2,2], activation='relu', input_shape = input_shape),
            layers.AveragePooling2D(pool_size = (2,1)),
            layers.Flatten(),
            layers.Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer = tf.keras.optimizer.Adam(learning_rate = learning_rate),
                      loss = 'mse')
        return model
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
    def predict(self,state):
        return self.q_network.predict(state[np.newaxis, ...])
    
    def train(self, state, action, reward, next_state, done):
        q_values = self.q_network.predict(state[np.newaxis, ...])[0]
        next_q_values = self.target_network.predict(next_state[np.newaxis, ...])[0]
        target = reward + (1 - done)* gamma * np.max(next_q_values)
        q_values[action] = target
        history = self.q_network.fit(state[np.newaxis, ...], 
                                     q_values[np.newaxis, ...], verbose = 0)
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

# Simulation
def simulate():
    interference_power, jammer_power, hs, hj, hi = initialize_environment()
    ddqn = DDQN((3, 2, T), N)
    history = []
    accumulated_rewards = []
    total_reward = 0
    
    for t in range(1e4):
        #Generate SINR and PSDs
        Ps = 5e-3 # Signal power in mW
        sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
        psds = np.random.uniform(0, 5, Ns)
        
        # Select Actions
        if len(history) >= T:
            state_matrix = create_state_matrix(history)
            q_values = ddqn.predict(state_matrix)
            communication_action, sensing_actions = select_actions(q_values, epsilon, list(range(N)))
        else:
            communication_action = random.choice(range(N))
            sensing_actions = random.sample(range(N), Ns)
            
        # Generate indication matrix
        indication_matrix = create_indication_matrix(communication_action, sinr, sensing_actions, psds)
        history.append(indication_matrix)
        
        if len(history) >= T:
            state_matrix = create_state_matrix(history)
            
            # Simulate next state
            next_sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
            next_psds = np.random.uniform(0, 5, Ns)
            next_indication_matrix = create_indication_matrix(communication_action, next_sinr, sensing_actions, next_psds)
            next_state_matrix = create_state_matrix(history[-(T - 1):] + [next_indication_matrix])
            
            # Compute reward
            reward = communication_feedback(sinr)
            total_reward += reward
            accumulated_rewards.append(total_reward)
            
            # Train DDQN
            done = t = 9999
            ddqn.train(state_matrix, communication_action, reward, next_state_matrix, done)
           
        # Update target network periodically
        if t % 100 == 0:
            ddqn.update_target_network()
            
    # Plot Results

    plt.figure(figsize = (12, 5))

    # Plot Accumulated Reward
    plt.subplot(1,2,1)
    plt.plot(accumulated_rewards)
    plt.title('Accumulated Reward (SINR)')
    plt.xlabel('Iteration')
    plt.ylabel('Accumulated Reward')

    # Plot Loss Function
    plt.subplot(1, 2, 2)
    plt.plot(ddqn.losses)
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

simulate()

#%%
# Enhanced code by considering decay rate for exploration
# and channel Interference and jammer action plus multi-objective RL

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Define Environment Parameters:
N = 6  # Total number of channels
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
    #sinr_reward = lambda_val * (sinr - mu_th)  # Scaled reward
    sinr_reward = lambda_val * max(0, sinr - mu_th)  # Scaled reward
    interference_penalty = -0.1 * interference_power[communication_action]
    jammer_penalty = -0.1 * jammer_power[communication_action]
    #reward = sinr_reward + interference_penalty + jammer_penalty
    reward = (sinr_reward + interference_penalty + jammer_penalty) / lambda_val # reward clipping
    return np.clip(reward, -lambda_val, lambda_val)  # Clip reward values

def sensing_feedback(psd):
    return lambda_val if psd < nu_th else -lambda_val

# Create Indication Matrix with Channel-Specific Features
def create_indication_matrix(communication_action, sinr, sensing_actions, psds, interference_power, jammer_power):
    indication_matrix = []
    # Add communication action with SINR, interference, and jammer power
    indication_matrix.append([communication_action,
                               communication_feedback(sinr, interference_power, jammer_power, communication_action),
                               interference_power[communication_action],
                               jammer_power[communication_action]])
    # Add sensing actions with PSD, interference, and jammer power
    for action, psd in zip(sensing_actions, psds):
        indication_matrix.append([action, sensing_feedback(psd), interference_power[action], jammer_power[action]])
    return np.array(indication_matrix)

# Create State Matrix
def create_state_matrix(history):
    return np.stack(history[-T:], axis=-1)

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define the Double DQN
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

# Action Selection with Epsilon Decay
def select_actions(q_values, epsilon, action_space):
    if random.random() < epsilon:
        communication_action = random.choice(action_space)
        sensing_actions = random.sample(action_space, Ns)
    else:
        communication_action = np.argmax(q_values)
        sensing_actions = random.sample(action_space, Ns)

    return communication_action, sensing_actions

# Simulation

def simulate():
    interference_power, jammer_power, hs, hj, hi = initialize_environment()
    ddqn = DDQN((Ns + 1, 4, T), N)  # Updated input shape to include additional features
    replay_buffer = ReplayBuffer(replay_buffer_size)
    history = []
    accumulated_rewards = []
    total_reward = 0
    global epsilon

    for t in range(int(2e3)):
        # Generate SINR and PSDs
        Ps = 5e-3  # Signal power in mW
        sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
        psds = np.random.uniform(0, 5, Ns)

        # Select Actions
        if len(history) >= T:
            state_matrix = create_state_matrix(history)
            q_values = ddqn.predict(state_matrix)
            communication_action, sensing_actions = select_actions(q_values, epsilon, list(range(N)))
        else:
            communication_action = random.choice(range(N))
            sensing_actions = random.sample(range(N), Ns)

        # Generate indication matrix
        indication_matrix = create_indication_matrix(communication_action, sinr, sensing_actions, psds,
                                                     interference_power, jammer_power)
        history.append(indication_matrix)

        if len(history) >= T:
            state_matrix = create_state_matrix(history)

            # Simulate next state
            next_sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
            next_psds = np.random.uniform(0, 5, Ns)
            next_indication_matrix = create_indication_matrix(communication_action, next_sinr, sensing_actions,
                                                              next_psds, interference_power, jammer_power)
            next_state_matrix = create_state_matrix(history[-(T - 1):] + [next_indication_matrix])

            # Compute reward
            reward = communication_feedback(sinr, interference_power, jammer_power, communication_action)
            total_reward += reward
            accumulated_rewards.append(total_reward)

            # Store experience in replay buffer
            replay_buffer.add((state_matrix, communication_action, reward, next_state_matrix, t == 9999))

            # Train on minibatch from replay buffer
            if len(replay_buffer.buffer) >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                ddqn.train(minibatch)

        # Update target network periodically
        if t % 100 == 0:
            ddqn.update_target_network()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * decay_rate)

    # Plot Results
    plt.figure(figsize=(12, 5))

    # Plot Accumulated Reward
    plt.subplot(1, 2, 1)
    plt.plot(accumulated_rewards)
    plt.title('Accumulated Reward (SINR)')
    plt.xlabel('Iteration')
    plt.ylabel('Accumulated Reward')

    # Plot Loss Function
    plt.subplot(1, 2, 2)
    plt.plot(ddqn.losses)
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

simulate()

#%%
import numpy as np

# Load raw float32 data
file_path = r"J:\PC Shared-WD\Document\Datasets\1. fb_as_2RBG_8RBG_0dB_16QAM.dat"
data = np.fromfile(file_path, dtype=np.float32)

print("First few samples:" , data[20:30])


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

# Simulation
def simulate():
    global epsilon
    ddqn = DDQN((Ns + 1, 4, T), N)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    history = []
    accumulated_rewards = []
    total_reward = 0

    for t in range(int(2e3)):
        # Randomize PSD values per bin per iteration (with small jitter)
        base_psd_db = np.array([-54, -45, -48, -51, -50, -49, -56])
        jitter = np.random.normal(0, 1, N)  # Gaussian noise
        psd_values_db = base_psd_db + jitter
        psd_values_mw = 10 ** (psd_values_db / 10)

        # Select weakest channel (lowest PSD)
        weakest_channel_index = np.argmin(psd_values_mw)
        communication_action = weakest_channel_index

        # Update Environment
        interference_power, jammer_power, hs, hj, hi = initialize_environment()
        Ps = 5e-3
        sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
        sensing_actions = random.sample(list(range(N)), Ns)

        # Pick random PSDs for sensing channels from jittered set
        psds = psd_values_mw[sensing_actions]

        # Create indication matrix
        indication_matrix = create_indication_matrix(communication_action, sinr, sensing_actions, psds,
                                                     interference_power, jammer_power)
        history.append(indication_matrix)

        if len(history) >= T:
            state_matrix = create_state_matrix(history)

            # Simulate next state
            next_psd_values_db = base_psd_db + np.random.normal(0, 1, N)
            next_psd_values_mw = 10 ** (next_psd_values_db / 10)
            next_psds = next_psd_values_mw[sensing_actions]
            next_sinr = (hs * Ps) / (sigma_noise + np.sum(hi * interference_power) + hj * np.sum(jammer_power))
            next_indication_matrix = create_indication_matrix(communication_action, next_sinr, sensing_actions,
                                                              next_psds, interference_power, jammer_power)
            next_state_matrix = create_state_matrix(history[-(T - 1):] + [next_indication_matrix])

            # Compute reward
            reward = communication_feedback(sinr, interference_power, jammer_power, communication_action)
            total_reward += reward
            accumulated_rewards.append(total_reward)

            # Store and train
            replay_buffer.add((state_matrix, communication_action, reward, next_state_matrix, t == 1999))
            if len(replay_buffer.buffer) >= batch_size:
                ddqn.train(replay_buffer.sample(batch_size))

        # Update target periodically
        if t % 100 == 0:
            ddqn.update_target_network()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * decay_rate)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accumulated_rewards)
    plt.title('Accumulated Reward (SINR)')
    plt.xlabel('Iteration')
    plt.ylabel('Accumulated Reward')

    plt.subplot(1, 2, 2)
    plt.plot(ddqn.losses)
    plt.title('Loss Function')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

simulate()

            

    




