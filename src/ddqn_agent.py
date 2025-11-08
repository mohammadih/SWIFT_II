import random
from collections import deque

import numpy as np
import tensorflow as tf

from .config import (
    LR,
    GAMMA,
    BATCH_SIZE,
    MEM_CAPACITY,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    N_SUBBANDS,
)


class DDQNAgent:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.n_actions = N_SUBBANDS

        self.online = self._build_network()
        self.target = self._build_network()
        self.target.set_weights(self.online.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(LR)
        self.memory = deque(maxlen=MEM_CAPACITY)

        self.epsilon = EPS_START
        self.step_count = 0

    def _build_network(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.state_dim,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(self.n_actions, activation=None),
            ]
        )

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.online(np.expand_dims(state, axis=0), training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon - EPS_DECAY)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        with tf.GradientTape() as tape:
            q_values = self.online(states)
            indices = np.arange(BATCH_SIZE)
            q_sa = tf.gather_nd(q_values, np.stack([indices, actions], axis=1))

            next_q_online = self.online(next_states)
            next_actions = tf.argmax(next_q_online, axis=1)
            next_q_target = self.target(next_states)
            next_q = tf.gather_nd(
                next_q_target,
                tf.stack([indices, tf.cast(next_actions, tf.int32)], axis=1),
            )

            targets = rewards + GAMMA * next_q * (1.0 - dones)
            loss = tf.reduce_mean(tf.square(targets - q_sa))

        grads = tape.gradient(loss, self.online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))

        self.step_count += 1
        self.update_epsilon()

    def update_target(self):
        self.target.set_weights(self.online.get_weights())
