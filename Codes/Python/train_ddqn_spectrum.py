"""
Main Training Script for DRL-Assisted Spectrum Selection

This script implements the complete training workflow for the DDQN agent
in the shared 5G-passive sensing spectrum environment.

Algorithmic Workflow:
1. Initialize environment (MATLAB channel + PSD sensing model)
2. Initialize DDQN networks and replay buffer
3. For each episode:
   a. Reset environment → s_0
   b. For each time step:
      - Select action using epsilon-greedy policy
      - Apply action in MATLAB PHY model
      - Compute reward (throughput - interference)
      - Observe new state
      - Store experience in replay buffer
      - Sample minibatch and update Q-network
      - Update target network periodically
   c. Decay epsilon after each episode
4. Evaluate performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

from ddqn_spectrum_selection import DDQNAgent
from spectrum_environment import SpectrumEnvironment


class DRLTrainer:
    """
    Trainer class for DDQN spectrum selection agent.
    
    Implements the complete training loop with evaluation metrics,
    visualization, and model checkpointing.
    """
    
    def __init__(
        self,
        env: SpectrumEnvironment,
        agent: DDQNAgent,
        num_episodes: int = 2000,
        max_steps_per_episode: int = 100,
        save_dir: str = './checkpoints',
        log_interval: int = 100
    ):
        """
        Initialize trainer.
        
        Args:
            env: Spectrum environment instance
            agent: DDQN agent instance
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_dir: Directory to save checkpoints and logs
            log_interval: Interval for logging statistics
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_throughput = []
        self.episode_interference = []
        self.episode_losses = []
        self.epsilon_history = []
        
        # Evaluation metrics
        self.avg_rewards = []
        self.avg_throughput = []
        self.avg_interference = []
        
    def train(self):
        """Execute training loop."""
        print("=" * 60)
        print("Starting DRL Training for Spectrum Selection")
        print("=" * 60)
        print(f"Episodes: {self.num_episodes}")
        print(f"Max steps per episode: {self.max_steps_per_episode}")
        print(f"Initial epsilon: {self.agent.epsilon}")
        print(f"Learning rate: {self.agent.learning_rate}")
        print(f"Discount factor: {self.agent.gamma}")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_interference = 0
            episode_losses = []
            
            for step in range(self.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Execute action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Update statistics
                episode_reward += reward
                episode_throughput += info['throughput']
                episode_interference += info['interference']
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            self.episode_throughput.append(episode_throughput / (step + 1))
            self.episode_interference.append(episode_interference / (step + 1))
            self.epsilon_history.append(self.agent.epsilon)
            
            if episode_losses:
                self.episode_losses.append(np.mean(episode_losses))
            
            # Compute moving averages
            window = min(100, episode + 1)
            self.avg_rewards.append(np.mean(self.episode_rewards[-window:]))
            self.avg_throughput.append(np.mean(self.episode_throughput[-window:]))
            self.avg_interference.append(np.mean(self.episode_interference[-window:]))
            
            # Logging
            if (episode + 1) % self.log_interval == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}")
                print(f"  Reward: {episode_reward:.4f} (avg: {self.avg_rewards[-1]:.4f})")
                print(f"  Throughput: {episode_throughput/(step+1):.4f} (avg: {self.avg_throughput[-1]:.4f})")
                print(f"  Interference: {episode_interference/(step+1):.4f} (avg: {self.avg_interference[-1]:.4f})")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                if episode_losses:
                    print(f"  Loss: {np.mean(episode_losses):.6f}")
                print("-" * 60)
            
            # Save checkpoint
            if (episode + 1) % 500 == 0:
                self.save_checkpoint(episode + 1)
        
        print("\nTraining completed!")
        self.save_checkpoint(self.num_episodes, final=True)
        self.plot_training_curves()
    
    def save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint and training statistics."""
        suffix = "_final" if final else f"_ep{episode}"
        
        # Save model
        model_path = os.path.join(self.save_dir, f"ddqn_model{suffix}.h5")
        self.agent.save_model(model_path)
        
        # Save training statistics
        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_throughput': self.episode_throughput,
            'episode_interference': self.episode_interference,
            'episode_losses': self.episode_losses,
            'epsilon_history': self.epsilon_history,
            'avg_rewards': self.avg_rewards,
            'avg_throughput': self.avg_throughput,
            'avg_interference': self.avg_interference
        }
        
        stats_path = os.path.join(self.save_dir, f"training_stats{suffix}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Checkpoint saved: {model_path}")
    
    def plot_training_curves(self):
        """Plot training curves and save figures."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(self.avg_rewards, label='Moving Average (100)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Throughput
        axes[0, 1].plot(self.episode_throughput, alpha=0.3, label='Episode Throughput')
        axes[0, 1].plot(self.avg_throughput, label='Moving Average (100)', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Throughput (bits/s/Hz)')
        axes[0, 1].set_title('Average Throughput per Episode')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Interference
        axes[0, 2].plot(self.episode_interference, alpha=0.3, label='Episode Interference')
        axes[0, 2].plot(self.avg_interference, label='Moving Average (100)', linewidth=2)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Interference Power')
        axes[0, 2].set_title('Average Interference per Episode')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Loss
        if self.episode_losses:
            axes[1, 0].plot(self.episode_losses, linewidth=1.5)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('DDQN Loss Convergence')
            axes[1, 0].grid(True)
        
        # Epsilon decay
        axes[1, 1].plot(self.epsilon_history, linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate Decay')
        axes[1, 1].grid(True)
        
        # Episode lengths
        axes[1, 2].plot(self.episode_lengths, alpha=0.5)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].set_title('Episode Lengths')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {fig_path}")
        plt.show()
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_throughput = []
        eval_interference = []
        selected_subbands = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_interference = 0
            
            for step in range(self.max_steps_per_episode):
                # Select action (no exploration during evaluation)
                action = self.agent.select_action(state, training=False)
                selected_subbands.append(action)
                
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_throughput += info['throughput']
                episode_interference += info['interference']
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_throughput.append(episode_throughput / (step + 1))
            eval_interference.append(episode_interference / (step + 1))
        
        # Compute metrics
        metrics = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_throughput': np.mean(eval_throughput),
            'std_throughput': np.std(eval_throughput),
            'avg_interference': np.mean(eval_interference),
            'std_interference': np.std(eval_interference),
            'subband_selection_distribution': np.bincount(selected_subbands, minlength=self.env.num_subbands).tolist()
        }
        
        print("\nEvaluation Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"  Average Throughput: {metrics['avg_throughput']:.4f} ± {metrics['std_throughput']:.4f}")
        print(f"  Average Interference: {metrics['avg_interference']:.4f} ± {metrics['std_interference']:.4f}")
        print(f"  Sub-band Selection Distribution: {metrics['subband_selection_distribution']}")
        
        return metrics


def main():
    """Main training function."""
    # Environment parameters
    num_subbands = 7
    transmit_power = 5e-3  # 5 mW
    noise_power = 1e-3
    
    # Reward function parameters
    alpha = 1.0  # Throughput weight
    beta = 1.0   # Interference weight
    
    # Create environment
    env = SpectrumEnvironment(
        num_subbands=num_subbands,
        transmit_power=transmit_power,
        noise_power=noise_power,
        channel_type='HFS',
        alpha=alpha,
        beta=beta,
        use_matlab=False  # Set to True if MATLAB engine is available
    )
    
    # Agent parameters
    state_shape = (num_subbands, 3)  # (PSD, CQI, interference_ratio)
    num_actions = num_subbands
    
    # Create agent
    agent = DDQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_freq=100,
        memory_size=10000,
        batch_size=32
    )
    
    # Training parameters
    num_episodes = 2000
    max_steps_per_episode = 100
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./checkpoints/ddqn_training_{timestamp}'
    
    trainer = DRLTrainer(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        save_dir=save_dir,
        log_interval=100
    )
    
    # Train agent
    trainer.train()
    
    # Evaluate agent
    metrics = trainer.evaluate(num_episodes=10)
    
    # Save evaluation metrics
    eval_path = os.path.join(save_dir, 'evaluation_metrics.json')
    with open(eval_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved to {eval_path}")


if __name__ == "__main__":
    main()
