"""
Performance Metrics and Evaluation Module for SWIFT II
Implements metrics as specified in the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import json


class PerformanceMetrics:
    """
    Performance metrics tracker for DRL-based spectrum selection
    
    Tracks and computes:
    - Average throughput improvement (ΔR)
    - Interference power reduction (ΔP_int)
    - Learning stability (σ²_Q)
    - Convergence rate
    """
    
    def __init__(self, save_dir: str = "./results"):
        """
        Initialize performance metrics tracker
        
        Args:
            save_dir: Directory to save results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_throughputs = []
        self.episode_interferences = []
        self.episode_sinr = []
        self.episode_losses = []
        self.q_values = []
        self.epsilon_values = []
        
        # Per-step metrics
        self.step_rewards = []
        self.step_actions = []
        self.step_sinr = []
        
        # Baseline comparison
        self.baseline_throughput = None
        self.baseline_interference = None
        
        # Convergence tracking
        self.convergence_episode = None
        self.convergence_threshold = 0.9  # 90% of stable reward
    
    def add_episode_metrics(
        self,
        episode: int,
        reward: float,
        throughput: float,
        interference: float,
        avg_sinr: float,
        avg_loss: float,
        epsilon: float,
        avg_q_value: float
    ):
        """
        Add metrics for a completed episode
        
        Args:
            episode: Episode number
            reward: Total episode reward
            throughput: Average throughput
            interference: Average interference power
            avg_sinr: Average SINR
            avg_loss: Average training loss
            epsilon: Current exploration rate
            avg_q_value: Average Q-value
        """
        self.episode_rewards.append(reward)
        self.episode_throughputs.append(throughput)
        self.episode_interferences.append(interference)
        self.episode_sinr.append(avg_sinr)
        self.episode_losses.append(avg_loss)
        self.epsilon_values.append(epsilon)
        self.q_values.append(avg_q_value)
        
        # Check for convergence
        if self.convergence_episode is None:
            self._check_convergence(episode)
    
    def add_step_metrics(
        self,
        step: int,
        action: int,
        reward: float,
        sinr: float
    ):
        """
        Add metrics for a single step
        
        Args:
            step: Step number
            action: Action taken
            reward: Reward received
            sinr: SINR value
        """
        self.step_rewards.append(reward)
        self.step_actions.append(action)
        self.step_sinr.append(sinr)
    
    def set_baseline(self, throughput: float, interference: float):
        """
        Set baseline performance for comparison
        
        Args:
            throughput: Baseline throughput
            interference: Baseline interference
        """
        self.baseline_throughput = throughput
        self.baseline_interference = interference
    
    def compute_throughput_improvement(self) -> float:
        """
        Compute average throughput improvement
        
        ΔR = (R_DRL - R_baseline) / R_baseline × 100%
        
        Returns:
            Throughput improvement percentage
        """
        if self.baseline_throughput is None or len(self.episode_throughputs) == 0:
            return 0.0
        
        avg_drl_throughput = np.mean(self.episode_throughputs[-100:])
        improvement = (avg_drl_throughput - self.baseline_throughput) / self.baseline_throughput * 100
        
        return improvement
    
    def compute_interference_reduction(self) -> float:
        """
        Compute interference power reduction
        
        ΔP_int = (P_baseline - P_DRL) / P_baseline × 100%
        
        Returns:
            Interference reduction percentage
        """
        if self.baseline_interference is None or len(self.episode_interferences) == 0:
            return 0.0
        
        avg_drl_interference = np.mean(self.episode_interferences[-100:])
        reduction = (self.baseline_interference - avg_drl_interference) / self.baseline_interference * 100
        
        return reduction
    
    def compute_learning_stability(self) -> float:
        """
        Compute learning stability (Q-value variance)
        
        σ²_Q = Var(Q(s,a))
        
        Returns:
            Q-value variance
        """
        if len(self.q_values) < 10:
            return 0.0
        
        recent_q_values = self.q_values[-100:]
        variance = np.var(recent_q_values)
        
        return variance
    
    def _check_convergence(self, episode: int, window: int = 50):
        """
        Check if training has converged
        
        Convergence is defined as reaching 90% stable reward
        
        Args:
            episode: Current episode
            window: Window size for stability check
        """
        if len(self.episode_rewards) < window * 2:
            return
        
        recent_rewards = self.episode_rewards[-window:]
        earlier_rewards = self.episode_rewards[-window*2:-window]
        
        recent_mean = np.mean(recent_rewards)
        earlier_mean = np.mean(earlier_rewards)
        
        # Check if improvement is less than 10%
        if abs(recent_mean - earlier_mean) / (abs(earlier_mean) + 1e-6) < 0.1:
            if self.convergence_episode is None:
                self.convergence_episode = episode
    
    def get_convergence_rate(self) -> Optional[int]:
        """
        Get convergence rate (episode number at convergence)
        
        Returns:
            Episode number where convergence occurred, or None
        """
        return self.convergence_episode
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics
        
        Returns:
            Dictionary of summary statistics
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'final_reward': self.episode_rewards[-1],
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]),
            'std_reward_last_100': np.std(self.episode_rewards[-100:]),
            'max_reward': np.max(self.episode_rewards),
            'avg_throughput': np.mean(self.episode_throughputs[-100:]),
            'avg_interference': np.mean(self.episode_interferences[-100:]),
            'avg_sinr_db': np.mean(self.episode_sinr[-100:]),
            'final_epsilon': self.epsilon_values[-1] if self.epsilon_values else 0.0,
            'convergence_episode': self.convergence_episode,
            'throughput_improvement_pct': self.compute_throughput_improvement(),
            'interference_reduction_pct': self.compute_interference_reduction(),
            'learning_stability': self.compute_learning_stability()
        }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary_statistics()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        if not summary:
            print("No data available")
            return
        
        print(f"\nTraining Statistics:")
        print(f"  Total Episodes: {summary['total_episodes']}")
        print(f"  Convergence Episode: {summary['convergence_episode'] or 'Not converged'}")
        print(f"  Final Epsilon: {summary['final_epsilon']:.4f}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Average Reward (last 100): {summary['avg_reward_last_100']:.2f} ± {summary['std_reward_last_100']:.2f}")
        print(f"  Max Reward: {summary['max_reward']:.2f}")
        print(f"  Average Throughput: {summary['avg_throughput']:.3f} bits/s/Hz")
        print(f"  Average Interference: {summary['avg_interference']:.3f} mW")
        print(f"  Average SINR: {summary['avg_sinr_db']:.2f} dB")
        
        print(f"\nComparison with Baseline:")
        print(f"  Throughput Improvement: {summary['throughput_improvement_pct']:.1f}%")
        print(f"  Interference Reduction: {summary['interference_reduction_pct']:.1f}%")
        print(f"  Learning Stability (σ²_Q): {summary['learning_stability']:.4f}")
        
        print("=" * 80 + "\n")
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save results to JSON file
        
        Args:
            filename: Optional filename, auto-generated if None
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        results = {
            'summary': self.get_summary_statistics(),
            'episode_rewards': self.episode_rewards,
            'episode_throughputs': self.episode_throughputs,
            'episode_interferences': self.episode_interferences,
            'episode_sinr': self.episode_sinr,
            'episode_losses': self.episode_losses,
            'epsilon_values': self.epsilon_values,
            'q_values': self.q_values
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def plot_training_curves(self, save: bool = True):
        """
        Plot training curves
        
        Args:
            save: Whether to save figures
        """
        if len(self.episode_rewards) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SWIFT II - DRL Training Performance', fontsize=16, fontweight='bold')
        
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        
        # 1. Accumulated Reward
        ax = axes[0, 0]
        ax.plot(episodes, np.cumsum(self.episode_rewards), linewidth=2, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Accumulated Reward over Episodes')
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Reward with moving average
        ax = axes[0, 1]
        ax.plot(episodes, self.episode_rewards, alpha=0.3, color='gray', label='Episode Reward')
        if len(self.episode_rewards) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            ax.plot(episodes[9:], moving_avg, linewidth=2, color='blue', label='Moving Avg (10)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. DDQN Loss Convergence
        ax = axes[0, 2]
        if len(self.episode_losses) > 0:
            ax.plot(episodes, self.episode_losses, linewidth=2, color='red')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('DDQN Loss Convergence')
            ax.grid(True, alpha=0.3)
        
        # 4. Exploration Rate Decay
        ax = axes[1, 0]
        ax.plot(episodes, self.epsilon_values, linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (ε)')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True, alpha=0.3)
        
        # 5. Average Throughput
        ax = axes[1, 1]
        ax.plot(episodes, self.episode_throughputs, linewidth=2, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Throughput (bits/s/Hz)')
        ax.set_title('Average Throughput per Episode')
        ax.grid(True, alpha=0.3)
        
        # 6. Average Interference
        ax = axes[1, 2]
        ax.plot(episodes, self.episode_interferences, linewidth=2, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Interference Power (mW)')
        ax.set_title('Average Interference per Episode')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"training_curves_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {filepath}")
        
        plt.show()
    
    def plot_action_distribution(self, num_channels: int = 7, save: bool = True):
        """
        Plot distribution of selected actions (channels)
        
        Args:
            num_channels: Number of channels
            save: Whether to save figure
        """
        if len(self.step_actions) == 0:
            print("No action data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.step_actions, bins=num_channels, rwidth=0.8, 
                 color='steelblue', edgecolor='black')
        plt.xlabel('Channel Index')
        plt.ylabel('Selection Count')
        plt.title('Channel Selection Frequency')
        plt.xticks(range(num_channels))
        plt.grid(True, alpha=0.3, axis='y')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"action_distribution_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Action distribution saved to: {filepath}")
        
        plt.show()
    
    def plot_comparison_with_baseline(
        self,
        baseline_name: str = "Random Selection",
        save: bool = True
    ):
        """
        Plot comparison with baseline method
        
        Args:
            baseline_name: Name of baseline method
            save: Whether to save figure
        """
        if self.baseline_throughput is None or self.baseline_interference is None:
            print("Baseline not set. Cannot plot comparison.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput comparison
        ax = axes[0]
        drl_throughput = np.mean(self.episode_throughputs[-100:])
        improvement = self.compute_throughput_improvement()
        
        methods = [baseline_name, 'DRL-DDQN']
        throughputs = [self.baseline_throughput, drl_throughput]
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax.bar(methods, throughputs, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Throughput (bits/s/Hz)')
        ax.set_title(f'Throughput Comparison\n(Improvement: {improvement:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Interference comparison
        ax = axes[1]
        drl_interference = np.mean(self.episode_interferences[-100:])
        reduction = self.compute_interference_reduction()
        
        interferences = [self.baseline_interference, drl_interference]
        
        bars = ax.bar(methods, interferences, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Interference Power (mW)')
        ax.set_title(f'Interference Comparison\n(Reduction: {reduction:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"baseline_comparison_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Baseline comparison saved to: {filepath}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing Performance Metrics Module...")
    
    metrics = PerformanceMetrics(save_dir="./test_results")
    
    # Simulate training data
    np.random.seed(42)
    for episode in range(200):
        reward = np.random.randn() * 10 + 50 + episode * 0.1
        throughput = 2.5 + episode * 0.01 + np.random.randn() * 0.1
        interference = 3.0 - episode * 0.005 + np.random.randn() * 0.1
        sinr_db = 10 + episode * 0.05 + np.random.randn() * 2
        loss = 1.0 / (1 + episode * 0.01) + np.random.randn() * 0.05
        epsilon = max(0.01, 1.0 * 0.995 ** episode)
        q_value = 100 + episode * 0.5 + np.random.randn() * 5
        
        metrics.add_episode_metrics(
            episode, reward, throughput, interference,
            sinr_db, loss, epsilon, q_value
        )
    
    # Set baseline
    metrics.set_baseline(throughput=2.0, interference=3.5)
    
    # Print summary
    metrics.print_summary()
    
    # Plot results
    metrics.plot_training_curves(save=False)
    metrics.plot_comparison_with_baseline(save=False)
    
    # Save results
    metrics.save_results()
