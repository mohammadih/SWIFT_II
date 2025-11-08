"""
Main Simulation Script for SWIFT II DRL-Assisted Spectrum Selection

This script implements the complete training and evaluation pipeline
as described in the paper.

Usage:
    python main.py [--config CONFIG_FILE] [--matlab] [--eval-only]
"""

import argparse
import numpy as np
import os
import sys
from datetime import datetime

# Import project modules
from config import ConfigManager
from environment import SpectrumEnvironment, MATLABPhyEnvironment
from ddqn_agent import DDQNAgent, DDQNTrainer
from metrics import PerformanceMetrics
from matlab_interface import create_matlab_interface


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SWIFT II - DRL-Assisted Spectrum Selection'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--matlab',
        action='store_true',
        help='Use MATLAB PHY layer (requires MATLAB Engine for Python)'
    )
    
    parser.add_argument(
        '--matlab-path',
        type=str,
        default='../Matlab',
        help='Path to MATLAB code directory'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Evaluation mode only (load existing model)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model weights for evaluation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting (useful for headless servers)'
    )
    
    return parser.parse_args()


def create_environment(config: ConfigManager, use_matlab: bool, matlab_path: str):
    """
    Create training environment
    
    Args:
        config: Configuration manager
        use_matlab: Whether to use MATLAB PHY layer
        matlab_path: Path to MATLAB code
    
    Returns:
        Environment instance
    """
    if use_matlab:
        print("Initializing MATLAB PHY environment...")
        matlab_interface = create_matlab_interface(
            use_real_matlab=True,
            matlab_path=matlab_path,
            timeout=config.simulation.matlab_timeout
        )
        env = MATLABPhyEnvironment(config, matlab_engine=matlab_interface)
    else:
        print("Using Python-based simulation environment...")
        env = SpectrumEnvironment(config)
    
    return env


def run_baseline_comparison(env, config: ConfigManager, num_episodes: int = 10):
    """
    Run baseline (random selection) for comparison
    
    Args:
        env: Environment instance
        config: Configuration manager
        num_episodes: Number of episodes for baseline
    
    Returns:
        Baseline throughput and interference
    """
    print("\nRunning baseline (random selection)...")
    
    total_throughput = 0.0
    total_interference = 0.0
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Random action selection
            action = np.random.randint(0, config.spectrum.num_channels)
            next_state, reward, done, info = env.step(action)
            
            total_throughput += info['throughput']
            total_interference += info['interference']
            total_steps += 1
            
            state = next_state
    
    avg_throughput = total_throughput / total_steps
    avg_interference = total_interference / total_steps
    
    print(f"Baseline Performance:")
    print(f"  Average Throughput: {avg_throughput:.3f} bits/s/Hz")
    print(f"  Average Interference: {avg_interference:.3f} mW")
    
    return avg_throughput, avg_interference


def train_agent(
    config: ConfigManager,
    env,
    baseline_throughput: float,
    baseline_interference: float,
    plot_enabled: bool = True
):
    """
    Train DDQN agent
    
    Args:
        config: Configuration manager
        env: Training environment
        baseline_throughput: Baseline throughput for comparison
        baseline_interference: Baseline interference for comparison
        plot_enabled: Whether to enable plotting
    
    Returns:
        Trained agent and metrics
    """
    print("\n" + "=" * 80)
    print("Training DDQN Agent")
    print("=" * 80)
    
    # Create agent
    agent = DDQNAgent(config)
    print(f"\nDDQN Network Architecture:")
    agent.q_network.summary()
    
    # Create trainer
    trainer = DDQNTrainer(agent, env, config)
    
    # Create metrics tracker
    metrics = PerformanceMetrics(save_dir=config.simulation.figure_save_directory)
    metrics.set_baseline(baseline_throughput, baseline_interference)
    
    # Train
    history = trainer.train()
    
    # Update metrics with training history
    for episode, (reward, throughput, interference, loss, epsilon, q_val) in enumerate(zip(
        history['episode_rewards'],
        history['episode_throughputs'],
        history['episode_interferences'],
        trainer.episode_avg_losses,
        history['epsilon_history'],
        history['q_values']
    )):
        # Calculate average SINR (approximation from throughput)
        avg_sinr_db = 10 * np.log10(2**throughput - 1 + 1e-10)
        
        metrics.add_episode_metrics(
            episode=episode,
            reward=reward,
            throughput=throughput,
            interference=interference,
            avg_sinr=avg_sinr_db,
            avg_loss=loss,
            epsilon=epsilon,
            avg_q_value=q_val
        )
    
    # Print and save results
    metrics.print_summary()
    metrics.save_results()
    
    if plot_enabled:
        metrics.plot_training_curves(save=True)
        metrics.plot_comparison_with_baseline(save=True)
    
    return agent, metrics


def evaluate_agent(
    agent: DDQNAgent,
    env,
    config: ConfigManager,
    num_episodes: int = 50,
    plot_enabled: bool = True
):
    """
    Evaluate trained agent
    
    Args:
        agent: Trained DDQN agent
        env: Evaluation environment
        config: Configuration manager
        num_episodes: Number of evaluation episodes
        plot_enabled: Whether to enable plotting
    
    Returns:
        Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating Agent")
    print("=" * 80)
    
    metrics = PerformanceMetrics(save_dir=config.simulation.figure_save_directory)
    
    total_reward = 0.0
    total_throughput = 0.0
    total_interference = 0.0
    all_actions = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_interference = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Greedy action selection (no exploration)
            action = agent.select_action(state, training=False)
            all_actions.append(action)
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info['throughput']
            episode_interference += info['interference']
            episode_steps += 1
            
            metrics.add_step_metrics(
                step=episode_steps,
                action=action,
                reward=reward,
                sinr=info['sinr']
            )
            
            state = next_state
        
        # Average over episode
        episode_throughput /= episode_steps
        episode_interference /= episode_steps
        avg_sinr_db = 10 * np.log10(2**episode_throughput - 1 + 1e-10)
        
        metrics.add_episode_metrics(
            episode=episode,
            reward=episode_reward,
            throughput=episode_throughput,
            interference=episode_interference,
            avg_sinr=avg_sinr_db,
            avg_loss=0.0,
            epsilon=0.0,
            avg_q_value=0.0
        )
        
        total_reward += episode_reward
        total_throughput += episode_throughput
        total_interference += episode_interference
    
    # Print evaluation results
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {total_reward/num_episodes:.2f}")
    print(f"  Average Throughput: {total_throughput/num_episodes:.3f} bits/s/Hz")
    print(f"  Average Interference: {total_interference/num_episodes:.3f} mW")
    
    # Action distribution
    action_counts = np.bincount(all_actions, minlength=config.spectrum.num_channels)
    print(f"\nChannel Selection Distribution:")
    for i, count in enumerate(action_counts):
        percentage = (count / len(all_actions)) * 100
        print(f"  Channel {i}: {count} times ({percentage:.1f}%)")
    
    if plot_enabled:
        metrics.plot_action_distribution(
            num_channels=config.spectrum.num_channels,
            save=True
        )
    
    return metrics


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load or create configuration
    if args.config is not None:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ConfigManager.from_dict(config_dict)
    else:
        config = ConfigManager()
    
    # Override configuration with command line arguments
    if args.episodes is not None:
        config.simulation.num_episodes = args.episodes
    
    if args.matlab:
        config.simulation.use_matlab_phy = True
    
    config.simulation.random_seed = args.seed
    
    # Set random seed
    config.set_random_seed()
    
    # Print configuration
    config.print_config()
    
    # Create environment
    env = create_environment(config, args.matlab, args.matlab_path)
    
    if args.eval_only:
        # Evaluation mode
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
            sys.exit(1)
        
        print(f"Loading model from: {args.model_path}")
        agent = DDQNAgent(config)
        agent.load_model(args.model_path)
        
        evaluate_agent(
            agent,
            env,
            config,
            num_episodes=config.simulation.num_eval_episodes,
            plot_enabled=not args.no_plot
        )
    else:
        # Training mode
        
        # Run baseline for comparison
        baseline_throughput, baseline_interference = run_baseline_comparison(
            env,
            config,
            num_episodes=10
        )
        
        # Train agent
        agent, metrics = train_agent(
            config,
            env,
            baseline_throughput,
            baseline_interference,
            plot_enabled=not args.no_plot
        )
        
        # Final evaluation
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        
        eval_metrics = evaluate_agent(
            agent,
            env,
            config,
            num_episodes=config.simulation.num_eval_episodes,
            plot_enabled=not args.no_plot
        )
    
    print("\n" + "=" * 80)
    print("Simulation Complete!")
    print("=" * 80)
    print(f"Results saved to: {config.simulation.figure_save_directory}")
    print(f"Models saved to: {config.simulation.model_save_directory}")


if __name__ == "__main__":
    main()
