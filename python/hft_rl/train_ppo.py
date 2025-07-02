"""PPO training script for market making agent."""

import argparse
import os
import time
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import HFTTradingEnv


class WandbCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    wandb.log({
                        'train/episode_reward': episode_reward,
                        'train/episode_length': episode_length,
                        'train/mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                    })
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
        
        # Log model performance metrics
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                'train/learning_rate': self.model.learning_rate,
                'train/n_updates': self.model._n_updates,
            })
        
        return True


class TradingMetricsCallback(BaseCallback):
    """Callback for logging trading-specific metrics."""
    
    def __init__(self, eval_env: gym.Env, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_pnl = 0
            episode_trades = 0
            max_inventory = 0
            
            for _ in range(1000):  # Max 1000 steps for evaluation
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_pnl = info.get('total_pnl', 0)
                episode_trades = info.get('total_trades', 0)
                max_inventory = max(max_inventory, abs(info.get('inventory', 0)))
                
                if terminated or truncated:
                    break
            
            # Calculate Sharpe ratio approximation
            recent_rewards = getattr(self, 'recent_eval_rewards', [])
            recent_rewards.append(episode_reward)
            recent_rewards = recent_rewards[-20:]  # Keep last 20 evaluations
            self.recent_eval_rewards = recent_rewards
            
            sharpe_ratio = 0
            if len(recent_rewards) > 1:
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                sharpe_ratio = mean_reward / (std_reward + 1e-8)
            
            # Log metrics
            wandb.log({
                'eval/episode_reward': episode_reward,
                'eval/episode_pnl': episode_pnl,
                'eval/episode_trades': episode_trades,
                'eval/max_inventory': max_inventory,
                'eval/sharpe_ratio': sharpe_ratio,
                'eval/mean_reward': np.mean(recent_rewards),
            })
            
            # Save best model
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward
                self.model.save(os.path.join(wandb.run.dir, 'best_model'))
                
        return True


def create_env(zmq_endpoint: str = "tcp://localhost:5555", **kwargs) -> gym.Env:
    """Create and configure the trading environment."""
    env = HFTTradingEnv(zmq_endpoint=zmq_endpoint, **kwargs)
    env = Monitor(env)
    return env


def train_ppo_agent(
    total_timesteps: int = 10_000_000,
    learning_rate: float = 5e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    zmq_endpoint: str = "tcp://localhost:5555",
    model_save_path: str = "models/ppo_market_maker",
    wandb_project: str = "hft-rl",
    wandb_run_name: str = None,
    **env_kwargs
) -> PPO:
    """
    Train PPO agent for market making.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per update
        batch_size: Batch size for training
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        zmq_endpoint: ZMQ endpoint for market data
        model_save_path: Path to save trained model
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        **env_kwargs: Additional environment arguments
        
    Returns:
        Trained PPO model
    """
    
    # Initialize Weights & Biases
    config = {
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm,
        **env_kwargs
    }
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        tags=['ppo', 'market-making', 'hft']
    )
    
    # Create training environment
    print("Creating training environment...")
    train_env = make_vec_env(
        lambda: create_env(zmq_endpoint=zmq_endpoint, **env_kwargs),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Normalize observations and rewards
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        gamma=gamma
    )
    
    # Create evaluation environment
    eval_env = create_env(zmq_endpoint=zmq_endpoint, **env_kwargs)
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/",
        policy_kwargs={
            'net_arch': [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            'activation_fn': 'tanh',
        }
    )
    
    # Setup callbacks
    callbacks = [
        WandbCallback(),
        TradingMetricsCallback(eval_env, eval_freq=10000),
        CheckpointCallback(
            save_freq=50000,
            save_path='./checkpoints/',
            name_prefix='ppo_market_maker'
        )
    ]
    
    # Train the model
    print(f"Starting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save normalization statistics
        train_env.save(f"{model_save_path}_vecnormalize.pkl")
        
        # Final evaluation
        print("Running final evaluation...")
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_pnl = 0
        episode_trades = 0
        
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_pnl = info.get('total_pnl', 0)
            episode_trades = info.get('total_trades', 0)
            
            if terminated or truncated:
                break
        
        final_metrics = {
            'final_episode_reward': episode_reward,
            'final_episode_pnl': episode_pnl,
            'final_episode_trades': episode_trades,
            'training_time': training_time,
        }
        
        wandb.log(final_metrics)
        print(f"Final evaluation - Reward: {episode_reward:.2f}, PnL: {episode_pnl:.2f}, Trades: {episode_trades}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save(f"{model_save_path}_interrupted")
        
    finally:
        train_env.close()
        eval_env.close()
        wandb.finish()
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO market making agent')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=10_000_000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.999,
                       help='Discount factor')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    # Environment parameters
    parser.add_argument('--zmq-endpoint', type=str, default='tcp://localhost:5555',
                       help='ZMQ endpoint for market data')
    parser.add_argument('--max-inventory', type=int, default=1000,
                       help='Maximum inventory')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum steps per episode')
    
    # Logging parameters
    parser.add_argument('--wandb-project', type=str, default='hft-rl',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--model-save-path', type=str, default='models/ppo_market_maker',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Environment configuration
    env_kwargs = {
        'max_inventory': args.max_inventory,
        'max_steps': args.max_steps,
    }
    
    # Train the agent
    model = train_ppo_agent(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        zmq_endpoint=args.zmq_endpoint,
        model_save_path=args.model_save_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        **env_kwargs
    )
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()