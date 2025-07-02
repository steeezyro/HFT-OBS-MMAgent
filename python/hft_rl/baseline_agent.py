"""Baseline market making agent implementation."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .env import HFTTradingEnv


class BaselineMarketMaker:
    """
    Baseline market making agent with simple heuristics.
    
    Strategy:
    - Maintain ±2 tick skew around mid price
    - Widen spreads when inventory is high
    - Reduce position size when inventory exceeds thresholds
    """
    
    def __init__(
        self,
        base_spread_ticks: int = 2,
        max_inventory: int = 1000,
        inventory_threshold: int = 100,
        spread_multiplier: float = 1.5,
        position_sizing: float = 1.0,
    ):
        self.base_spread_ticks = base_spread_ticks
        self.max_inventory = max_inventory
        self.inventory_threshold = inventory_threshold
        self.spread_multiplier = spread_multiplier
        self.position_sizing = position_sizing
        
        # Trading history
        self.trade_history: List[Dict] = []
        self.pnl_history: List[float] = []
        self.inventory_history: List[int] = []
        self.timestamp_history: List[float] = []
        
        # Performance metrics
        self.total_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
    
    def get_action(self, observation: np.ndarray, info: Dict) -> np.ndarray:
        """
        Generate trading action based on current market state.
        
        Args:
            observation: Environment observation vector
            info: Environment info dictionary
            
        Returns:
            Action array [bid_offset, ask_offset] in ticks
        """
        inventory = info.get('inventory', 0)
        mid_price = info.get('mid_price', 0)
        spread = info.get('spread', 0.01)
        volatility = info.get('volatility', 0.001)
        
        # Base spread calculation
        base_spread = self.base_spread_ticks
        
        # Adjust spread based on inventory
        inventory_ratio = abs(inventory) / self.max_inventory
        if abs(inventory) > self.inventory_threshold:
            spread_adjustment = self.spread_multiplier * inventory_ratio
            base_spread = int(base_spread * spread_adjustment)
        
        # Adjust spread based on volatility
        vol_adjustment = max(0.5, min(2.0, volatility / 0.001))
        base_spread = int(base_spread * vol_adjustment)
        
        # Inventory bias - skew quotes to reduce inventory
        inventory_skew = 0
        if abs(inventory) > self.inventory_threshold / 2:
            skew_intensity = min(2.0, inventory_ratio * 3)
            inventory_skew = int(np.sign(inventory) * skew_intensity)
        
        # Calculate final offsets
        bid_offset = -base_spread - inventory_skew
        ask_offset = base_spread - inventory_skew
        
        # Position sizing - reduce size when inventory is high
        if abs(inventory) > self.inventory_threshold:
            # Reduce aggressiveness
            bid_offset -= 1
            ask_offset += 1
        
        # Ensure we don't cross the spread
        bid_offset = min(bid_offset, -1)
        ask_offset = max(ask_offset, 1)
        
        # Clip to action space bounds
        bid_offset = np.clip(bid_offset, -5, 5)
        ask_offset = np.clip(ask_offset, -5, 5)
        
        return np.array([bid_offset, ask_offset], dtype=np.float32)
    
    def update_history(self, info: Dict) -> None:
        """Update trading history with current state."""
        current_time = time.time()
        
        self.timestamp_history.append(current_time)
        self.pnl_history.append(info.get('total_pnl', 0))
        self.inventory_history.append(info.get('inventory', 0))
        
        # Update performance metrics
        current_pnl = info.get('total_pnl', 0)
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
        
        drawdown = self.peak_pnl - current_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        self.total_pnl = current_pnl
        self.total_trades = info.get('total_trades', 0)
    
    def run_episode(
        self, 
        env: HFTTradingEnv, 
        max_steps: int = 1000,
        render: bool = False
    ) -> Dict:
        """
        Run a complete trading episode.
        
        Args:
            env: Trading environment
            max_steps: Maximum number of steps
            render: Whether to render environment
            
        Returns:
            Episode summary statistics
        """
        observation, info = env.reset()
        
        episode_reward = 0.0
        episode_trades = 0
        start_time = time.time()
        
        for step in range(max_steps):
            # Get action from strategy
            action = self.get_action(observation, info)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            episode_trades = info.get('total_trades', 0)
            self.update_history(info)
            
            if render and step % 100 == 0:
                env.render()
            
            if terminated or truncated:
                break
        
        end_time = time.time()
        episode_duration = end_time - start_time
        
        # Calculate episode statistics
        final_pnl = info.get('total_pnl', 0)
        final_inventory = info.get('inventory', 0)
        
        return {
            'episode_reward': episode_reward,
            'final_pnl': final_pnl,
            'final_inventory': final_inventory,
            'total_trades': episode_trades,
            'duration': episode_duration,
            'steps': step + 1,
            'max_drawdown': self.max_drawdown,
            'trades_per_second': episode_trades / max(episode_duration, 1),
        }
    
    def save_blotter(self, filepath: str) -> None:
        """Save trading blotter to Parquet file."""
        if not self.timestamp_history:
            print("No trading history to save")
            return
        
        df = pd.DataFrame({
            'timestamp': self.timestamp_history,
            'pnl': self.pnl_history,
            'inventory': self.inventory_history,
        })
        
        # Add derived metrics
        df['returns'] = df['pnl'].pct_change()
        df['cumulative_pnl'] = df['pnl'].cummax()
        df['drawdown'] = df['cumulative_pnl'] - df['pnl']
        df['inventory_abs'] = df['inventory'].abs()
        
        # Save to Parquet
        df.to_parquet(filepath, index=False)
        print(f"Blotter saved to {filepath}")
    
    def generate_equity_curve(self, filepath: str) -> None:
        """Generate and save equity curve plot."""
        if not self.pnl_history:
            print("No PnL history available")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('PnL Curve', 'Inventory', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        timestamps = pd.to_datetime(self.timestamp_history, unit='s')
        
        # PnL curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.pnl_history,
                mode='lines',
                name='PnL',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Inventory
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.inventory_history,
                mode='lines',
                name='Inventory',
                line=dict(color='orange', width=1.5)
            ),
            row=2, col=1
        )
        
        # Drawdown
        cummax_pnl = np.maximum.accumulate(self.pnl_history)
        drawdown = cummax_pnl - np.array(self.pnl_history)
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=-drawdown,  # Negative for visual effect
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Baseline Market Maker Performance<br>'
                  f'Total PnL: ${self.total_pnl:.2f} | '
                  f'Max DD: ${self.max_drawdown:.2f} | '
                  f'Trades: {self.total_trades}',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Shares", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown ($)", row=3, col=1)
        
        # Save plot
        fig.write_html(filepath)
        print(f"Equity curve saved to {filepath}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.pnl_history:
            return {}
        
        pnl_series = pd.Series(self.pnl_history)
        returns = pnl_series.pct_change().dropna()
        
        # Basic metrics
        total_return = pnl_series.iloc[-1] - pnl_series.iloc[0]
        
        # Risk metrics
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_dd = self.max_drawdown
        
        if len(returns) > 1:
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252 * 24 * 60)  # Annualized
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    sortino_ratio = mean_return / downside_std * np.sqrt(252 * 24 * 60)
        
        # Trading metrics
        avg_inventory = np.mean(np.abs(self.inventory_history)) if self.inventory_history else 0
        max_inventory = np.max(np.abs(self.inventory_history)) if self.inventory_history else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd,
            'total_trades': self.total_trades,
            'avg_inventory': avg_inventory,
            'max_inventory': max_inventory,
            'final_pnl': self.total_pnl,
        }