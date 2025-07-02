"""Gymnasium environment for HFT market making."""

import struct
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import zmq
from gymnasium import spaces


class HFTTradingEnv(gym.Env):
    """
    High-frequency trading environment for market making.
    
    Observation space: 50-dimensional vector containing:
    - Best bid/ask prices and quantities (4 dims)
    - Level-2 order book data (20 dims)
    - Recent price movements (10 dims)
    - Inventory and PnL metrics (6 dims)
    - Technical indicators (10 dims)
    
    Action space: 2-dimensional continuous:
    - Bid offset in ticks [-5, 5]
    - Ask offset in ticks [-5, 5]
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        zmq_endpoint: str = "tcp://localhost:5555",
        max_inventory: int = 1000,
        inventory_penalty: float = 0.01,
        tick_size: float = 0.01,
        lot_size: int = 100,
        transaction_cost: float = 0.0010,  # 10 bps
        max_steps: int = 10000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # Environment parameters
        self.zmq_endpoint = zmq_endpoint
        self.max_inventory = max_inventory
        self.inventory_penalty = inventory_penalty
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
        # State tracking
        self.reset()
        
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Connect to ZMQ if not already connected
        if not hasattr(self, '_connected') or not self._connected:
            try:
                self.socket.connect(self.zmq_endpoint)
                self._connected = True
            except zmq.error.ZMQError as e:
                print(f"Failed to connect to ZMQ: {e}")
                self._connected = False
        
        # Reset state
        self.inventory = 0
        self.pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.step_count = 0
        self.last_mid_price = 150.0
        
        # Order book state
        self.best_bid = 0
        self.best_ask = 0
        self.bid_qty = 0
        self.ask_qty = 0
        self.bid_levels = 0
        self.ask_levels = 0
        
        # Historical data for features
        self.price_history = np.full(20, 150.0)
        self.volume_history = np.zeros(20)
        self.spread_history = np.full(20, 0.01)
        
        # Technical indicators
        self.ema_short = 150.0
        self.ema_long = 150.0
        self.volatility = 0.001
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Get latest market data
        market_data = self._get_market_data()
        if market_data is not None:
            self._update_market_state(market_data)
        
        # Execute trading action
        bid_offset, ask_offset = action
        reward = self._execute_action(bid_offset, ask_offset)
        
        # Check termination conditions
        terminated = (
            abs(self.inventory) > self.max_inventory or
            self.step_count >= self.max_steps
        )
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Step: {self.step_count}")
            print(f"Inventory: {self.inventory}")
            print(f"PnL: {self.pnl:.2f}")
            print(f"Unrealized PnL: {self.unrealized_pnl:.2f}")
            print(f"Best Bid: {self.best_bid/10000:.4f}")
            print(f"Best Ask: {self.best_ask/10000:.4f}")
            print(f"Mid Price: {self.last_mid_price:.4f}")
            print("-" * 40)
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
    
    def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Receive market data from ZMQ socket."""
        if not self._connected:
            return None
            
        try:
            # Receive packed book update
            message = self.socket.recv(zmq.NOBLOCK)
            
            # Unpack the data (matches PackedBookUpdate struct from C++)
            data = struct.unpack('<QqqllLLdd', message)
            
            return {
                'timestamp': data[0],
                'best_bid': data[1],
                'best_ask': data[2],
                'bid_qty': data[3],
                'ask_qty': data[4],
                'bid_levels': data[5],
                'ask_levels': data[6],
                'spread': data[7],
                'mid_price': data[8],
            }
        except zmq.Again:
            # No data available
            return None
        except (struct.error, zmq.error.ZMQError) as e:
            print(f"Error receiving market data: {e}")
            return None
    
    def _update_market_state(self, market_data: Dict[str, Any]) -> None:
        """Update internal market state from received data."""
        self.best_bid = market_data['best_bid']
        self.best_ask = market_data['best_ask']
        self.bid_qty = market_data['bid_qty']
        self.ask_qty = market_data['ask_qty']
        self.bid_levels = market_data['bid_levels']
        self.ask_levels = market_data['ask_levels']
        
        if market_data['mid_price'] > 0:
            # Update price history
            self.price_history[:-1] = self.price_history[1:]
            self.price_history[-1] = market_data['mid_price']
            self.last_mid_price = market_data['mid_price']
            
            # Update spread history
            self.spread_history[:-1] = self.spread_history[1:]
            self.spread_history[-1] = market_data['spread']
            
            # Update technical indicators
            alpha_short = 2.0 / (10 + 1)  # 10-period EMA
            alpha_long = 2.0 / (50 + 1)   # 50-period EMA
            
            self.ema_short = alpha_short * market_data['mid_price'] + (1 - alpha_short) * self.ema_short
            self.ema_long = alpha_long * market_data['mid_price'] + (1 - alpha_long) * self.ema_long
            
            # Update volatility (simple rolling standard deviation)
            self.volatility = np.std(self.price_history[-10:]) if len(self.price_history) >= 10 else 0.001
        
        # Update unrealized PnL
        if self.inventory != 0 and self.last_mid_price > 0:
            self.unrealized_pnl = self.inventory * self.last_mid_price
    
    def _execute_action(self, bid_offset: float, ask_offset: float) -> float:
        """Execute trading action and calculate reward."""
        if self.best_bid == 0 or self.best_ask == 0:
            return 0.0  # No market data available
        
        # Convert offsets to actual prices
        bid_price = self.best_bid + int(bid_offset * self.tick_size * 10000)
        ask_price = self.best_ask + int(ask_offset * self.tick_size * 10000)
        
        # Simulate order fills (simplified model)
        reward = 0.0
        
        # Probability of fill depends on how aggressive the quotes are
        bid_fill_prob = self._calculate_fill_probability(bid_price, self.best_bid, True)
        ask_fill_prob = self._calculate_fill_probability(ask_price, self.best_ask, False)
        
        # Execute fills
        if np.random.random() < bid_fill_prob and self.inventory < self.max_inventory:
            # Buy order filled
            fill_qty = self.lot_size
            trade_price = bid_price / 10000.0
            
            self.inventory += fill_qty
            self.pnl -= fill_qty * (trade_price + self.transaction_cost)
            self.total_trades += 1
            
            reward += 0.5  # Positive reward for providing liquidity
        
        if np.random.random() < ask_fill_prob and self.inventory > -self.max_inventory:
            # Sell order filled
            fill_qty = self.lot_size
            trade_price = ask_price / 10000.0
            
            self.inventory -= fill_qty
            self.pnl += fill_qty * (trade_price - self.transaction_cost)
            self.total_trades += 1
            
            reward += 0.5  # Positive reward for providing liquidity
        
        # Inventory penalty
        inventory_penalty = self.inventory_penalty * abs(self.inventory) / self.max_inventory
        reward -= inventory_penalty
        
        # PnL change reward
        current_total_pnl = self.pnl + self.unrealized_pnl
        pnl_change = current_total_pnl - getattr(self, '_last_total_pnl', 0)
        reward += pnl_change * 0.1  # Scale PnL reward
        self._last_total_pnl = current_total_pnl
        
        return reward
    
    def _calculate_fill_probability(self, quote_price: int, market_price: int, is_bid: bool) -> float:
        """Calculate probability of order fill based on price aggressiveness."""
        if market_price == 0:
            return 0.0
        
        if is_bid:
            # Higher bid prices are more likely to fill
            aggressiveness = (quote_price - market_price) / market_price
        else:
            # Lower ask prices are more likely to fill
            aggressiveness = (market_price - quote_price) / market_price
        
        # Convert to probability (0 to 1)
        base_prob = 0.01  # Base fill probability
        max_prob = 0.1    # Maximum fill probability
        
        prob = base_prob + (max_prob - base_prob) * max(0, min(1, aggressiveness * 100))
        return prob
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = np.zeros(50, dtype=np.float32)
        
        # Best bid/ask and quantities (4 dims)
        obs[0] = self.best_bid / 10000.0 if self.best_bid > 0 else 0
        obs[1] = self.best_ask / 10000.0 if self.best_ask > 0 else 0
        obs[2] = self.bid_qty / 1000.0  # Normalize quantity
        obs[3] = self.ask_qty / 1000.0
        
        # Order book depth (4 dims)
        obs[4] = self.bid_levels
        obs[5] = self.ask_levels
        obs[6] = (self.best_ask - self.best_bid) / 10000.0 if self.best_bid > 0 and self.best_ask > 0 else 0
        obs[7] = self.last_mid_price
        
        # Price history features (10 dims)
        returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
        obs[8:18] = np.nan_to_num(returns, 0.0)
        
        # Technical indicators (6 dims)
        obs[18] = self.ema_short
        obs[19] = self.ema_long
        obs[20] = self.ema_short - self.ema_long  # Signal line
        obs[21] = self.volatility
        obs[22] = np.mean(self.spread_history[-5:])  # Average spread
        obs[23] = np.std(self.spread_history[-5:])   # Spread volatility
        
        # Position and PnL metrics (6 dims)
        obs[24] = self.inventory / self.max_inventory  # Normalized inventory
        obs[25] = self.pnl / 10000.0  # Normalized PnL
        obs[26] = self.unrealized_pnl / 10000.0
        obs[27] = (self.pnl + self.unrealized_pnl) / 10000.0  # Total PnL
        obs[28] = self.total_trades / 1000.0  # Normalized trade count
        obs[29] = self.step_count / self.max_steps  # Time remaining
        
        # Market microstructure features (20 dims)
        # Volume imbalance
        total_qty = self.bid_qty + self.ask_qty
        obs[30] = (self.bid_qty - self.ask_qty) / max(total_qty, 1)
        
        # Price momentum (short-term)
        if len(self.price_history) >= 5:
            obs[31] = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
        
        # Volatility regimes
        recent_vol = np.std(self.price_history[-5:]) if len(self.price_history) >= 5 else 0
        obs[32] = recent_vol / (self.volatility + 1e-8)
        
        # Fill remaining features with derived metrics
        for i in range(33, 50):
            obs[i] = np.sin(i * self.step_count * 0.01)  # Synthetic time features
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary."""
        return {
            'inventory': self.inventory,
            'pnl': self.pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.pnl + self.unrealized_pnl,
            'total_trades': self.total_trades,
            'best_bid': self.best_bid / 10000.0 if self.best_bid > 0 else 0,
            'best_ask': self.best_ask / 10000.0 if self.best_ask > 0 else 0,
            'mid_price': self.last_mid_price,
            'spread': (self.best_ask - self.best_bid) / 10000.0 if self.best_bid > 0 and self.best_ask > 0 else 0,
            'volatility': self.volatility,
        }