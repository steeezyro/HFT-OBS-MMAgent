"""Test suite for HFT trading environment."""

import struct
import numpy as np
import pytest
import zmq
from unittest.mock import Mock, patch

from hft_rl.env import HFTTradingEnv


class TestHFTTradingEnv:
    """Test suite for the HFT trading environment."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = HFTTradingEnv(
            zmq_endpoint="tcp://localhost:5556",  # Different port for testing
            max_inventory=100,
            max_steps=1000,
            seed=42
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.observation_space.shape == (50,)
        assert self.env.action_space.shape == (2,)
        assert self.env.max_inventory == 100
        assert self.env.max_steps == 1000
    
    def test_reset(self):
        """Test environment reset."""
        observation, info = self.env.reset()
        
        assert observation.shape == (50,)
        assert isinstance(info, dict)
        assert self.env.inventory == 0
        assert self.env.pnl == 0.0
        assert self.env.step_count == 0
    
    def test_action_space_bounds(self):
        """Test action space bounds."""
        # Valid actions
        valid_action = np.array([2.5, -1.5])
        assert self.env.action_space.contains(valid_action)
        
        # Invalid actions (out of bounds)
        invalid_action = np.array([6.0, -6.0])
        assert not self.env.action_space.contains(invalid_action)
    
    def test_observation_space(self):
        """Test observation space."""
        observation, _ = self.env.reset()
        
        # Check observation is within bounds
        assert self.env.observation_space.contains(observation)
        
        # Check observation structure
        assert len(observation) == 50
        assert observation.dtype == np.float32
    
    @patch('zmq.Socket.recv')
    def test_market_data_processing(self, mock_recv):
        """Test market data processing."""
        # Mock market data
        mock_data = struct.pack('<QqqllLLdd', 
                               1640995200000000000,  # timestamp
                               150000,               # best_bid
                               150100,               # best_ask
                               1000,                 # bid_qty
                               1500,                 # ask_qty
                               5,                    # bid_levels
                               3,                    # ask_levels
                               0.01,                 # spread
                               150.05)               # mid_price
        
        mock_recv.return_value = mock_data
        
        observation, _ = self.env.reset()
        action = np.array([1.0, -1.0])
        
        # Step should process market data
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        assert self.env.best_bid == 150000
        assert self.env.best_ask == 150100
        assert self.env.last_mid_price == 150.05
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        observation, _ = self.env.reset()
        
        # Set some market state
        self.env.best_bid = 150000
        self.env.best_ask = 150100
        self.env.last_mid_price = 150.05
        
        action = np.array([1.0, -1.0])  # Moderately aggressive
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Reward should be a float
        assert isinstance(reward, (int, float))
        
        # Inventory penalty should be minimal at start
        assert abs(reward) < 10  # Reasonable reward range
    
    def test_inventory_management(self):
        """Test inventory tracking."""
        observation, _ = self.env.reset()
        
        # Simulate some trades by directly modifying inventory
        self.env.inventory = 50
        self.env.pnl = 100.0
        
        action = np.array([0.0, 0.0])  # Neutral action
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        assert info['inventory'] == 50
        assert info['pnl'] == 100.0
    
    def test_termination_conditions(self):
        """Test environment termination conditions."""
        observation, _ = self.env.reset()
        
        # Test inventory limit termination
        self.env.inventory = 150  # Exceeds max_inventory of 100
        
        action = np.array([0.0, 0.0])
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated
    
    def test_step_limit(self):
        """Test maximum step limit."""
        observation, _ = self.env.reset()
        
        # Run until step limit
        terminated = False
        truncated = False
        step_count = 0
        
        while not terminated and not truncated and step_count < self.env.max_steps + 10:
            action = np.array([0.0, 0.0])
            observation, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1
        
        assert step_count <= self.env.max_steps
        assert terminated or step_count == self.env.max_steps
    
    def test_fill_probability_calculation(self):
        """Test order fill probability calculation."""
        # Test bid fill probability
        market_price = 150000
        aggressive_price = 150050  # 50 ticks above market
        conservative_price = 149950  # 50 ticks below market
        
        aggressive_prob = self.env._calculate_fill_probability(
            aggressive_price, market_price, True
        )
        conservative_prob = self.env._calculate_fill_probability(
            conservative_price, market_price, True
        )
        
        # More aggressive orders should have higher fill probability
        assert aggressive_prob > conservative_prob
        assert 0 <= aggressive_prob <= 1
        assert 0 <= conservative_prob <= 1
    
    def test_observation_features(self):
        """Test observation feature extraction."""
        observation, _ = self.env.reset()
        
        # Set some market state
        self.env.best_bid = 150000
        self.env.best_ask = 150100
        self.env.bid_qty = 1000
        self.env.ask_qty = 1500
        self.env.inventory = 25
        self.env.pnl = 50.0
        
        obs = self.env._get_observation()
        
        # Check key features are present
        assert obs[0] == 15.0  # best_bid normalized
        assert obs[1] == 15.01  # best_ask normalized
        assert obs[2] == 1.0   # bid_qty normalized
        assert obs[3] == 1.5   # ask_qty normalized
        assert obs[24] == 0.25  # inventory normalized (25/100)
    
    def test_info_dictionary(self):
        """Test info dictionary content."""
        observation, info = self.env.reset()
        
        required_keys = [
            'inventory', 'pnl', 'unrealized_pnl', 'total_pnl',
            'total_trades', 'best_bid', 'best_ask', 'mid_price',
            'spread', 'volatility'
        ]
        
        for key in required_keys:
            assert key in info
            assert isinstance(info[key], (int, float))
    
    def test_seed_reproducibility(self):
        """Test that seeding produces reproducible results."""
        # Create two environments with the same seed
        env1 = HFTTradingEnv(seed=123, max_steps=10)
        env2 = HFTTradingEnv(seed=123, max_steps=10)
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Initial observations might not be identical due to market data,
        # but random components should be the same
        assert env1.inventory == env2.inventory
        assert env1.pnl == env2.pnl
        
        env1.close()
        env2.close()
    
    def test_render_modes(self):
        """Test different render modes."""
        # Test with human render mode
        env_render = HFTTradingEnv(render_mode="human", max_steps=10)
        observation, _ = env_render.reset()
        
        # Should not raise exception
        env_render.render()
        
        env_render.close()
    
    def test_close_cleanup(self):
        """Test environment cleanup on close."""
        env = HFTTradingEnv()
        env.close()
        
        # Should not raise exception when closing twice
        env.close()


class TestHFTTradingEnvIntegration:
    """Integration tests for the HFT trading environment."""
    
    def test_full_episode(self):
        """Test a complete episode."""
        env = HFTTradingEnv(max_steps=100, seed=42)
        
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert isinstance(total_reward, (int, float))
        
        env.close()
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        env = HFTTradingEnv(max_steps=50, seed=42)
        
        for episode in range(3):
            observation, info = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            assert isinstance(episode_reward, (int, float))
        
        env.close()


# Mock ZMQ for testing without actual socket connection
@pytest.fixture
def mock_zmq():
    """Mock ZMQ context and socket."""
    with patch('zmq.Context') as mock_context:
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket
        mock_socket.connect.return_value = None
        mock_socket.recv.side_effect = zmq.Again  # No data available
        yield mock_socket


def test_env_with_mock_zmq(mock_zmq):
    """Test environment with mocked ZMQ."""
    env = HFTTradingEnv(max_steps=10)
    
    observation, info = env.reset()
    action = np.array([1.0, -1.0])
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    assert observation.shape == (50,)
    assert isinstance(reward, (int, float))
    
    env.close()


if __name__ == '__main__':
    pytest.main([__file__])