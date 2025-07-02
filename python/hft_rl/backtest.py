"""Comprehensive backtesting framework for market making strategies."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats

warnings.filterwarnings('ignore')


@jit(nopython=True)
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series using Numba for speed."""
    returns = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        if prices[i] != 0:
            returns[i] = (prices[i + 1] - prices[i]) / prices[i]
    return returns


@jit(nopython=True)
def calculate_drawdown(pnl: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate drawdown series and maximum drawdown."""
    cummax = np.zeros_like(pnl)
    drawdown = np.zeros_like(pnl)
    
    cummax[0] = pnl[0]
    drawdown[0] = 0.0
    max_dd = 0.0
    
    for i in range(1, len(pnl)):
        cummax[i] = max(cummax[i-1], pnl[i])
        drawdown[i] = cummax[i] - pnl[i]
        max_dd = max(max_dd, drawdown[i])
    
    return drawdown, max_dd


@jit(nopython=True)
def calculate_turnover(positions: np.ndarray) -> float:
    """Calculate portfolio turnover."""
    total_turnover = 0.0
    for i in range(1, len(positions)):
        total_turnover += abs(positions[i] - positions[i-1])
    return total_turnover


class TransactionCostModel:
    """
    Advanced transaction cost model combining spread costs and market impact.
    
    Based on Almgren-Chriss model with additional fixed fee component.
    """
    
    def __init__(
        self,
        spread_cost: float = 0.0005,  # 5 bps spread cost
        impact_coeff: float = 0.1,    # Market impact coefficient
        fixed_fee: float = 0.001,     # 10 bps fixed fee
        temporary_impact: float = 0.5, # Temporary impact decay
    ):
        self.spread_cost = spread_cost
        self.impact_coeff = impact_coeff
        self.fixed_fee = fixed_fee
        self.temporary_impact = temporary_impact
    
    def calculate_cost(
        self, 
        trade_size: float, 
        price: float, 
        volume: float, 
        volatility: float
    ) -> float:
        """
        Calculate transaction cost for a trade.
        
        Args:
            trade_size: Size of the trade (signed)
            price: Current price
            volume: Average daily volume
            volatility: Price volatility
            
        Returns:
            Total transaction cost
        """
        notional = abs(trade_size * price)
        
        # Spread cost (always positive)
        spread_cost = notional * self.spread_cost
        
        # Market impact (Almgren-Chriss style)
        participation_rate = abs(trade_size) / max(volume, 1)
        impact_cost = (
            notional * 
            self.impact_coeff * 
            volatility * 
            np.sqrt(participation_rate)
        )
        
        # Fixed fee
        fixed_cost = notional * self.fixed_fee
        
        return spread_cost + impact_cost + fixed_cost


class RiskMetrics:
    """Risk metrics calculation utilities."""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - rf_rate) / np.std(returns)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < rf_rate]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return (np.mean(returns) - rf_rate) / np.std(downside_returns)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return np.inf if np.mean(returns) > 0 else 0.0
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    @staticmethod
    def var_cvar(returns: np.ndarray, confidence: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0, 0.0
        
        sorted_returns = np.sort(returns)
        var_index = int(confidence * len(sorted_returns))
        
        var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[-1]
        cvar = np.mean(sorted_returns[:var_index]) if var_index > 0 else var
        
        return var, cvar
    
    @staticmethod
    def tail_ratio(returns: np.ndarray, percentile: float = 0.95) -> float:
        """Calculate tail ratio (positive tail / negative tail)."""
        if len(returns) == 0:
            return 1.0
        
        upper_tail = np.percentile(returns, percentile * 100)
        lower_tail = np.percentile(returns, (1 - percentile) * 100)
        
        if lower_tail == 0:
            return np.inf if upper_tail > 0 else 1.0
        
        return abs(upper_tail / lower_tail)


class Backtester:
    """
    Comprehensive backtesting engine for market making strategies.
    
    Features:
    - Vectorized and event-driven execution modes
    - Advanced transaction cost modeling
    - Comprehensive risk metrics
    - Portfolio analytics
    - Performance attribution
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        risk_free_rate: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.risk_free_rate = risk_free_rate
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: pd.DataFrame = pd.DataFrame()
    
    def run_vectorized_backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 0.01,
        max_position: float = 0.1,
        rebalance_freq: str = '1min',
    ) -> Dict[str, Any]:
        """
        Run vectorized backtest for quick performance estimation.
        
        Args:
            price_data: DataFrame with OHLCV data
            signals: DataFrame with trading signals
            position_size: Base position size as fraction of capital
            max_position: Maximum position as fraction of capital
            rebalance_freq: Rebalancing frequency
            
        Returns:
            Dictionary with backtest results
        """
        # Align data
        aligned_data = pd.concat([price_data, signals], axis=1).fillna(method='ffill')
        
        # Calculate positions
        positions = self._calculate_positions(
            aligned_data, position_size, max_position, rebalance_freq
        )
        
        # Calculate returns and PnL
        returns = calculate_returns(aligned_data['close'].values)
        portfolio_returns = self._calculate_portfolio_returns(positions, returns)
        
        # Calculate transaction costs
        transaction_costs = self._calculate_transaction_costs_vectorized(
            positions, aligned_data
        )
        
        # Net returns after costs
        net_returns = portfolio_returns - transaction_costs
        cumulative_pnl = np.cumsum(net_returns) * self.initial_capital
        
        # Calculate metrics
        drawdown, max_drawdown = calculate_drawdown(cumulative_pnl)
        
        results = {
            'total_return': cumulative_pnl[-1] / self.initial_capital,
            'annualized_return': np.mean(net_returns) * 252,
            'volatility': np.std(net_returns) * np.sqrt(252),
            'sharpe_ratio': RiskMetrics.sharpe_ratio(net_returns, self.risk_free_rate/252),
            'sortino_ratio': RiskMetrics.sortino_ratio(net_returns, self.risk_free_rate/252),
            'max_drawdown': max_drawdown / self.initial_capital,
            'calmar_ratio': RiskMetrics.calmar_ratio(net_returns, max_drawdown / self.initial_capital),
            'turnover': calculate_turnover(positions) / len(positions),
            'transaction_costs': np.sum(transaction_costs) * self.initial_capital,
            'win_rate': np.sum(net_returns > 0) / len(net_returns),
            'profit_factor': np.sum(net_returns[net_returns > 0]) / abs(np.sum(net_returns[net_returns < 0])) if np.sum(net_returns < 0) != 0 else np.inf,
        }
        
        # Risk metrics
        var_95, cvar_95 = RiskMetrics.var_cvar(net_returns, 0.05)
        results.update({
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_ratio': RiskMetrics.tail_ratio(net_returns),
        })
        
        # Store results
        self.results = results
        self.portfolio_history = pd.DataFrame({
            'timestamp': aligned_data.index,
            'positions': positions,
            'pnl': cumulative_pnl,
            'returns': net_returns,
            'drawdown': drawdown,
        })
        
        return results
    
    def run_event_driven_backtest(
        self,
        market_data: pd.DataFrame,
        strategy_func: callable,
        initial_inventory: int = 0,
    ) -> Dict[str, Any]:
        """
        Run event-driven backtest for detailed trade-by-trade analysis.
        
        Args:
            market_data: DataFrame with market data
            strategy_func: Strategy function that takes (data, state) and returns action
            initial_inventory: Starting inventory
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize state
        state = {
            'inventory': initial_inventory,
            'cash': self.initial_capital,
            'pnl': 0.0,
            'trades': 0,
        }
        
        portfolio_history = []
        trades = []
        
        # Event loop
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            # Get strategy action
            action = strategy_func(row, state, i)
            
            if action is not None and action != 0:
                # Execute trade
                trade_result = self._execute_trade(
                    action, row['close'], row.get('volume', 1000), 
                    row.get('volatility', 0.01), timestamp, state
                )
                
                if trade_result:
                    trades.append(trade_result)
            
            # Update portfolio value
            portfolio_value = state['cash'] + state['inventory'] * row['close']
            unrealized_pnl = state['inventory'] * row['close']
            total_pnl = state['pnl'] + unrealized_pnl
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': state['cash'],
                'inventory': state['inventory'],
                'realized_pnl': state['pnl'],
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'price': row['close'],
            })
        
        # Convert to DataFrame
        self.portfolio_history = pd.DataFrame(portfolio_history)
        self.trades = trades
        
        # Calculate performance metrics
        returns = self.portfolio_history['total_pnl'].pct_change().dropna()
        results = self._calculate_performance_metrics(returns, self.portfolio_history)
        
        self.results = results
        return results
    
    def _calculate_positions(
        self, 
        data: pd.DataFrame, 
        position_size: float, 
        max_position: float,
        rebalance_freq: str
    ) -> np.ndarray:
        """Calculate position sizes from signals."""
        # Simple signal-based position sizing
        signals = data.get('signal', pd.Series(0, index=data.index))
        
        # Apply position sizing
        raw_positions = signals * position_size
        
        # Apply position limits
        positions = np.clip(raw_positions, -max_position, max_position)
        
        # Apply rebalancing frequency
        if rebalance_freq != '1min':
            rebalance_mask = data.index.to_series().dt.floor(rebalance_freq).diff() != pd.Timedelta(0)
            positions = positions.where(rebalance_mask, method='ffill')
        
        return positions.values
    
    def _calculate_portfolio_returns(
        self, positions: np.ndarray, asset_returns: np.ndarray
    ) -> np.ndarray:
        """Calculate portfolio returns from positions and asset returns."""
        # Lag positions by 1 to avoid look-ahead bias
        lagged_positions = np.concatenate([[0], positions[:-1]])
        return lagged_positions * asset_returns
    
    def _calculate_transaction_costs_vectorized(
        self, positions: np.ndarray, data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate transaction costs in vectorized manner."""
        # Calculate position changes (trades)
        trades = np.diff(np.concatenate([[0], positions]))
        prices = data['close'].values[1:]  # Align with trades
        volumes = data.get('volume', pd.Series(1000, index=data.index)).values[1:]
        volatilities = data.get('volatility', pd.Series(0.01, index=data.index)).values[1:]
        
        costs = np.zeros_like(trades)
        for i, trade in enumerate(trades):
            if trade != 0:
                costs[i] = self.transaction_cost_model.calculate_cost(
                    trade, prices[i], volumes[i], volatilities[i]
                ) / self.initial_capital  # As fraction of capital
        
        return costs
    
    def _execute_trade(
        self, 
        quantity: float, 
        price: float, 
        volume: float, 
        volatility: float,
        timestamp: pd.Timestamp,
        state: Dict
    ) -> Optional[Dict]:
        """Execute a single trade and update state."""
        if quantity == 0:
            return None
        
        # Calculate transaction cost
        cost = self.transaction_cost_model.calculate_cost(quantity, price, volume, volatility)
        
        # Check if we have enough cash (for buys) or inventory (for sells)
        total_cost = abs(quantity * price) + cost
        
        if quantity > 0 and state['cash'] < total_cost:
            return None  # Insufficient cash
        
        if quantity < 0 and state['inventory'] < abs(quantity):
            return None  # Insufficient inventory
        
        # Execute trade
        state['inventory'] += quantity
        state['cash'] -= quantity * price + cost
        state['trades'] += 1
        
        # Update realized PnL for sales
        if quantity < 0:  # Sale
            state['pnl'] += abs(quantity) * price - cost
        else:  # Purchase
            state['pnl'] -= cost  # Only subtract cost for purchases
        
        return {
            'timestamp': timestamp,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'inventory_after': state['inventory'],
            'cash_after': state['cash'],
            'pnl_after': state['pnl'],
        }
    
    def _calculate_performance_metrics(
        self, returns: pd.Series, portfolio_history: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = portfolio_history['total_pnl'].iloc[-1] / self.initial_capital
        annualized_return = returns.mean() * 252 * 24 * 60  # Assuming minute data
        volatility = returns.std() * np.sqrt(252 * 24 * 60)
        
        # Risk metrics
        drawdown_series = portfolio_history['total_pnl'].cummax() - portfolio_history['total_pnl']
        max_drawdown = drawdown_series.max() / self.initial_capital
        
        # Trade metrics
        num_trades = len(self.trades)
        avg_trade_size = np.mean([abs(t['quantity']) for t in self.trades]) if self.trades else 0
        
        # Inventory metrics
        inventory_series = portfolio_history['inventory']
        avg_inventory = inventory_series.abs().mean()
        max_inventory = inventory_series.abs().max()
        inventory_cvar = np.percentile(inventory_series.abs(), 95)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': RiskMetrics.sharpe_ratio(returns, self.risk_free_rate/(252*24*60)),
            'sortino_ratio': RiskMetrics.sortino_ratio(returns, self.risk_free_rate/(252*24*60)),
            'max_drawdown': max_drawdown,
            'calmar_ratio': RiskMetrics.calmar_ratio(returns, max_drawdown),
            'num_trades': num_trades,
            'avg_trade_size': avg_trade_size,
            'avg_inventory': avg_inventory,
            'max_inventory': max_inventory,
            'inventory_cvar': inventory_cvar,
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else np.inf,
        }
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate detailed performance report."""
        if self.portfolio_history.empty:
            return pd.DataFrame()
        
        # Daily aggregation
        daily_returns = self.portfolio_history.set_index('timestamp')['total_pnl'].resample('D').last().pct_change().dropna()
        
        # Monthly aggregation
        monthly_returns = self.portfolio_history.set_index('timestamp')['total_pnl'].resample('M').last().pct_change().dropna()
        
        # Performance by period
        report_data = []
        
        for period, returns in [('Daily', daily_returns), ('Monthly', monthly_returns)]:
            if len(returns) > 0:
                report_data.append({
                    'Period': period,
                    'Mean Return': returns.mean(),
                    'Std Return': returns.std(),
                    'Sharpe Ratio': RiskMetrics.sharpe_ratio(returns),
                    'Sortino Ratio': RiskMetrics.sortino_ratio(returns),
                    'Skewness': stats.skew(returns),
                    'Kurtosis': stats.kurtosis(returns),
                    'Max Return': returns.max(),
                    'Min Return': returns.min(),
                })
        
        return pd.DataFrame(report_data)
    
    def get_trade_analytics(self) -> Dict[str, Any]:
        """Get detailed trade analytics."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Trade size distribution
        trade_sizes = trades_df['quantity'].abs()
        
        # Trade timing analysis
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_by_hour = trades_df.groupby('hour').size()
        
        # PnL per trade
        pnl_per_trade = trades_df['pnl_after'].diff().dropna()
        
        return {
            'total_trades': len(self.trades),
            'avg_trade_size': trade_sizes.mean(),
            'median_trade_size': trade_sizes.median(),
            'trade_size_std': trade_sizes.std(),
            'largest_trade': trade_sizes.max(),
            'avg_pnl_per_trade': pnl_per_trade.mean(),
            'win_rate': (pnl_per_trade > 0).mean(),
            'avg_winning_trade': pnl_per_trade[pnl_per_trade > 0].mean(),
            'avg_losing_trade': pnl_per_trade[pnl_per_trade < 0].mean(),
            'best_trade': pnl_per_trade.max(),
            'worst_trade': pnl_per_trade.min(),
            'trades_by_hour': trades_by_hour.to_dict(),
        }