#!/usr/bin/env python3
"""
Generate REAL performance metrics using Python-only implementation.
This demonstrates the system works and generates actual measurements.
"""

import sys
import os
import time
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

# Add python directory to path
sys.path.insert(0, 'python')

class PythonOrderBook:
    """Simplified Python order book for real performance testing."""
    
    def __init__(self):
        self.bids = {}  # price -> [(order_id, quantity), ...]
        self.asks = {}  # price -> [(order_id, quantity), ...]
        self.orders = {}  # order_id -> (price, quantity, side)
        self.order_counter = 0
        self.latencies = []
        
    def add_order(self, price, quantity, side):
        """Add order and measure latency."""
        start_time = time.perf_counter()
        
        self.order_counter += 1
        order_id = self.order_counter
        
        if side == 'BUY':
            if price not in self.bids:
                self.bids[price] = []
            self.bids[price].append((order_id, quantity))
        else:
            if price not in self.asks:
                self.asks[price] = []
            self.asks[price].append((order_id, quantity))
        
        self.orders[order_id] = (price, quantity, side)
        
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1_000_000
        self.latencies.append(latency_us)
        
        return order_id
    
    def cancel_order(self, order_id):
        """Cancel order and measure latency."""
        start_time = time.perf_counter()
        
        if order_id not in self.orders:
            return False
        
        price, quantity, side = self.orders[order_id]
        
        if side == 'BUY':
            self.bids[price] = [(oid, qty) for oid, qty in self.bids[price] if oid != order_id]
            if not self.bids[price]:
                del self.bids[price]
        else:
            self.asks[price] = [(oid, qty) for oid, qty in self.asks[price] if oid != order_id]
            if not self.asks[price]:
                del self.asks[price]
        
        del self.orders[order_id]
        
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1_000_000
        self.latencies.append(latency_us)
        
        return True
    
    def best_bid(self):
        """Get best bid price."""
        return max(self.bids.keys()) if self.bids else None
    
    def best_ask(self):
        """Get best ask price."""
        return min(self.asks.keys()) if self.asks else None

class RealMetricsGenerator:
    """Generate real performance metrics using Python implementation."""
    
    def __init__(self):
        self.results = {}
        
    def run_order_book_benchmark(self, num_orders=100000):
        """Run order book benchmark and collect REAL latency metrics."""
        print(f"📊 Running order book benchmark with {num_orders:,} orders...")
        print("This measures REAL latency on your hardware...")
        
        order_book = PythonOrderBook()
        random.seed(42)  # Reproducible results
        
        # Generate realistic test data
        base_price = 150.0
        active_orders = []
        
        start_time = time.perf_counter()
        
        for i in range(num_orders):
            operation = random.choice(['add', 'add', 'add', 'cancel'])  # 75% adds, 25% cancels
            
            if operation == 'add' or not active_orders:
                # Add order
                price = base_price + random.uniform(-1, 1)  # $149-151 range
                quantity = random.randint(100, 1000)
                side = random.choice(['BUY', 'SELL'])
                
                order_id = order_book.add_order(price, quantity, side)
                active_orders.append(order_id)
                
                # Prevent memory explosion
                if len(active_orders) > 1000:
                    active_orders = active_orders[-500:]
                    
            else:
                # Cancel random order
                order_id = random.choice(active_orders)
                if order_book.cancel_order(order_id):
                    active_orders.remove(order_id)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Calculate REAL metrics
        latencies = order_book.latencies
        latencies.sort()
        
        results = {
            'total_operations': len(latencies),
            'total_time_seconds': total_duration,
            'average_latency_us': statistics.mean(latencies),
            'median_latency_us': statistics.median(latencies),
            'p50_latency_us': latencies[len(latencies) * 50 // 100],
            'p95_latency_us': latencies[len(latencies) * 95 // 100],
            'p99_latency_us': latencies[len(latencies) * 99 // 100],
            'p999_latency_us': latencies[len(latencies) * 999 // 1000] if len(latencies) >= 1000 else latencies[-1],
            'max_latency_us': max(latencies),
            'min_latency_us': min(latencies),
            'throughput_ops_per_sec': len(latencies) / total_duration,
            'orders_in_book': len(order_book.orders),
            'bid_levels': len(order_book.bids),
            'ask_levels': len(order_book.asks)
        }
        
        print("✅ Order book benchmark completed")
        print(f"\n🏎️  REAL ORDER BOOK PERFORMANCE (YOUR HARDWARE)")
        print(f"   Operations: {results['total_operations']:,}")
        print(f"   ⚡ P50 Latency: {results['p50_latency_us']:.2f} μs")
        print(f"   ⚡ P95 Latency: {results['p95_latency_us']:.2f} μs")
        print(f"   ⚡ P99 Latency: {results['p99_latency_us']:.2f} μs")
        print(f"   ⚡ Max Latency: {results['max_latency_us']:.2f} μs")
        print(f"   🚀 Throughput: {results['throughput_ops_per_sec']:,.0f} ops/sec")
        
        # Target validation
        print(f"\n📊 PERFORMANCE TARGET VALIDATION:")
        if results['p50_latency_us'] <= 10:
            print(f"   ✅ P50 target MET: {results['p50_latency_us']:.2f} μs ≤ 10 μs")
        else:
            print(f"   ❌ P50 target MISSED: {results['p50_latency_us']:.2f} μs > 10 μs")
            print(f"      (Python implementation - C++ would be faster)")
        
        if results['throughput_ops_per_sec'] >= 100000:
            print(f"   ✅ Throughput target MET: {results['throughput_ops_per_sec']:,.0f} ≥ 100k ops/sec")
        else:
            print(f"   ⚠️  Throughput: {results['throughput_ops_per_sec']:,.0f} ops/sec")
            print(f"      (Python implementation - C++ would be faster)")
        
        self.results['order_book_benchmark'] = results
        return results
    
    def run_trading_simulation(self, days=10):
        """Run trading simulation and compute REAL strategy metrics."""
        print(f"\n📈 Running trading simulation for {days} day(s)...")
        
        # Generate realistic market data
        np.random.seed(42)
        periods = days * 24 * 60 * 60  # seconds
        
        # Realistic market parameters
        initial_price = 150.0
        drift = 0.05 / 252 / (24 * 60 * 60)  # 5% annual
        volatility = 0.20 / (252 * 24 * 60 * 60) ** 0.5  # 20% annual
        
        # Generate price path
        returns = np.random.normal(drift, volatility, periods)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Market making simulation
        capital = 1_000_000
        inventory = 0
        cash = capital
        pnl_history = []
        trades = []
        
        for i in range(len(prices)):
            current_price = prices[i]
            
            # Simple market making logic
            if i % 60 == 0:  # Trade every minute
                # Market making signal (mean reversion)
                if i >= 60:
                    price_change = (current_price - prices[i-60]) / prices[i-60]
                    
                    # Trade against recent moves
                    if price_change > 0.001 and inventory > -100:  # Price up, sell
                        trade_qty = -10
                        trade_price = current_price * 1.0001  # Small edge
                        
                        inventory += trade_qty
                        cash -= trade_qty * trade_price
                        
                        trades.append({
                            'time': i,
                            'price': trade_price,
                            'quantity': trade_qty,
                            'inventory_after': inventory
                        })
                        
                    elif price_change < -0.001 and inventory < 100:  # Price down, buy
                        trade_qty = 10
                        trade_price = current_price * 0.9999  # Small edge
                        
                        inventory += trade_qty
                        cash -= trade_qty * trade_price
                        
                        trades.append({
                            'time': i,
                            'price': trade_price,
                            'quantity': trade_qty,
                            'inventory_after': inventory
                        })
            
            # Calculate PnL
            portfolio_value = cash + inventory * current_price
            pnl = portfolio_value - capital
            pnl_history.append(pnl)
        
        # Calculate REAL performance metrics
        pnl_series = np.array(pnl_history)
        returns_series = np.diff(pnl_series) / capital
        returns_series = returns_series[returns_series != 0]  # Remove zero returns
        
        if len(returns_series) > 0:
            sharpe_ratio = np.mean(returns_series) / np.std(returns_series) * np.sqrt(252 * 24 * 60 * 60)
            
            # Drawdown calculation
            cummax = np.maximum.accumulate(pnl_series)
            drawdown = cummax - pnl_series
            max_drawdown = np.max(drawdown)
            max_drawdown_pct = max_drawdown / capital
            
            # Win rate
            winning_trades = [t for t in trades if (t['quantity'] > 0 and prices[min(t['time'] + 60, len(prices)-1)] > t['price']) or 
                             (t['quantity'] < 0 and prices[min(t['time'] + 60, len(prices)-1)] < t['price'])]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
        else:
            sharpe_ratio = 0
            max_drawdown_pct = 0
            win_rate = 0
        
        results = {
            'final_pnl': pnl_series[-1],
            'total_return_pct': pnl_series[-1] / capital * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct * 100,
            'total_trades': len(trades),
            'win_rate_pct': win_rate * 100,
            'final_inventory': inventory,
            'days_simulated': days
        }
        
        print("✅ Trading simulation completed")
        print(f"\n💹 REAL TRADING STRATEGY PERFORMANCE")
        print(f"   Final PnL: ${results['final_pnl']:,.2f}")
        print(f"   Total Return: {results['total_return_pct']:.2f}%")
        print(f"   ⚡ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"   📊 Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"   🔄 Total Trades: {results['total_trades']}")
        print(f"   📦 Final Inventory: {results['final_inventory']} shares")
        
        # Target validation
        print(f"\n📊 TRADING TARGET VALIDATION:")
        if results['sharpe_ratio'] > 0.5:
            print(f"   ✅ Sharpe target MET: {results['sharpe_ratio']:.3f} > 0.5")
        else:
            print(f"   ⚠️  Sharpe ratio: {results['sharpe_ratio']:.3f} (target: > 0.5)")
        
        if results['max_drawdown_pct'] < 15:
            print(f"   ✅ Drawdown target MET: {results['max_drawdown_pct']:.2f}% < 15%")
        else:
            print(f"   ⚠️  Max drawdown: {results['max_drawdown_pct']:.2f}% (target: < 15%)")
        
        self.results['trading_simulation'] = results
        return results
    
    def run_python_tests(self):
        """Run Python module tests."""
        print(f"\n🐍 Testing Python modules...")
        
        try:
            # Test imports
            from hft_rl.env import HFTTradingEnv
            from hft_rl.baseline_agent import BaselineMarketMaker
            from hft_rl.backtest import Backtester
            
            print("   ✅ Core module imports successful")
            
            # Test environment creation
            env = HFTTradingEnv(max_steps=10)
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            
            print("   ✅ Environment functionality working")
            
            # Test baseline agent
            agent = BaselineMarketMaker()
            action = agent.get_action(obs, info)
            
            print("   ✅ Baseline agent working")
            
            self.results['python_tests'] = {'passed': True}
            return True
            
        except Exception as e:
            print(f"   ❌ Python test failed: {e}")
            self.results['python_tests'] = {'passed': False, 'error': str(e)}
            return False
    
    def generate_final_report(self):
        """Generate comprehensive real metrics report."""
        print(f"\n" + "="*60)
        print("🎯 FINAL REAL METRICS REPORT")
        print("="*60)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Generated: {timestamp}")
        print(f"System: Python {sys.version.split()[0]} on {sys.platform}")
        
        # Summary
        order_book = self.results.get('order_book_benchmark', {})
        trading = self.results.get('trading_simulation', {})
        tests = self.results.get('python_tests', {})
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"   Order Book P50 Latency: {order_book.get('p50_latency_us', 'N/A')} μs")
        print(f"   Order Book Throughput: {order_book.get('throughput_ops_per_sec', 'N/A'):,.0f} ops/sec")
        print(f"   Trading Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"   Trading Max Drawdown: {trading.get('max_drawdown_pct', 'N/A'):.2f}%")
        print(f"   Python Tests: {'✅ PASSED' if tests.get('passed') else '❌ FAILED'}")
        
        # Save results
        output_file = f"real_python_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Complete results saved to: {output_file}")
        print(f"\n🎉 SUCCESS: All metrics are REAL measurements!")
        print(f"   • Latencies measured on your actual hardware")
        print(f"   • Sharpe ratio computed from simulated trading") 
        print(f"   • All numbers are calculated, not hardcoded")
        
        return output_file

def main():
    print("🚀 PYTHON-BASED REAL METRICS GENERATOR")
    print("=" * 50)
    print("This generates ACTUAL performance measurements")
    print("using pure Python implementation.\n")
    
    generator = RealMetricsGenerator()
    
    try:
        # Run order book benchmark
        generator.run_order_book_benchmark(50000)  # 50k operations
        
        # Run trading simulation 
        generator.run_trading_simulation(110)  # 10 days
        
        # Test Python modules
        generator.run_python_tests()
        
        # Generate final report
        output_file = generator.generate_final_report()
        
        print(f"\n✅ Real metrics generation completed successfully!")
        print(f"📁 Results: {output_file}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())