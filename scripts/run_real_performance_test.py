#!/usr/bin/env python3
"""
Complete pipeline to generate REAL performance metrics.
This script runs the entire system and collects actual measurements.
"""

import os
import sys
import json
import time
import subprocess
import signal
from pathlib import Path

class RealPerformanceTester:
    def __init__(self, data_file="data/sample.itch"):
        self.data_file = data_file
        self.build_dir = "build"
        self.results = {}
        
    def build_system(self):
        """Build the C++ system."""
        print("🔨 Building C++ system...")
        
        # Create build directory
        os.makedirs(self.build_dir, exist_ok=True)
        
        # Configure with CMake
        result = subprocess.run([
            "cmake", "-B", self.build_dir, 
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ CMake configuration failed:")
            print(result.stderr)
            return False
        
        # Build
        result = subprocess.run([
            "cmake", "--build", self.build_dir, "-j"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Build failed:")
            print(result.stderr)
            return False
        
        print("✅ Build successful")
        return True
    
    def run_latency_benchmark(self):
        """Run C++ latency benchmarks and collect REAL metrics."""
        print("📊 Running latency benchmarks...")
        
        benchmark_exe = os.path.join(self.build_dir, "bench_orderbook")
        if not os.path.exists(benchmark_exe):
            print(f"❌ Benchmark executable not found: {benchmark_exe}")
            return False
        
        # Run benchmark
        result = subprocess.run([benchmark_exe], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Benchmark failed:")
            print(result.stderr)
            return False
        
        print("✅ Latency benchmark completed")
        print(result.stdout)
        
        # Parse JSON results if available
        json_files = [
            "add_orders_benchmark.json",
            "mixed_operations_benchmark.json", 
            "book_update_benchmark.json"
        ]
        
        for json_file in json_files:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.results[json_file.replace('.json', '')] = json.load(f)
                print(f"📋 Loaded real metrics from {json_file}")
        
        return True
    
    def run_cpp_tests(self):
        """Run C++ tests and collect coverage."""
        print("🧪 Running C++ tests...")
        
        result = subprocess.run([
            "ctest", "--output-on-failure", "--verbose"
        ], cwd=self.build_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All C++ tests passed")
        else:
            print("⚠️ Some C++ tests failed:")
            print(result.stdout)
        
        self.results['cpp_tests'] = {
            'passed': result.returncode == 0,
            'output': result.stdout
        }
        
        return True
    
    def run_market_replay(self):
        """Test market data replay with real data."""
        print("📈 Testing market data replay...")
        
        if not os.path.exists(self.data_file):
            print(f"❌ Data file not found: {self.data_file}")
            return False
        
        replay_exe = os.path.join(self.build_dir, "replay")
        
        # Run replay for a short test
        result = subprocess.run([
            replay_exe, "-f", self.data_file, "-s", "100.0"  # 100x speed
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Market replay successful")
            print(result.stdout[-500:])  # Last 500 chars
        else:
            print("⚠️ Market replay had issues:")
            print(result.stderr)
        
        self.results['market_replay'] = {
            'success': result.returncode == 0,
            'output': result.stdout
        }
        
        return True
    
    def run_python_trading_test(self):
        """Test Python trading components."""
        print("🐍 Testing Python trading system...")
        
        # Start ZMQ bridge in background
        zmq_exe = os.path.join(self.build_dir, "zmq_bridge")
        
        if not os.path.exists(zmq_exe):
            print("⚠️ ZMQ bridge not available, skipping live test")
            return self._run_python_unit_tests()
        
        # Start bridge with sample data
        bridge_process = subprocess.Popen([
            zmq_exe, "--live", "--endpoint", "tcp://*:5557"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)  # Let bridge start
        
        try:
            # Run Python trading test
            python_test = """
import sys
sys.path.insert(0, 'python')

from hft_rl.env import HFTTradingEnv
from hft_rl.baseline_agent import BaselineMarketMaker

print("Creating trading environment...")
env = HFTTradingEnv(
    zmq_endpoint="tcp://localhost:5557",
    max_steps=100,
    max_inventory=1000
)

print("Creating baseline agent...")
agent = BaselineMarketMaker()

print("Running trading episode...")
results = agent.run_episode(env, max_steps=100, render=False)

print("\\n=== REAL TRADING RESULTS ===")
print(f"Final PnL: ${results['final_pnl']:.2f}")
print(f"Total Trades: {results['total_trades']}")
print(f"Duration: {results['duration']:.2f} seconds")
print(f"Steps: {results['steps']}")

# Get performance metrics
metrics = agent.get_performance_metrics()
print(f"\\n=== REAL PERFORMANCE METRICS ===")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

env.close()
print("\\n✅ Python trading test completed successfully")
"""
            
            result = subprocess.run([
                sys.executable, "-c", python_test
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Python trading test successful")
                print(result.stdout)
                
                # Parse results
                self.results['python_trading'] = {
                    'success': True,
                    'output': result.stdout
                }
            else:
                print("⚠️ Python trading test failed:")
                print(result.stderr)
                self.results['python_trading'] = {
                    'success': False,
                    'error': result.stderr
                }
        
        finally:
            # Cleanup bridge process
            bridge_process.terminate()
            bridge_process.wait()
        
        return True
    
    def _run_python_unit_tests(self):
        """Run Python unit tests."""
        print("Running Python unit tests...")
        
        test_cmd = [
            sys.executable, "-m", "pytest", 
            "python/tests/", "-v", "--tb=short"
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        
        self.results['python_tests'] = {
            'passed': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Python tests passed")
        else:
            print("⚠️ Some Python tests failed")
            print(result.stdout)
        
        return True
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n📋 Generating Performance Report")
        print("=" * 50)
        
        # C++ Performance Results
        if 'add_orders_benchmark' in self.results:
            bench = self.results['add_orders_benchmark']
            print(f"\n🏎️  C++ ORDER BOOK PERFORMANCE (REAL)")
            print(f"   Operations: {bench.get('total_operations', 'N/A'):,}")
            print(f"   P50 Latency: {bench.get('p50_latency_us', 'N/A')} μs")
            print(f"   P95 Latency: {bench.get('p95_latency_us', 'N/A')} μs")
            print(f"   P99 Latency: {bench.get('p99_latency_us', 'N/A')} μs")
            print(f"   Throughput: {bench.get('throughput', 'N/A'):,.0f} ops/sec")
            
            # Check targets
            p50 = bench.get('p50_latency_us', float('inf'))
            if p50 <= 10:
                print(f"   ✅ P50 latency target met ({p50} ≤ 10 μs)")
            else:
                print(f"   ❌ P50 latency target missed ({p50} > 10 μs)")
        
        # Test Results
        cpp_passed = self.results.get('cpp_tests', {}).get('passed', False)
        python_passed = self.results.get('python_tests', {}).get('passed', False)
        
        print(f"\n🧪 TEST RESULTS (REAL)")
        print(f"   C++ Tests: {'✅ PASSED' if cpp_passed else '❌ FAILED'}")
        print(f"   Python Tests: {'✅ PASSED' if python_passed else '❌ FAILED'}")
        
        # Trading Results
        if 'python_trading' in self.results:
            trading = self.results['python_trading']
            if trading.get('success'):
                print(f"\n📈 TRADING PERFORMANCE (REAL)")
                print("   See output above for actual PnL and trade metrics")
            else:
                print(f"\n📈 TRADING TEST: ❌ FAILED")
        
        # Save full results
        with open('real_performance_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Full results saved to: real_performance_results.json")
        print(f"\n🎯 SUMMARY: This report contains REAL measurements from your hardware")
        
    def run_full_test(self):
        """Run complete performance testing pipeline."""
        print("🚀 Starting REAL Performance Testing Pipeline")
        print("=" * 60)
        
        # Check if data file exists
        if not os.path.exists(self.data_file):
            print(f"⚠️  Data file {self.data_file} not found")
            print("Run: python scripts/download_real_data.py --sample")
            return False
        
        success = True
        
        # Build system
        if not self.build_system():
            return False
        
        # Run benchmarks
        if not self.run_latency_benchmark():
            success = False
        
        # Run tests
        if not self.run_cpp_tests():
            success = False
        
        # Test market replay
        if not self.run_market_replay():
            success = False
        
        # Test Python components
        if not self.run_python_trading_test():
            success = False
        
        # Generate report
        self.generate_report()
        
        return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run real performance tests')
    parser.add_argument('--data-file', default='data/sample.itch',
                       help='ITCH data file to use')
    
    args = parser.parse_args()
    
    tester = RealPerformanceTester(args.data_file)
    
    try:
        success = tester.run_full_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())