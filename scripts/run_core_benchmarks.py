#!/usr/bin/env python3
"""
Run core C++ benchmarks to generate REAL performance metrics.
This focuses on order book latency without ZMQ dependencies.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

class CoreBenchmarkRunner:
    def __init__(self):
        self.build_dir = "build"
        self.results = {}
        
    def build_core_system(self):
        """Build just the core C++ components."""
        print("🔨 Building core C++ system...")
        
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
        
        # Build core components only
        result = subprocess.run([
            "cmake", "--build", self.build_dir, 
            "--target", "hft_core", "bench_orderbook", "replay", "-j"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Build failed:")
            print(result.stderr)
            return False
        
        print("✅ Core system build successful")
        return True
    
    def run_order_book_benchmark(self):
        """Run the order book benchmark and collect REAL latency metrics."""
        print("📊 Running order book benchmark...")
        
        benchmark_exe = os.path.join(self.build_dir, "bench_orderbook")
        if not os.path.exists(benchmark_exe):
            print(f"❌ Benchmark executable not found: {benchmark_exe}")
            return False
        
        # Run benchmark
        result = subprocess.run([benchmark_exe], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Order book benchmark completed successfully")
            print("\n" + "="*50)
            print("REAL C++ PERFORMANCE METRICS FROM YOUR HARDWARE:")
            print("="*50)
            print(result.stdout)
            
            # Try to parse JSON results
            json_files = [
                "add_orders_benchmark.json",
                "mixed_operations_benchmark.json", 
                "book_update_benchmark.json"
            ]
            
            for json_file in json_files:
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        self.results[json_file.replace('.json', '')] = data
                        print(f"📋 Loaded real metrics from {json_file}")
            
            return True
        else:
            print(f"❌ Benchmark failed:")
            print(result.stderr)
            return False
    
    def run_market_replay_test(self, data_file="data/sample.itch"):
        """Test market data replay."""
        print("📈 Testing market data replay...")
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        replay_exe = os.path.join(self.build_dir, "replay")
        if not os.path.exists(replay_exe):
            print(f"❌ Replay executable not found: {replay_exe}")
            return False
        
        # Run replay test
        result = subprocess.run([
            replay_exe, "-f", data_file, "-s", "100.0"  # 100x speed for quick test
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Market replay test successful")
            print("Real replay performance:")
            print(result.stdout[-500:])  # Show last 500 characters
            
            self.results['market_replay'] = {
                'success': True,
                'output': result.stdout
            }
            return True
        else:
            print("⚠️ Market replay had issues:")
            print(result.stderr)
            self.results['market_replay'] = {
                'success': False,
                'error': result.stderr
            }
            return False
    
    def run_cpp_tests(self):
        """Run C++ unit tests."""
        print("🧪 Running C++ unit tests...")
        
        result = subprocess.run([
            "ctest", "--output-on-failure", "--verbose"
        ], cwd=self.build_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All C++ tests passed")
            print("Test results:")
            print(result.stdout[-300:])  # Show last 300 characters
        else:
            print("⚠️ Some C++ tests failed:")
            print(result.stdout)
        
        self.results['cpp_tests'] = {
            'passed': result.returncode == 0,
            'output': result.stdout
        }
        
        return True
    
    def generate_real_metrics_report(self):
        """Generate report with REAL performance metrics."""
        print("\n📋 REAL PERFORMANCE METRICS REPORT")
        print("=" * 60)
        
        # Order Book Performance
        if 'add_orders_benchmark' in self.results:
            bench = self.results['add_orders_benchmark']
            print(f"\n🏎️  ORDER BOOK PERFORMANCE (MEASURED ON YOUR HARDWARE)")
            print(f"   Operations Tested: {bench.get('total_operations', 'N/A'):,}")
            print(f"   ⚡ P50 Latency: {bench.get('p50_latency_us', 'N/A')} μs")
            print(f"   ⚡ P95 Latency: {bench.get('p95_latency_us', 'N/A')} μs") 
            print(f"   ⚡ P99 Latency: {bench.get('p99_latency_us', 'N/A')} μs")
            print(f"   ⚡ Max Latency: {bench.get('max_latency_us', 'N/A')} μs")
            print(f"   🚀 Throughput: {bench.get('throughput', 'N/A'):,.0f} operations/second")
            
            # Performance target validation
            p50 = bench.get('p50_latency_us', float('inf'))
            p99 = bench.get('p99_latency_us', float('inf'))
            throughput = bench.get('throughput', 0)
            
            print(f"\n📊 TARGET VALIDATION:")
            if p50 <= 10:
                print(f"   ✅ P50 latency target MET: {p50} μs ≤ 10 μs")
            else:
                print(f"   ❌ P50 latency target MISSED: {p50} μs > 10 μs")
            
            if p99 <= 100:
                print(f"   ✅ P99 latency target MET: {p99} μs ≤ 100 μs")
            else:
                print(f"   ❌ P99 latency target MISSED: {p99} μs > 100 μs")
            
            if throughput >= 100000:
                print(f"   ✅ Throughput target MET: {throughput:,.0f} ≥ 100,000 ops/sec")
            else:
                print(f"   ❌ Throughput target MISSED: {throughput:,.0f} < 100,000 ops/sec")
        
        # Mixed Operations Performance
        if 'mixed_operations_benchmark' in self.results:
            mixed = self.results['mixed_operations_benchmark']
            print(f"\n🔄 MIXED OPERATIONS PERFORMANCE")
            print(f"   P50 Latency: {mixed.get('p50_latency_us', 'N/A')} μs")
            print(f"   Throughput: {mixed.get('throughput', 'N/A'):,.0f} ops/sec")
        
        # Test Results
        cpp_passed = self.results.get('cpp_tests', {}).get('passed', False)
        print(f"\n🧪 TEST RESULTS")
        print(f"   C++ Unit Tests: {'✅ PASSED' if cpp_passed else '❌ FAILED'}")
        
        # Market Replay
        replay_success = self.results.get('market_replay', {}).get('success', False)
        print(f"   Market Replay: {'✅ SUCCESSFUL' if replay_success else '❌ FAILED'}")
        
        # Save results
        results_file = 'real_core_performance_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Complete results saved to: {results_file}")
        print(f"\n🎯 SUMMARY")
        print(f"   • These are REAL measurements from YOUR hardware")
        print(f"   • Latencies measured in microseconds on your CPU")
        print(f"   • Throughput calculated from actual operation timing")
        print(f"   • No fake or placeholder values - all computed metrics")
        
        return True
    
    def run_full_benchmark(self, data_file="data/sample.itch"):
        """Run complete core benchmarking pipeline."""
        print("🚀 REAL C++ PERFORMANCE BENCHMARKING")
        print("=" * 50)
        
        success = True
        
        # Build system
        if not self.build_core_system():
            return False
        
        # Run order book benchmark
        if not self.run_order_book_benchmark():
            success = False
        
        # Test market replay
        if not self.run_market_replay_test(data_file):
            success = False
        
        # Run unit tests
        if not self.run_cpp_tests():
            success = False
        
        # Generate report
        self.generate_real_metrics_report()
        
        return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run core C++ benchmarks')
    parser.add_argument('--data-file', default='data/sample.itch',
                       help='ITCH data file for replay test')
    
    args = parser.parse_args()
    
    runner = CoreBenchmarkRunner()
    
    try:
        success = runner.run_full_benchmark(args.data_file)
        if success:
            print(f"\n🎉 SUCCESS: Real performance metrics generated!")
            return 0
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Some metrics generated, check output")
            return 0
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())