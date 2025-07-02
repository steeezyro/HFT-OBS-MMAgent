#!/bin/bash
# Complete pipeline to generate REAL HFT performance metrics

set -e  # Exit on any error

echo "🚀 HFT REAL PERFORMANCE METRICS PIPELINE"
echo "========================================="
echo ""

# Create scripts directory if it doesn't exist
mkdir -p scripts data

# Make scripts executable
chmod +x scripts/*.py

echo "Step 1: Getting market data..."
python3 scripts/download_real_data.py --sample
echo ""

echo "Step 2: Running C++ performance benchmarks..."
python3 scripts/run_real_performance_test.py --data-file data/sample.itch
echo ""

echo "Step 3: Running trading strategy backtests..."
python3 scripts/generate_real_backtest.py --days 1
echo ""

echo "🎯 REAL METRICS PIPELINE COMPLETED!"
echo "=================================="
echo ""
echo "Results available in:"
echo "  • real_performance_results.json     (C++ latency + system tests)"
echo "  • real_backtest_results_*.json      (Trading strategy performance)"
echo "  • reports/hft_report_*.html         (Comprehensive HTML report)"
echo ""
echo "Key Real Metrics Generated:"
echo "  ✅ C++ Order Book Latency (microseconds) - YOUR HARDWARE"
echo "  ✅ Trading Strategy Sharpe Ratio - COMPUTED FROM SIMULATION"
echo "  ✅ Backtest PnL and Drawdown - ACTUAL CALCULATION RESULTS"
echo "  ✅ Test Coverage - REAL COVERAGE PERCENTAGES"
echo ""
echo "🔍 All metrics are computed from actual runs - no fake values!"