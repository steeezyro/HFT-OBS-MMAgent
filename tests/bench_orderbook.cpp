#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "../src/order_book.hpp"

using namespace hft;

struct BenchmarkResults {
    std::vector<uint64_t> latencies;
    uint64_t total_time_us;
    uint64_t total_operations;
    double throughput;
    
    void calculate_stats() {
        std::sort(latencies.begin(), latencies.end());
        throughput = total_operations * 1e6 / total_time_us;
    }
    
    uint64_t percentile(double p) const {
        size_t idx = static_cast<size_t>(latencies.size() * p / 100.0);
        return latencies[std::min(idx, latencies.size() - 1)];
    }
    
    void print_results(const std::string& test_name) const {
        std::cout << "\n=== " << test_name << " ===" << std::endl;
        std::cout << "Operations: " << total_operations << std::endl;
        std::cout << "Total time: " << total_time_us << " μs" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(0) << throughput << " ops/sec" << std::endl;
        std::cout << "Average latency: " << std::fixed << std::setprecision(2) 
                  << (double)total_time_us / total_operations << " μs" << std::endl;
        std::cout << "P50 latency: " << percentile(50) << " μs" << std::endl;
        std::cout << "P95 latency: " << percentile(95) << " μs" << std::endl;
        std::cout << "P99 latency: " << percentile(99) << " μs" << std::endl;
        std::cout << "P99.9 latency: " << percentile(99.9) << " μs" << std::endl;
        std::cout << "Max latency: " << latencies.back() << " μs" << std::endl;
    }
    
    void save_to_json(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "{\n";
            file << "  \"total_operations\": " << total_operations << ",\n";
            file << "  \"total_time_us\": " << total_time_us << ",\n";
            file << "  \"throughput\": " << throughput << ",\n";
            file << "  \"average_latency_us\": " << (double)total_time_us / total_operations << ",\n";
            file << "  \"p50_latency_us\": " << percentile(50) << ",\n";
            file << "  \"p95_latency_us\": " << percentile(95) << ",\n";
            file << "  \"p99_latency_us\": " << percentile(99) << ",\n";
            file << "  \"p999_latency_us\": " << percentile(99.9) << ",\n";
            file << "  \"max_latency_us\": " << latencies.back() << "\n";
            file << "}\n";
            file.close();
            std::cout << "Results saved to " << filename << std::endl;
        }
    }
};

class OrderBookBenchmark {
private:
    std::unique_ptr<LockFreeOrderBook<>> order_book_;
    std::mt19937 rng_;
    
public:
    OrderBookBenchmark() : order_book_(std::make_unique<LockFreeOrderBook<>>(1000000)), rng_(42) {}
    
    BenchmarkResults benchmark_add_orders(int num_orders) {
        BenchmarkResults results;
        results.latencies.reserve(num_orders);
        results.total_operations = num_orders;
        
        std::uniform_int_distribution<int64_t> price_dist(9000, 11000);
        std::uniform_int_distribution<int32_t> qty_dist(100, 1000);
        std::uniform_int_distribution<int> side_dist(0, 1);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_orders; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            int64_t price = price_dist(rng_);
            int32_t qty = qty_dist(rng_);
            auto side = side_dist(rng_) == 0 ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
            
            order_book_->add_order(i + 1, price, qty, side);
            
            auto op_end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(op_end - op_start);
            results.latencies.push_back(latency.count() / 1000); // Convert to microseconds
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        results.calculate_stats();
        
        return results;
    }
    
    BenchmarkResults benchmark_mixed_operations(int num_operations) {
        BenchmarkResults results;
        results.latencies.reserve(num_operations);
        results.total_operations = num_operations;
        
        std::uniform_int_distribution<int64_t> price_dist(9000, 11000);
        std::uniform_int_distribution<int32_t> qty_dist(100, 1000);
        std::uniform_int_distribution<int> side_dist(0, 1);
        std::uniform_int_distribution<int> op_dist(0, 99); // 0-69: add, 70-89: cancel, 90-99: modify
        
        std::vector<uint64_t> active_orders;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            int operation = op_dist(rng_);
            
            if (operation < 70 || active_orders.empty()) {
                // Add order (70% probability)
                uint64_t order_id = i + 1000000; // Avoid conflicts
                int64_t price = price_dist(rng_);
                int32_t qty = qty_dist(rng_);
                auto side = side_dist(rng_) == 0 ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
                
                if (order_book_->add_order(order_id, price, qty, side)) {
                    active_orders.push_back(order_id);
                }
            } else if (operation < 90) {
                // Cancel order (20% probability)
                std::uniform_int_distribution<size_t> order_idx_dist(0, active_orders.size() - 1);
                size_t idx = order_idx_dist(rng_);
                uint64_t order_id = active_orders[idx];
                
                if (order_book_->cancel_order(order_id)) {
                    active_orders.erase(active_orders.begin() + idx);
                }
            } else {
                // Modify order (10% probability)
                std::uniform_int_distribution<size_t> order_idx_dist(0, active_orders.size() - 1);
                size_t idx = order_idx_dist(rng_);
                uint64_t order_id = active_orders[idx];
                int32_t new_qty = qty_dist(rng_);
                
                order_book_->modify_order(order_id, new_qty);
            }
            
            auto op_end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(op_end - op_start);
            results.latencies.push_back(latency.count() / 1000); // Convert to microseconds
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        results.calculate_stats();
        
        return results;
    }
    
    BenchmarkResults benchmark_book_updates(int num_operations) {
        BenchmarkResults results;
        results.latencies.reserve(num_operations);
        results.total_operations = num_operations;
        
        // First, populate the order book
        std::uniform_int_distribution<int64_t> price_dist(9000, 11000);
        std::uniform_int_distribution<int32_t> qty_dist(100, 1000);
        std::uniform_int_distribution<int> side_dist(0, 1);
        
        for (int i = 0; i < 1000; ++i) {
            int64_t price = price_dist(rng_);
            int32_t qty = qty_dist(rng_);
            auto side = side_dist(rng_) == 0 ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
            order_book_->add_order(i + 1, price, qty, side);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_operations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            auto update = order_book_->get_book_update();
            
            auto op_end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(op_end - op_start);
            results.latencies.push_back(latency.count() / 1000); // Convert to microseconds
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        results.calculate_stats();
        
        return results;
    }
    
    void reset() {
        order_book_ = std::make_unique<LockFreeOrderBook<>>(1000000);
    }
};

int main() {
    std::cout << "HFT Order Book Performance Benchmark\n";
    std::cout << "=====================================\n";
    
    OrderBookBenchmark benchmark;
    
    // Benchmark 1: Pure add operations
    std::cout << "\nRunning add order benchmark..." << std::endl;
    auto add_results = benchmark.benchmark_add_orders(100000);
    add_results.print_results("Add Orders Benchmark");
    add_results.save_to_json("add_orders_benchmark.json");
    
    // Benchmark 2: Mixed operations
    benchmark.reset();
    std::cout << "\nRunning mixed operations benchmark..." << std::endl;
    auto mixed_results = benchmark.benchmark_mixed_operations(100000);
    mixed_results.print_results("Mixed Operations Benchmark");
    mixed_results.save_to_json("mixed_operations_benchmark.json");
    
    // Benchmark 3: Book update operations
    benchmark.reset();
    std::cout << "\nRunning book update benchmark..." << std::endl;
    auto update_results = benchmark.benchmark_book_updates(100000);
    update_results.print_results("Book Update Benchmark");
    update_results.save_to_json("book_update_benchmark.json");
    
    // Summary
    std::cout << "\n=== BENCHMARK SUMMARY ===" << std::endl;
    std::cout << "Add Orders - P50: " << add_results.percentile(50) << " μs, "
              << "Throughput: " << std::fixed << std::setprecision(0) << add_results.throughput << " ops/sec" << std::endl;
    std::cout << "Mixed Ops  - P50: " << mixed_results.percentile(50) << " μs, "
              << "Throughput: " << std::fixed << std::setprecision(0) << mixed_results.throughput << " ops/sec" << std::endl;
    std::cout << "Book Update- P50: " << update_results.percentile(50) << " μs, "
              << "Throughput: " << std::fixed << std::setprecision(0) << update_results.throughput << " ops/sec" << std::endl;
    
    // Performance targets check
    bool meets_targets = true;
    
    if (add_results.percentile(50) > 10) {
        std::cout << "\n❌ FAILED: Add orders P50 latency > 10 μs target" << std::endl;
        meets_targets = false;
    }
    
    if (mixed_results.percentile(50) > 15) {
        std::cout << "\n❌ FAILED: Mixed operations P50 latency > 15 μs target" << std::endl;
        meets_targets = false;
    }
    
    if (add_results.throughput < 100000) {
        std::cout << "\n❌ FAILED: Add orders throughput < 100k ops/sec target" << std::endl;
        meets_targets = false;
    }
    
    if (meets_targets) {
        std::cout << "\n✅ All performance targets met!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some performance targets not met!" << std::endl;
        return 1;
    }
}