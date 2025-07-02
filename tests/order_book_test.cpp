#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include "../src/order_book.hpp"

using namespace hft;

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        order_book = std::make_unique<LockFreeOrderBook<>>(10000);
    }

    void TearDown() override {
        order_book.reset();
    }

    std::unique_ptr<LockFreeOrderBook<>> order_book;
};

TEST_F(OrderBookTest, BasicAddOrder) {
    // Test adding a buy order
    EXPECT_TRUE(order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY));
    EXPECT_EQ(order_book->best_bid(), 10000);
    EXPECT_EQ(order_book->bid_quantity(), 100);
    
    // Test adding a sell order
    EXPECT_TRUE(order_book->add_order(2, 10010, 150, LockFreeOrderBook<>::Side::SELL));
    EXPECT_EQ(order_book->best_ask(), 10010);
    EXPECT_EQ(order_book->ask_quantity(), 150);
}

TEST_F(OrderBookTest, OrderPriority) {
    // Add orders at different price levels
    EXPECT_TRUE(order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(2, 10005, 200, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(3, 9995, 150, LockFreeOrderBook<>::Side::BUY));
    
    // Best bid should be the highest price
    EXPECT_EQ(order_book->best_bid(), 10005);
    EXPECT_EQ(order_book->bid_quantity(), 200);
    
    // Add sell orders
    EXPECT_TRUE(order_book->add_order(4, 10010, 100, LockFreeOrderBook<>::Side::SELL));
    EXPECT_TRUE(order_book->add_order(5, 10015, 200, LockFreeOrderBook<>::Side::SELL));
    EXPECT_TRUE(order_book->add_order(6, 10008, 150, LockFreeOrderBook<>::Side::SELL));
    
    // Best ask should be the lowest price
    EXPECT_EQ(order_book->best_ask(), 10008);
    EXPECT_EQ(order_book->ask_quantity(), 150);
}

TEST_F(OrderBookTest, CancelOrder) {
    // Add some orders
    EXPECT_TRUE(order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(2, 10005, 200, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(3, 10010, 150, LockFreeOrderBook<>::Side::SELL));
    
    // Cancel the best bid
    EXPECT_TRUE(order_book->cancel_order(2));
    EXPECT_EQ(order_book->best_bid(), 10000);
    EXPECT_EQ(order_book->bid_quantity(), 100);
    
    // Cancel non-existent order
    EXPECT_FALSE(order_book->cancel_order(999));
}

TEST_F(OrderBookTest, ModifyOrder) {
    // Add an order
    EXPECT_TRUE(order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY));
    EXPECT_EQ(order_book->bid_quantity(), 100);
    
    // Modify the order quantity
    EXPECT_TRUE(order_book->modify_order(1, 200));
    EXPECT_EQ(order_book->bid_quantity(), 200);
    
    // Modify non-existent order
    EXPECT_FALSE(order_book->modify_order(999, 100));
}

TEST_F(OrderBookTest, MultipleOrdersSamePrice) {
    // Add multiple orders at the same price level
    EXPECT_TRUE(order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(2, 10000, 150, LockFreeOrderBook<>::Side::BUY));
    EXPECT_TRUE(order_book->add_order(3, 10000, 200, LockFreeOrderBook<>::Side::BUY));
    
    // Total quantity should be sum of all orders
    EXPECT_EQ(order_book->best_bid(), 10000);
    EXPECT_EQ(order_book->bid_quantity(), 450);
    
    // Cancel one order
    EXPECT_TRUE(order_book->cancel_order(2));
    EXPECT_EQ(order_book->bid_quantity(), 300);
    
    // Cancel another order
    EXPECT_TRUE(order_book->cancel_order(1));
    EXPECT_EQ(order_book->bid_quantity(), 200);
    
    // Cancel last order
    EXPECT_TRUE(order_book->cancel_order(3));
    EXPECT_EQ(order_book->best_bid(), std::numeric_limits<int64_t>::max());
}

TEST_F(OrderBookTest, BookUpdates) {
    // Add some orders
    order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY);
    order_book->add_order(2, 10010, 150, LockFreeOrderBook<>::Side::SELL);
    
    // Check if updates are available
    EXPECT_TRUE(order_book->has_updates());
    
    // Pop an update
    LockFreeOrderBook<>::BookUpdate update;
    EXPECT_TRUE(order_book->pop_update(update));
    
    EXPECT_EQ(update.best_bid, 10000);
    EXPECT_EQ(update.best_ask, 10010);
    EXPECT_EQ(update.bid_qty, 100);
    EXPECT_EQ(update.ask_qty, 150);
}

TEST_F(OrderBookTest, Statistics) {
    // Initial stats should be zero
    EXPECT_EQ(order_book->total_orders(), 0);
    EXPECT_EQ(order_book->total_cancellations(), 0);
    EXPECT_EQ(order_book->total_modifications(), 0);
    
    // Add some orders
    order_book->add_order(1, 10000, 100, LockFreeOrderBook<>::Side::BUY);
    order_book->add_order(2, 10010, 150, LockFreeOrderBook<>::Side::SELL);
    EXPECT_EQ(order_book->total_orders(), 2);
    
    // Modify an order
    order_book->modify_order(1, 200);
    EXPECT_EQ(order_book->total_modifications(), 1);
    
    // Cancel an order
    order_book->cancel_order(2);
    EXPECT_EQ(order_book->total_cancellations(), 1);
}

TEST_F(OrderBookTest, PerformanceBenchmark) {
    const int NUM_ORDERS = 10000;
    std::vector<uint64_t> latencies;
    latencies.reserve(NUM_ORDERS);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> price_dist(9000, 11000);
    std::uniform_int_distribution<int32_t> qty_dist(100, 1000);
    std::uniform_int_distribution<int> side_dist(0, 1);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ORDERS; ++i) {
        auto order_start = std::chrono::high_resolution_clock::now();
        
        int64_t price = price_dist(gen);
        int32_t qty = qty_dist(gen);
        auto side = side_dist(gen) == 0 ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
        
        order_book->add_order(i + 1, price, qty, side);
        
        auto order_end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(order_end - order_start);
        latencies.push_back(latency.count());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    auto p50 = latencies[latencies.size() / 2];
    auto p95 = latencies[latencies.size() * 95 / 100];
    auto p99 = latencies[latencies.size() * 99 / 100];
    auto max_latency = latencies.back();
    
    double avg_latency = static_cast<double>(total_duration.count()) / NUM_ORDERS;
    double throughput = NUM_ORDERS * 1e6 / total_duration.count(); // orders per second
    
    std::cout << "\n=== Performance Benchmark Results ===" << std::endl;
    std::cout << "Orders processed: " << NUM_ORDERS << std::endl;
    std::cout << "Total time: " << total_duration.count() << " μs" << std::endl;
    std::cout << "Average latency: " << avg_latency << " μs" << std::endl;
    std::cout << "P50 latency: " << p50 << " μs" << std::endl;
    std::cout << "P95 latency: " << p95 << " μs" << std::endl;
    std::cout << "P99 latency: " << p99 << " μs" << std::endl;
    std::cout << "Max latency: " << max_latency << " μs" << std::endl;
    std::cout << "Throughput: " << throughput << " orders/sec" << std::endl;
    
    // Performance requirements
    EXPECT_LT(p50, 10); // P50 should be less than 10 μs
    EXPECT_LT(p95, 50); // P95 should be less than 50 μs
    EXPECT_GT(throughput, 100000); // Should handle 100k+ orders/sec
    
    std::cout << "Total orders in book: " << order_book->total_orders() << std::endl;
}

TEST_F(OrderBookTest, StressTest) {
    const int NUM_OPERATIONS = 50000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> price_dist(9000, 11000);
    std::uniform_int_distribution<int32_t> qty_dist(100, 1000);
    std::uniform_int_distribution<int> side_dist(0, 1);
    std::uniform_int_distribution<int> op_dist(0, 2); // 0=add, 1=cancel, 2=modify
    
    std::vector<uint64_t> active_orders;
    
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        int operation = op_dist(gen);
        
        if (operation == 0 || active_orders.empty()) {
            // Add order
            uint64_t order_id = i + 1;
            int64_t price = price_dist(gen);
            int32_t qty = qty_dist(gen);
            auto side = side_dist(gen) == 0 ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
            
            if (order_book->add_order(order_id, price, qty, side)) {
                active_orders.push_back(order_id);
            }
        } else if (operation == 1 && !active_orders.empty()) {
            // Cancel order
            std::uniform_int_distribution<size_t> order_idx_dist(0, active_orders.size() - 1);
            size_t idx = order_idx_dist(gen);
            uint64_t order_id = active_orders[idx];
            
            if (order_book->cancel_order(order_id)) {
                active_orders.erase(active_orders.begin() + idx);
            }
        } else if (operation == 2 && !active_orders.empty()) {
            // Modify order
            std::uniform_int_distribution<size_t> order_idx_dist(0, active_orders.size() - 1);
            size_t idx = order_idx_dist(gen);
            uint64_t order_id = active_orders[idx];
            int32_t new_qty = qty_dist(gen);
            
            order_book->modify_order(order_id, new_qty);
        }
    }
    
    std::cout << "\n=== Stress Test Results ===" << std::endl;
    std::cout << "Operations: " << NUM_OPERATIONS << std::endl;
    std::cout << "Active orders: " << active_orders.size() << std::endl;
    std::cout << "Total orders: " << order_book->total_orders() << std::endl;
    std::cout << "Total cancellations: " << order_book->total_cancellations() << std::endl;
    std::cout << "Total modifications: " << order_book->total_modifications() << std::endl;
    
    // Book should still be functional
    auto update = order_book->get_book_update();
    std::cout << "Best bid: " << update.best_bid << std::endl;
    std::cout << "Best ask: " << update.best_ask << std::endl;
    std::cout << "Bid levels: " << update.bid_levels << std::endl;
    std::cout << "Ask levels: " << update.ask_levels << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}