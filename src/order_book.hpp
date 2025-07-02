#pragma once

#include <queue>
#include <memory>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_map>

namespace hft {

template <typename PriceT = int64_t, typename QtyT = int32_t>
class LockFreeOrderBook {
public:
    using Price = PriceT;
    using Quantity = QtyT;
    using OrderId = uint64_t;
    using Timestamp = std::chrono::nanoseconds;

    enum class Side : uint8_t { BUY = 0, SELL = 1 };

    struct Order {
        OrderId id;
        Price price;
        Quantity quantity;
        Side side;
        Timestamp timestamp;
        Order* next;
        Order* prev;
    };

    struct BookUpdate {
        Price best_bid;
        Price best_ask;
        Quantity bid_qty;
        Quantity ask_qty;
        Timestamp timestamp;
        uint32_t bid_levels;
        uint32_t ask_levels;
    };

    struct PriceLevel {
        Price price;
        Quantity total_quantity;
        uint32_t order_count;
        Order* head;
        Order* tail;
        PriceLevel* next;
        PriceLevel* prev;
    };

    explicit LockFreeOrderBook(size_t max_orders = 1000000);
    ~LockFreeOrderBook();

    bool add_order(OrderId id, Price price, Quantity qty, Side side);
    bool modify_order(OrderId id, Quantity new_qty);
    bool cancel_order(OrderId id);
    
    Price best_bid() const noexcept { return best_bid_; }
    Price best_ask() const noexcept { return best_ask_; }
    Quantity bid_quantity() const noexcept { return best_bid_qty_; }
    Quantity ask_quantity() const noexcept { return best_ask_qty_; }
    
    BookUpdate get_book_update() const;
    bool has_updates() const { 
        std::lock_guard<std::mutex> lock(update_queue_mutex_);
        return !update_queue_.empty(); 
    }
    bool pop_update(BookUpdate& update) { 
        std::lock_guard<std::mutex> lock(update_queue_mutex_);
        if (update_queue_.empty()) return false;
        update = update_queue_.front();
        update_queue_.pop();
        return true;
    }

    // Statistics
    uint64_t total_orders() const noexcept { return total_orders_; }
    uint64_t total_cancellations() const noexcept { return total_cancellations_; }
    uint64_t total_modifications() const noexcept { return total_modifications_; }

private:
    void update_best_prices();
    void remove_empty_level(PriceLevel* level, Side side);
    PriceLevel* find_or_create_level(Price price, Side side);
    Order* allocate_order();
    void deallocate_order(Order* order);

    // Memory management
    std::unique_ptr<char[]> order_pool_memory_;
    std::unique_ptr<char[]> level_pool_memory_;
    size_t order_pool_size_;
    size_t level_pool_size_;
    std::atomic<size_t> order_pool_offset_{0};
    std::atomic<size_t> level_pool_offset_{0};
    
    // Price levels
    PriceLevel* bid_levels_;
    PriceLevel* ask_levels_;
    
    // Best prices cache
    Price best_bid_;
    Price best_ask_;
    Quantity best_bid_qty_;
    Quantity best_ask_qty_;
    
    // Order lookup
    std::unordered_map<OrderId, Order*> orders_;
    
    // Update queue
    std::queue<BookUpdate> update_queue_;
    mutable std::mutex update_queue_mutex_;
    
    // Statistics
    std::atomic<uint64_t> total_orders_{0};
    std::atomic<uint64_t> total_cancellations_{0};
    std::atomic<uint64_t> total_modifications_{0};
    
    static constexpr Price INVALID_PRICE = std::numeric_limits<Price>::max();
};

// Explicit template instantiations
extern template class LockFreeOrderBook<int64_t, int32_t>;
extern template class LockFreeOrderBook<int32_t, int32_t>;

} // namespace hft