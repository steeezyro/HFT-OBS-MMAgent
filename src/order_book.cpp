#include "order_book.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

namespace hft {

template <typename PriceT, typename QtyT>
LockFreeOrderBook<PriceT, QtyT>::LockFreeOrderBook(size_t max_orders)
    : order_pool_size_(max_orders * sizeof(Order))
    , level_pool_size_(1000 * sizeof(PriceLevel))
    , bid_levels_(nullptr)
    , ask_levels_(nullptr)
    , best_bid_(INVALID_PRICE)
    , best_ask_(INVALID_PRICE)
    , best_bid_qty_(0)
    , best_ask_qty_(0) {
    orders_.reserve(max_orders);
    
    // Allocate memory pools
    order_pool_memory_ = std::make_unique<char[]>(order_pool_size_);
    level_pool_memory_ = std::make_unique<char[]>(level_pool_size_);
}

template <typename PriceT, typename QtyT>
LockFreeOrderBook<PriceT, QtyT>::~LockFreeOrderBook() {
    // Cleanup is handled by pool destructors
}

template <typename PriceT, typename QtyT>
bool LockFreeOrderBook<PriceT, QtyT>::add_order(OrderId id, Price price, Quantity qty, Side side) {
    if (orders_.find(id) != orders_.end() || qty <= 0) {
        return false;
    }

    Order* order = allocate_order();
    if (!order) return false;

    order->id = id;
    order->price = price;
    order->quantity = qty;
    order->side = side;
    order->timestamp = std::chrono::high_resolution_clock::now().time_since_epoch();
    order->next = nullptr;
    order->prev = nullptr;

    PriceLevel* level = find_or_create_level(price, side);
    if (!level) {
        deallocate_order(order);
        return false;
    }

    // Insert order at end of price level
    if (level->tail) {
        level->tail->next = order;
        order->prev = level->tail;
        level->tail = order;
    } else {
        level->head = level->tail = order;
    }

    level->total_quantity += qty;
    level->order_count++;
    orders_[id] = order;
    total_orders_++;

    update_best_prices();
    
    BookUpdate update = get_book_update();
    {
        std::lock_guard<std::mutex> lock(update_queue_mutex_);
        update_queue_.push(update);
    }

    return true;
}

template <typename PriceT, typename QtyT>
bool LockFreeOrderBook<PriceT, QtyT>::modify_order(OrderId id, Quantity new_qty) {
    auto it = orders_.find(id);
    if (it == orders_.end() || new_qty <= 0) {
        return false;
    }

    Order* order = it->second;
    PriceLevel* level = find_or_create_level(order->price, order->side);
    
    level->total_quantity = level->total_quantity - order->quantity + new_qty;
    order->quantity = new_qty;
    total_modifications_++;

    update_best_prices();
    
    BookUpdate update = get_book_update();
    {
        std::lock_guard<std::mutex> lock(update_queue_mutex_);
        update_queue_.push(update);
    }

    return true;
}

template <typename PriceT, typename QtyT>
bool LockFreeOrderBook<PriceT, QtyT>::cancel_order(OrderId id) {
    auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }

    Order* order = it->second;
    PriceLevel* level = find_or_create_level(order->price, order->side);

    // Remove from linked list
    if (order->prev) {
        order->prev->next = order->next;
    } else {
        level->head = order->next;
    }

    if (order->next) {
        order->next->prev = order->prev;
    } else {
        level->tail = order->prev;
    }

    level->total_quantity -= order->quantity;
    level->order_count--;

    if (level->order_count == 0) {
        remove_empty_level(level, order->side);
    }

    orders_.erase(it);
    deallocate_order(order);
    total_cancellations_++;

    update_best_prices();
    
    BookUpdate update = get_book_update();
    {
        std::lock_guard<std::mutex> lock(update_queue_mutex_);
        update_queue_.push(update);
    }

    return true;
}

template <typename PriceT, typename QtyT>
void LockFreeOrderBook<PriceT, QtyT>::update_best_prices() {
    // Update best bid
    if (bid_levels_) {
        best_bid_ = bid_levels_->price;
        best_bid_qty_ = bid_levels_->total_quantity;
    } else {
        best_bid_ = INVALID_PRICE;
        best_bid_qty_ = 0;
    }

    // Update best ask
    if (ask_levels_) {
        best_ask_ = ask_levels_->price;
        best_ask_qty_ = ask_levels_->total_quantity;
    } else {
        best_ask_ = INVALID_PRICE;
        best_ask_qty_ = 0;
    }
}

template <typename PriceT, typename QtyT>
typename LockFreeOrderBook<PriceT, QtyT>::PriceLevel* 
LockFreeOrderBook<PriceT, QtyT>::find_or_create_level(Price price, Side side) {
    PriceLevel** levels = (side == Side::BUY) ? &bid_levels_ : &ask_levels_;
    
    // Find existing level
    PriceLevel* current = *levels;
    PriceLevel* prev = nullptr;
    
    while (current) {
        if (current->price == price) {
            return current;
        }
        
        bool should_insert = (side == Side::BUY) ? (price > current->price) : (price < current->price);
        if (should_insert) {
            break;
        }
        
        prev = current;
        current = current->next;
    }
    
    // Create new level
    size_t level_offset = level_pool_offset_.fetch_add(sizeof(PriceLevel));
    if (level_offset + sizeof(PriceLevel) > level_pool_size_) {
        return nullptr; // Pool exhausted
    }
    PriceLevel* new_level = reinterpret_cast<PriceLevel*>(level_pool_memory_.get() + level_offset);
    
    new_level->price = price;
    new_level->total_quantity = 0;
    new_level->order_count = 0;
    new_level->head = nullptr;
    new_level->tail = nullptr;
    new_level->next = current;
    new_level->prev = prev;
    
    if (prev) {
        prev->next = new_level;
    } else {
        *levels = new_level;
    }
    
    if (current) {
        current->prev = new_level;
    }
    
    return new_level;
}

template <typename PriceT, typename QtyT>
void LockFreeOrderBook<PriceT, QtyT>::remove_empty_level(PriceLevel* level, Side side) {
    PriceLevel** levels = (side == Side::BUY) ? &bid_levels_ : &ask_levels_;
    
    if (level->prev) {
        level->prev->next = level->next;
    } else {
        *levels = level->next;
    }
    
    if (level->next) {
        level->next->prev = level->prev;
    }
    
    // Simple pool - no deallocation in this implementation
}

template <typename PriceT, typename QtyT>
typename LockFreeOrderBook<PriceT, QtyT>::Order* 
LockFreeOrderBook<PriceT, QtyT>::allocate_order() {
    size_t offset = order_pool_offset_.fetch_add(sizeof(Order));
    if (offset + sizeof(Order) > order_pool_size_) {
        return nullptr; // Pool exhausted
    }
    return reinterpret_cast<Order*>(order_pool_memory_.get() + offset);
}

template <typename PriceT, typename QtyT>
void LockFreeOrderBook<PriceT, QtyT>::deallocate_order(Order* order) {
    // Simple pool - no deallocation in this implementation
    // In production, you'd implement a free list
}

template <typename PriceT, typename QtyT>
typename LockFreeOrderBook<PriceT, QtyT>::BookUpdate 
LockFreeOrderBook<PriceT, QtyT>::get_book_update() const {
    BookUpdate update;
    update.best_bid = best_bid_;
    update.best_ask = best_ask_;
    update.bid_qty = best_bid_qty_;
    update.ask_qty = best_ask_qty_;
    update.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    // Count levels
    update.bid_levels = 0;
    PriceLevel* level = bid_levels_;
    while (level) {
        update.bid_levels++;
        level = level->next;
    }
    
    update.ask_levels = 0;
    level = ask_levels_;
    while (level) {
        update.ask_levels++;
        level = level->next;
    }
    
    return update;
}

// Explicit template instantiations
template class LockFreeOrderBook<int64_t, int32_t>;
template class LockFreeOrderBook<int32_t, int32_t>;

} // namespace hft