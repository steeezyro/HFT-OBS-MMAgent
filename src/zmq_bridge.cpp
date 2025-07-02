#include "order_book.hpp"
#include "itch_parser.hpp"
// #include <zmq.hpp>  // Commented out to avoid dependency issues
#include <chrono>
#include <iostream>
#include <thread>
#include <signal.h>

namespace hft {

struct PackedBookUpdate {
    uint64_t timestamp;
    int64_t best_bid;
    int64_t best_ask;
    int32_t bid_qty;
    int32_t ask_qty;
    uint32_t bid_levels;
    uint32_t ask_levels;
    double spread;
    double mid_price;
} __attribute__((packed));

class ZMQBridge {
public:
    ZMQBridge(const std::string& endpoint = "tcp://*:5555")
        : context_(1)
        , socket_(context_, ZMQ_PUSH)
        , order_book_(1000000)
        , messages_sent_(0)
        , running_(true) {
        
        socket_.bind(endpoint);
        std::cout << "ZMQ Bridge listening on " << endpoint << std::endl;
    }
    
    ~ZMQBridge() {
        running_ = false;
        socket_.close();
        context_.close();
    }
    
    void run_with_itch_file(const std::string& itch_file, double speed_multiplier = 1.0) {
        ITCHParser parser(itch_file);
        
        std::cout << "Starting ITCH replay with ZMQ bridge..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t first_timestamp = 0;
        
        bool success = parser.parse([&](const ITCHMessage& msg) {
            if (!running_) return;
            
            // Apply speed control
            if (speed_multiplier < 100.0 && first_timestamp > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
                auto expected_elapsed = std::chrono::nanoseconds(static_cast<int64_t>(
                    (msg.timestamp - first_timestamp) / speed_multiplier));
                
                if (elapsed < expected_elapsed) {
                    std::this_thread::sleep_for(expected_elapsed - elapsed);
                }
            }
            
            if (first_timestamp == 0) {
                first_timestamp = msg.timestamp;
            }
            
            // Process message and update order book
            bool book_changed = process_itch_message(msg);
            
            // Send update if book changed
            if (book_changed) {
                send_book_update();
            }
        });
        
        if (success) {
            std::cout << "ITCH replay completed. Sent " << messages_sent_ << " book updates." << std::endl;
        } else {
            std::cerr << "ITCH replay failed!" << std::endl;
        }
    }
    
    void run_live_mode() {
        std::cout << "Running in live mode - sending synthetic book updates..." << std::endl;
        
        // Synthetic market data generator for testing
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_walk(-0.01, 0.01);
        std::uniform_int_distribution<> qty_dist(100, 1000);
        
        double base_price = 150.0;
        uint64_t order_id = 1;
        
        while (running_) {
            // Add some random orders
            for (int i = 0; i < 10; ++i) {
                double price_offset = price_walk(gen);
                int64_t price = static_cast<int64_t>((base_price + price_offset) * 10000);
                int32_t qty = qty_dist(gen);
                
                auto side = (i % 2 == 0) ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
                order_book_.add_order(order_id++, price, qty, side);
            }
            
            send_book_update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void stop() {
        running_ = false;
    }
    
    uint64_t messages_sent() const { return messages_sent_; }

private:
    bool process_itch_message(const ITCHMessage& msg) {
        switch (static_cast<ITCHMessageType>(msg.message_type)) {
            case ITCHMessageType::ADD_ORDER:
            case ITCHMessageType::ADD_ORDER_MPID: {
                auto side = (msg.side == 'B') ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
                return order_book_.add_order(msg.order_id, msg.price, msg.quantity, side);
            }
            case ITCHMessageType::ORDER_CANCEL:
                return order_book_.cancel_order(msg.order_id);
            case ITCHMessageType::ORDER_DELETE:
                return order_book_.cancel_order(msg.order_id);
            case ITCHMessageType::ORDER_EXECUTED:
            case ITCHMessageType::ORDER_EXECUTED_WITH_PRICE:
                return order_book_.modify_order(msg.order_id, msg.quantity - msg.executed_quantity);
            default:
                return false;
        }
    }
    
    void send_book_update() {
        auto book_update = order_book_.get_book_update();
        
        PackedBookUpdate packed;
        packed.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        packed.best_bid = book_update.best_bid;
        packed.best_ask = book_update.best_ask;
        packed.bid_qty = book_update.bid_qty;
        packed.ask_qty = book_update.ask_qty;
        packed.bid_levels = book_update.bid_levels;
        packed.ask_levels = book_update.ask_levels;
        
        // Calculate derived metrics
        if (packed.best_bid != std::numeric_limits<int64_t>::max() && 
            packed.best_ask != std::numeric_limits<int64_t>::max()) {
            packed.spread = (packed.best_ask - packed.best_bid) / 10000.0;
            packed.mid_price = (packed.best_ask + packed.best_bid) / 20000.0;
        } else {
            packed.spread = 0.0;
            packed.mid_price = 0.0;
        }
        
        zmq::message_t message(sizeof(PackedBookUpdate));
        memcpy(message.data(), &packed, sizeof(PackedBookUpdate));
        
        try {
            socket_.send(message, zmq::send_flags::dontwait);
            messages_sent_++;
            
            if (messages_sent_ % 10000 == 0) {
                std::cout << "Sent " << messages_sent_ << " updates. "
                          << "Bid: " << packed.best_bid / 10000.0 << ", "
                          << "Ask: " << packed.best_ask / 10000.0 << ", "
                          << "Spread: " << packed.spread << std::endl;
            }
        } catch (const zmq::error_t& e) {
            if (e.num() != EAGAIN) {  // EAGAIN is expected with DONTWAIT
                std::cerr << "ZMQ send error: " << e.what() << std::endl;
            }
        }
    }
    
    zmq::context_t context_;
    zmq::socket_t socket_;
    LockFreeOrderBook<> order_book_;
    std::atomic<uint64_t> messages_sent_;
    std::atomic<bool> running_;
};

} // namespace hft

// Global bridge instance for signal handling
hft::ZMQBridge* g_bridge = nullptr;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_bridge) {
        g_bridge->stop();
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  -f, --file PATH      ITCH file path for replay mode\n"
              << "  -s, --speed MULT     Speed multiplier for replay (default: 1.0)\n"
              << "  -e, --endpoint ADDR  ZMQ endpoint (default: tcp://*:5555)\n"
              << "  -l, --live           Run in live mode (synthetic data)\n"
              << "  -h, --help           Show this help message\n"
              << "\nModes:\n"
              << "  Replay mode: Replays ITCH file and sends book updates\n"
              << "  Live mode:   Generates synthetic book updates\n"
              << "\nExamples:\n"
              << "  " << program_name << " -f sample.itch -s 10.0\n"
              << "  " << program_name << " --live --endpoint tcp://*:6666\n";
}

int main(int argc, char* argv[]) {
    std::string itch_file;
    std::string endpoint = "tcp://*:5555";
    double speed = 1.0;
    bool live_mode = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                itch_file = argv[++i];
            }
        } else if (arg == "-s" || arg == "--speed") {
            if (i + 1 < argc) {
                speed = std::stod(argv[++i]);
            }
        } else if (arg == "-e" || arg == "--endpoint") {
            if (i + 1 < argc) {
                endpoint = argv[++i];
            }
        } else if (arg == "-l" || arg == "--live") {
            live_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!live_mode && itch_file.empty()) {
        std::cerr << "Error: ITCH file required for replay mode, or use --live for live mode\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Set up signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        hft::ZMQBridge bridge(endpoint);
        g_bridge = &bridge;
        
        if (live_mode) {
            bridge.run_live_mode();
        } else {
            // Create sample file if needed
            if (itch_file == "sample.itch") {
                std::cout << "Creating sample ITCH file..." << std::endl;
                hft::ITCHSampleGenerator::write_sample_file(itch_file, 10000);
            }
            
            bridge.run_with_itch_file(itch_file, speed);
        }
        
        std::cout << "ZMQ Bridge stopped. Total messages sent: " << bridge.messages_sent() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}