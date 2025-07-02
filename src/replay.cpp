#include "itch_parser.hpp"
#include "order_book.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <openssl/md5.h>
#include <limits>
#include <thread>

namespace hft {

class ReplayEngine {
public:
    ReplayEngine(const std::string& itch_file, double speed_multiplier = 1.0, 
                 const std::string& symbol_filter = "")
        : itch_file_(itch_file)
        , speed_multiplier_(speed_multiplier)
        , symbol_filter_(symbol_filter)
        , order_book_(1000000)
        , messages_processed_(0)
        , start_time_(std::chrono::high_resolution_clock::now()) {
        
        // Initialize MD5 context
        MD5_Init(&md5_context_);
    }
    
    bool run() {
        ITCHParser parser(itch_file_);
        
        std::cout << "Starting replay of " << itch_file_ << std::endl;
        std::cout << "Speed multiplier: " << speed_multiplier_ << "x" << std::endl;
        std::cout << "Symbol filter: " << (symbol_filter_.empty() ? "ALL" : symbol_filter_) << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = parser.parse([this](const ITCHMessage& msg) {
            process_message(msg);
        }, symbol_filter_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (success) {
            std::cout << "\nReplay completed successfully!" << std::endl;
            std::cout << "Messages processed: " << messages_processed_ << std::endl;
            std::cout << "Duration: " << duration.count() << " ms" << std::endl;
            std::cout << "Rate: " << parser.parse_rate_mbps() << " MB/s" << std::endl;
            std::cout << "Message rate: " << (messages_processed_ * 1000.0 / duration.count()) << " msgs/s" << std::endl;
            
            // Print final MD5 hash
            unsigned char hash[MD5_DIGEST_LENGTH];
            MD5_Final(hash, &md5_context_);
            
            std::cout << "Final book hash: ";
            for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
                printf("%02x", hash[i]);
            }
            std::cout << std::endl;
            
            print_book_stats();
        } else {
            std::cerr << "Replay failed!" << std::endl;
            return false;
        }
        
        return true;
    }

private:
    void process_message(const ITCHMessage& msg) {
        messages_processed_++;
        
        // Apply speed control
        if (speed_multiplier_ < 100.0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_);
            auto expected_elapsed = std::chrono::nanoseconds(static_cast<int64_t>(
                (msg.timestamp - first_timestamp_) / speed_multiplier_));
            
            if (elapsed < expected_elapsed) {
                std::this_thread::sleep_for(expected_elapsed - elapsed);
            }
        }
        
        // Process message
        switch (static_cast<ITCHMessageType>(msg.message_type)) {
            case ITCHMessageType::ADD_ORDER:
            case ITCHMessageType::ADD_ORDER_MPID:
                add_order(msg);
                break;
            case ITCHMessageType::ORDER_CANCEL:
                cancel_order(msg);
                break;
            case ITCHMessageType::ORDER_DELETE:
                delete_order(msg);
                break;
            case ITCHMessageType::ORDER_EXECUTED:
            case ITCHMessageType::ORDER_EXECUTED_WITH_PRICE:
                execute_order(msg);
                break;
            default:
                break;
        }
        
        // Update MD5 hash every millisecond
        if (messages_processed_ % 1000 == 0) {
            update_book_hash();
        }
        
        // Print progress
        if (messages_processed_ % 100000 == 0) {
            std::cout << "Processed " << messages_processed_ << " messages, "
                      << "Best bid: " << order_book_.best_bid() << ", "
                      << "Best ask: " << order_book_.best_ask() << std::endl;
        }
        
        if (first_timestamp_ == 0) {
            first_timestamp_ = msg.timestamp;
        }
    }
    
    void add_order(const ITCHMessage& msg) {
        auto side = (msg.side == 'B') ? LockFreeOrderBook<>::Side::BUY : LockFreeOrderBook<>::Side::SELL;
        order_book_.add_order(msg.order_id, msg.price, msg.quantity, side);
    }
    
    void cancel_order(const ITCHMessage& msg) {
        // For cancellation, we need to modify the order
        // In a real implementation, we'd track the remaining quantity
        order_book_.cancel_order(msg.order_id);
    }
    
    void delete_order(const ITCHMessage& msg) {
        order_book_.cancel_order(msg.order_id);
    }
    
    void execute_order(const ITCHMessage& msg) {
        // For execution, we reduce the order quantity
        // In a real implementation, we'd track executed quantity
        order_book_.modify_order(msg.order_id, msg.quantity - msg.executed_quantity);
    }
    
    void update_book_hash() {
        auto update = order_book_.get_book_update();
        MD5_Update(&md5_context_, &update.best_bid, sizeof(update.best_bid));
        MD5_Update(&md5_context_, &update.best_ask, sizeof(update.best_ask));
        MD5_Update(&md5_context_, &update.bid_qty, sizeof(update.bid_qty));
        MD5_Update(&md5_context_, &update.ask_qty, sizeof(update.ask_qty));
    }
    
    void print_book_stats() {
        std::cout << "\n=== Order Book Statistics ===" << std::endl;
        std::cout << "Total orders: " << order_book_.total_orders() << std::endl;
        std::cout << "Total cancellations: " << order_book_.total_cancellations() << std::endl;
        std::cout << "Total modifications: " << order_book_.total_modifications() << std::endl;
        std::cout << "Best bid: " << order_book_.best_bid() << std::endl;
        std::cout << "Best ask: " << order_book_.best_ask() << std::endl;
        std::cout << "Bid quantity: " << order_book_.bid_quantity() << std::endl;
        std::cout << "Ask quantity: " << order_book_.ask_quantity() << std::endl;
    }
    
    std::string itch_file_;
    double speed_multiplier_;
    std::string symbol_filter_;
    LockFreeOrderBook<> order_book_;
    
    size_t messages_processed_;
    std::chrono::high_resolution_clock::time_point start_time_;
    uint64_t first_timestamp_;
    
    MD5_CTX md5_context_;
};

} // namespace hft

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  -f, --file PATH      ITCH file path (required)\n"
              << "  -s, --speed MULT     Speed multiplier (default: 1.0)\n"
              << "  -S, --symbol SYM     Symbol filter (default: all symbols)\n"
              << "  -h, --help           Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " -f sample.itch -s 10.0 -S AAPL\n"
              << "  " << program_name << " --file data.itch --speed 0.1\n";
}

int main(int argc, char* argv[]) {
    std::string file_path;
    double speed = 1.0;
    std::string symbol_filter;
    
    static struct option long_options[] = {
        {"file",   required_argument, 0, 'f'},
        {"speed",  required_argument, 0, 's'},
        {"symbol", required_argument, 0, 'S'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    while ((c = getopt_long(argc, argv, "f:s:S:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'f':
                file_path = optarg;
                break;
            case 's':
                speed = std::stod(optarg);
                break;
            case 'S':
                symbol_filter = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
            default:
                break;
        }
    }
    
    if (file_path.empty()) {
        std::cerr << "Error: ITCH file path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Create sample file if it doesn't exist
    if (file_path == "sample.itch") {
        std::cout << "Creating sample ITCH file..." << std::endl;
        hft::ITCHSampleGenerator::write_sample_file(file_path, 10000);
    }
    
    try {
        hft::ReplayEngine engine(file_path, speed, symbol_filter);
        return engine.run() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}