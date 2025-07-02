#include "itch_parser.hpp"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hft {

ITCHParser::ITCHParser(const std::string& file_path)
    : file_path_(file_path)
    , mapped_data_(nullptr)
    , file_size_(0)
    , fd_(-1)
    , messages_parsed_(0)
    , parse_rate_mbps_(0.0) {
}

ITCHParser::~ITCHParser() {
    unmap_file();
}

bool ITCHParser::map_file() {
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "Failed to open file: " << file_path_ << std::endl;
        return false;
    }
    
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        return false;
    }
    
    file_size_ = st.st_size;
    mapped_data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        return false;
    }
    
    // Advise kernel about access pattern
    madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);
    
    return true;
}

void ITCHParser::unmap_file() {
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

bool ITCHParser::parse(MessageCallback callback, const std::string& symbol_filter) {
    return parse_range(callback, 0, file_size_, symbol_filter);
}

bool ITCHParser::parse_range(MessageCallback callback, size_t start_offset, size_t end_offset,
                             const std::string& symbol_filter) {
    if (!map_file()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    messages_parsed_ = 0;
    
    size_t offset = start_offset;
    while (offset + 2 < end_offset) {
        uint16_t message_length = parse_uint16(mapped_data_ + offset);
        offset += 2;
        
        if (offset + message_length > end_offset) {
            break;
        }
        
        ITCHMessage msg;
        if (parse_message(mapped_data_ + offset, offset, msg)) {
            // Apply symbol filter if specified
            if (symbol_filter.empty() || 
                strncmp(msg.symbol, symbol_filter.c_str(), std::min(symbol_filter.length(), size_t(8))) == 0) {
                callback(msg);
            }
            messages_parsed_++;
        }
        
        offset += message_length;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double seconds = duration.count() / 1e6;
    double bytes_processed = end_offset - start_offset;
    parse_rate_mbps_ = (bytes_processed / (1024 * 1024)) / seconds;
    
    return true;
}

bool ITCHParser::parse_message(const uint8_t* data, size_t offset, ITCHMessage& msg) {
    msg.message_type = data[0];
    
    switch (static_cast<ITCHMessageType>(msg.message_type)) {
        case ITCHMessageType::ADD_ORDER:
            return parse_add_order(data, msg);
        case ITCHMessageType::ADD_ORDER_MPID:
            return parse_add_order_mpid(data, msg);
        case ITCHMessageType::ORDER_EXECUTED:
            return parse_order_executed(data, msg);
        case ITCHMessageType::ORDER_EXECUTED_WITH_PRICE:
            return parse_order_executed_with_price(data, msg);
        case ITCHMessageType::ORDER_CANCEL:
            return parse_order_cancel(data, msg);
        case ITCHMessageType::ORDER_DELETE:
            return parse_order_delete(data, msg);
        case ITCHMessageType::ORDER_REPLACE:
            return parse_order_replace(data, msg);
        default:
            return false;
    }
}

bool ITCHParser::parse_add_order(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'A') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    msg.side = data[17];
    msg.quantity = parse_uint32(data + 18);
    memcpy(msg.symbol, data + 22, 8);
    msg.price = parse_uint32(data + 30);
    
    return true;
}

bool ITCHParser::parse_add_order_mpid(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'F') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    msg.side = data[17];
    msg.quantity = parse_uint32(data + 18);
    memcpy(msg.symbol, data + 22, 8);
    msg.price = parse_uint32(data + 30);
    
    return true;
}

bool ITCHParser::parse_order_executed(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'E') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    msg.executed_quantity = parse_uint32(data + 17);
    msg.match_number = parse_uint64(data + 21);
    
    return true;
}

bool ITCHParser::parse_order_executed_with_price(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'C') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    msg.executed_quantity = parse_uint32(data + 17);
    msg.match_number = parse_uint64(data + 21);
    msg.printable = data[29];
    msg.price = parse_uint32(data + 30);
    
    return true;
}

bool ITCHParser::parse_order_cancel(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'X') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    msg.quantity = parse_uint32(data + 17); // cancelled shares
    
    return true;
}

bool ITCHParser::parse_order_delete(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'D') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9);
    
    return true;
}

bool ITCHParser::parse_order_replace(const uint8_t* data, ITCHMessage& msg) {
    if (data[0] != 'U') return false;
    
    msg.timestamp = parse_uint64(data + 1);
    msg.order_id = parse_uint64(data + 9); // original order id
    // New order details follow...
    
    return true;
}

uint16_t ITCHParser::parse_uint16(const uint8_t* data) {
    return (static_cast<uint16_t>(data[0]) << 8) | data[1];
}

uint32_t ITCHParser::parse_uint32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           data[3];
}

uint64_t ITCHParser::parse_uint64(const uint8_t* data) {
    return (static_cast<uint64_t>(parse_uint32(data)) << 32) |
           parse_uint32(data + 4);
}

double ITCHParser::parse_price(uint32_t raw_price) {
    return raw_price / 10000.0; // ITCH prices are in 1/10000 units
}

// Sample generator implementation
std::vector<uint8_t> ITCHSampleGenerator::generate_sample_messages(size_t count) {
    std::vector<uint8_t> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> order_id_dist(1, 1000000);
    std::uniform_int_distribution<uint32_t> price_dist(1000000, 2000000); // $100-$200 in 1/10000 units
    std::uniform_int_distribution<uint32_t> qty_dist(100, 10000);
    std::uniform_int_distribution<int> side_dist(0, 1);
    
    const std::vector<std::string> symbols = {"AAPL    ", "MSFT    ", "GOOGL   ", "TSLA    "};
    
    uint64_t timestamp = 1640995200000000000ULL; // 2022-01-01 00:00:00 UTC in nanoseconds
    
    for (size_t i = 0; i < count; ++i) {
        timestamp += 1000000; // 1ms increments
        
        char side = (side_dist(gen) == 0) ? 'B' : 'S';
        auto msg = create_add_order_message(
            timestamp,
            order_id_dist(gen),
            symbols[i % symbols.size()],
            price_dist(gen),
            qty_dist(gen),
            side
        );
        
        data.insert(data.end(), msg.begin(), msg.end());
    }
    
    return data;
}

void ITCHSampleGenerator::write_sample_file(const std::string& path, size_t message_count) {
    auto data = generate_sample_messages(message_count);
    
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
    }
}

std::vector<uint8_t> ITCHSampleGenerator::create_add_order_message(
    uint64_t timestamp, uint64_t order_id, const std::string& symbol,
    uint32_t price, uint32_t quantity, char side) {
    
    std::vector<uint8_t> msg(36); // 2 bytes length + 34 bytes message
    
    // Message length
    msg[0] = 0x00;
    msg[1] = 0x22; // 34 bytes
    
    // Message type
    msg[2] = 'A';
    
    // Timestamp (8 bytes, big-endian)
    for (int i = 0; i < 8; ++i) {
        msg[3 + i] = (timestamp >> (56 - i * 8)) & 0xFF;
    }
    
    // Order ID (8 bytes, big-endian)
    for (int i = 0; i < 8; ++i) {
        msg[11 + i] = (order_id >> (56 - i * 8)) & 0xFF;
    }
    
    // Side
    msg[19] = side;
    
    // Quantity (4 bytes, big-endian)
    for (int i = 0; i < 4; ++i) {
        msg[20 + i] = (quantity >> (24 - i * 8)) & 0xFF;
    }
    
    // Symbol (8 bytes)
    memcpy(&msg[24], symbol.c_str(), 8);
    
    // Price (4 bytes, big-endian)
    for (int i = 0; i < 4; ++i) {
        msg[32 + i] = (price >> (24 - i * 8)) & 0xFF;
    }
    
    return msg;
}

std::vector<uint8_t> ITCHSampleGenerator::create_cancel_order_message(
    uint64_t timestamp, uint64_t order_id, uint32_t cancelled_shares) {
    
    std::vector<uint8_t> msg(25); // 2 bytes length + 23 bytes message
    
    // Message length
    msg[0] = 0x00;
    msg[1] = 0x17; // 23 bytes
    
    // Message type
    msg[2] = 'X';
    
    // Timestamp (8 bytes, big-endian)
    for (int i = 0; i < 8; ++i) {
        msg[3 + i] = (timestamp >> (56 - i * 8)) & 0xFF;
    }
    
    // Order ID (8 bytes, big-endian)
    for (int i = 0; i < 8; ++i) {
        msg[11 + i] = (order_id >> (56 - i * 8)) & 0xFF;
    }
    
    // Cancelled shares (4 bytes, big-endian)
    for (int i = 0; i < 4; ++i) {
        msg[19 + i] = (cancelled_shares >> (24 - i * 8)) & 0xFF;
    }
    
    return msg;
}

} // namespace hft