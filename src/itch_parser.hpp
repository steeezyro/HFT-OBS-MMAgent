#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hft {

struct ITCHMessage {
    uint8_t message_type;
    uint64_t timestamp;
    uint64_t order_id;
    char symbol[8];
    uint32_t price;
    uint32_t quantity;
    char side;
    
    // Additional fields for different message types
    uint64_t match_number;
    uint32_t executed_quantity;
    char printable;
};

enum class ITCHMessageType : uint8_t {
    SYSTEM_EVENT = 'S',
    STOCK_DIRECTORY = 'R',
    STOCK_TRADING_ACTION = 'H',
    REG_SHO_RESTRICTION = 'Y',
    MARKET_PARTICIPANT_POSITION = 'L',
    MWCB_DECLINE_LEVEL = 'V',
    MWCB_STATUS = 'W',
    PRICE_LEVEL_UPDATE = 'P',
    ADD_ORDER = 'A',
    ADD_ORDER_MPID = 'F',
    ORDER_EXECUTED = 'E',
    ORDER_EXECUTED_WITH_PRICE = 'C',
    ORDER_CANCEL = 'X',
    ORDER_DELETE = 'D',
    ORDER_REPLACE = 'U',
    TRADE = 'P',
    CROSS_TRADE = 'Q',
    BROKEN_TRADE = 'B',
    NOII = 'I'
};

class ITCHParser {
public:
    using MessageCallback = std::function<void(const ITCHMessage&)>;
    
    explicit ITCHParser(const std::string& file_path);
    ~ITCHParser();
    
    bool parse(MessageCallback callback, const std::string& symbol_filter = "");
    bool parse_range(MessageCallback callback, size_t start_offset, size_t end_offset, 
                     const std::string& symbol_filter = "");
    
    size_t file_size() const { return file_size_; }
    size_t messages_parsed() const { return messages_parsed_; }
    double parse_rate_mbps() const { return parse_rate_mbps_; }
    
    // Static utility functions
    static uint16_t parse_uint16(const uint8_t* data);
    static uint32_t parse_uint32(const uint8_t* data);
    static uint64_t parse_uint64(const uint8_t* data);
    static double parse_price(uint32_t raw_price);
    
private:
    bool map_file();
    void unmap_file();
    bool parse_message(const uint8_t* data, size_t offset, ITCHMessage& msg);
    
    std::string file_path_;
    uint8_t* mapped_data_;
    size_t file_size_;
    int fd_;
    
    // Statistics
    size_t messages_parsed_;
    double parse_rate_mbps_;
    
    // Message parsers
    bool parse_add_order(const uint8_t* data, ITCHMessage& msg);
    bool parse_add_order_mpid(const uint8_t* data, ITCHMessage& msg);
    bool parse_order_executed(const uint8_t* data, ITCHMessage& msg);
    bool parse_order_executed_with_price(const uint8_t* data, ITCHMessage& msg);
    bool parse_order_cancel(const uint8_t* data, ITCHMessage& msg);
    bool parse_order_delete(const uint8_t* data, ITCHMessage& msg);
    bool parse_order_replace(const uint8_t* data, ITCHMessage& msg);
};

// Helper class for creating sample ITCH data for testing
class ITCHSampleGenerator {
public:
    static std::vector<uint8_t> generate_sample_messages(size_t count = 1000);
    static void write_sample_file(const std::string& path, size_t message_count = 10000);
    
private:
    static std::vector<uint8_t> create_add_order_message(uint64_t timestamp, uint64_t order_id,
                                                         const std::string& symbol, uint32_t price,
                                                         uint32_t quantity, char side);
    static std::vector<uint8_t> create_cancel_order_message(uint64_t timestamp, uint64_t order_id,
                                                            uint32_t cancelled_shares);
};

} // namespace hft