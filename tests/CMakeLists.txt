cmake_minimum_required(VERSION 3.20)

# Test configuration
enable_testing()

# Find required packages
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Test executables
add_executable(order_book_test
    order_book_test.cpp
)

target_link_libraries(order_book_test
    hft_core
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_executable(itch_parser_test
    itch_parser_test.cpp
)

target_link_libraries(itch_parser_test
    hft_core
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

add_executable(bench_orderbook
    bench_orderbook.cpp
)

target_link_libraries(bench_orderbook
    hft_core
    Threads::Threads
)

# Register tests
add_test(NAME OrderBookTest COMMAND order_book_test)
add_test(NAME ITCHParserTest COMMAND itch_parser_test)

# Performance test (not run by default)
add_test(NAME BenchmarkOrderBook COMMAND bench_orderbook)