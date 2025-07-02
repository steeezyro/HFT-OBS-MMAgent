cmake_minimum_required(VERSION 3.20)
project(HFT_OrderBook_RL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Performance flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
endif()

# Include FetchContent for dependencies
include(cmake/FetchContent.cmake)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(include)

# Create main library
add_library(hft_core
    src/order_book.cpp
    src/itch_parser.cpp
)

target_include_directories(hft_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hft_core PUBLIC
    Threads::Threads
)

# Executables
add_executable(replay src/replay.cpp)
target_link_libraries(replay hft_core)

# Skip ZMQ bridge for now to avoid dependency issues
# add_executable(zmq_bridge src/zmq_bridge.cpp)
# target_link_libraries(zmq_bridge hft_core)

# Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
add_executable(bench_orderbook tests/bench_orderbook.cpp)
target_link_libraries(bench_orderbook hft_core)

# Custom targets
add_custom_target(bench
    COMMAND ./bench_orderbook
    DEPENDS bench_orderbook
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance benchmarks"
)

add_custom_target(report
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python python/hft_rl/report.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating performance report"
)

# Installation
install(TARGETS hft_core replay zmq_bridge
    EXPORT HFTOrderBookTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)