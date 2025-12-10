cmake_minimum_required(VERSION 3.10)

project(SDP_LTriDP_SLIC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs ximgproc)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Build segmentation library
add_library(sdp_ltridp_segmentation STATIC
    src/slic.cpp
)

target_link_libraries(sdp_ltridp_segmentation
    ${OpenCV_LIBS}
)

# Tests subdirectory
add_subdirectory(tests)
