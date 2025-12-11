cmake_minimum_required(VERSION 3.12)
project(SuperpixelImageSearch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 👇 OpenCV on your machine
set(OpenCV_DIR "C:/opencv/build")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ============================================================
# 1) Main SuperpixelImageSearch program (superpixel_ris)
# ============================================================
add_executable(superpixel_ris
    src/main.cpp
    ../SuperDuperPixels/src/sdp_slic.cpp
    ../SuperDuperPixels/src/superduperpixel.cpp
)

target_include_directories(superpixel_ris PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src                      # COCOLabelIndex.hpp, utils.hpp, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../SuperDuperPixels/src # sdp_slic.hpp, superduperpixel.hpp
)

target_link_libraries(superpixel_ris PRIVATE
    ${OpenCV_LIBS}
)

# ============================================================
# 2) Distance calculation tool (if you still use it)
# ============================================================
add_executable(distance_calculation
    src/distance_calculation.cpp
)

target_include_directories(distance_calculation PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(distance_calculation PRIVATE
    ${OpenCV_LIBS}
)

# ============================================================
# 3) Pipeline demo executable (CUSTOM_SDSLIC_SIFT only)
# ============================================================
add_executable(pipeline_demo
    src/pipeline_demo.cpp
    ../SuperDuperPixels/src/sdp_slic.cpp
    ../SuperDuperPixels/src/superduperpixel.cpp
)

target_include_directories(pipeline_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../SuperDuperPixels/src
)

target_link_libraries(pipeline_demo PRIVATE
    ${OpenCV_LIBS}
)
