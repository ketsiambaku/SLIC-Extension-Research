cmake_minimum_required(VERSION 3.12)
project(superpixel_ris CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 👇 OpenCV on your machine
set(OpenCV_DIR "C:/opencv/build")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(superpixel_ris
    src/main.cpp
    ../SuperDuperPixels/src/sdp_slic.cpp
    ../SuperDuperPixels/src/superduperpixel.cpp
)

target_include_directories(superpixel_ris PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../SuperDuperPixels/src
)

target_link_libraries(superpixel_ris PRIVATE
    ${OpenCV_LIBS}
)
