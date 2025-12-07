cmake_minimum_required(VERSION 3.10)
project(SuperpixelEvaluator VERSION 1.0 LANGUAGES CXX)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Create evaluator library
add_library(superpixel_evaluator STATIC
    evaluator.cpp
)

target_include_directories(superpixel_evaluator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(superpixel_evaluator PUBLIC
    ${OpenCV_LIBS}
)

# Set C++ standard
target_compile_features(superpixel_evaluator PUBLIC cxx_std_11)
