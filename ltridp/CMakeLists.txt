cmake_minimum_required(VERSION 3.10)
project(SLIC_Extension_Research VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(preprocessing)
add_subdirectory(feature)
add_subdirectory(segmentation)

# Enable testing (will be used if GTest is available in tests/)
enable_testing()
add_subdirectory(tests)