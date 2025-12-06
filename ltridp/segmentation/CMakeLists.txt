# Segmentation module CMakeLists.txt
# LTriDP-enhanced SLIC superpixel segmentation

cmake_minimum_required(VERSION 3.10)

# Module source files
set(SEGMENTATION_SOURCES
    slic.cpp
)

# Create segmentation library
add_library(ltridp_segmentation ${SEGMENTATION_SOURCES})

# Include directories
target_include_directories(ltridp_segmentation
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link OpenCV
target_link_libraries(ltridp_segmentation
    PUBLIC
        ${OpenCV_LIBS}
)

# Set C++ standard
set_target_properties(ltridp_segmentation PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)
