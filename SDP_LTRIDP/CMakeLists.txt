cmake_minimum_required(VERSION 3.10)

project(SuperDuperPixels)

set(OpenCV_DIR C:/Libraries/opencv/contrib_release/install)

# Request the core components we use. If your OpenCV was built with contrib
# and you prefer to use the installed ximgproc module instead of copying
# the implementation here, add `ximgproc` to the COMPONENTS list.
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/slic.cpp src/superduperpixel.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
