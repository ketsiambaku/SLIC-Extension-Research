# Tests CMakeLists.txt
# Unit tests and sample testing programs

cmake_minimum_required(VERSION 3.10)

# Find Google Test (optional, if not found, unit tests will be skipped)
find_package(GTest)

if(GTest_FOUND)
    message(STATUS "Google Test found - building unit tests")
    
    include_directories(${GTEST_INCLUDE_DIRS})
    
    # Unit tests executable
    add_executable(test_preprocessing test_preprocessing.cpp)
    target_link_libraries(test_preprocessing 
        preprocessing 
        ${OpenCV_LIBS}
        GTest::gtest 
        GTest::gtest_main
    )
        add_test(NAME PreprocessingUnitTests COMMAND test_preprocessing)
else()
    message(STATUS "Google Test not found - skipping unit tests")
endif()

# MRI sample testing program
add_executable(test_with_mri_samples test_with_mri_samples.cpp)
target_link_libraries(test_with_mri_samples 
    preprocessing 
    ${OpenCV_LIBS}
)

# Complete pipeline test program
add_executable(test_complete_pipeline test_complete_pipeline.cpp)
target_link_libraries(test_complete_pipeline
    preprocessing
    feature
    ltridp_segmentation
    ${OpenCV_LIBS}
)