# Test executables for SDP_LTriDP SLIC

# Path to ltridp build directory
set(LTRIDP_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ltridp/build)

add_executable(test_complete_pipeline_v2
    test_complete_pipeline_v2.cpp
)

target_link_libraries(test_complete_pipeline_v2
    sdp_ltridp_segmentation
    ${LTRIDP_BUILD_DIR}/preprocessing/libpreprocessing.a
    ${LTRIDP_BUILD_DIR}/feature/libfeature.a
    ${LTRIDP_BUILD_DIR}/evaluation_build/libsuperpixel_evaluator.a
    ${OpenCV_LIBS}
)

target_include_directories(test_complete_pipeline_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ltridp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ltridp/segmentation
)
