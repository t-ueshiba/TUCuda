project(labelingTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB sources *.cc *cu)
cuda_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUCuda ${OpenCV_LIBRARIES})
