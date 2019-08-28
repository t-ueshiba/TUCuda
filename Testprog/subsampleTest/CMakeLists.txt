project(subsampleTest)

file(GLOB sources *.cc *cu)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUCuda++ TUTools++)
