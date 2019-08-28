project(stereoTest)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-relaxed-constexpr")

file(GLOB sources *.cc *cu)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUCuda++ TUTools++)
