project(cudaStereoTest)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-relaxed-constexpr")

file(GLOB sources *.cc *cu)
cuda_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUCuda)
