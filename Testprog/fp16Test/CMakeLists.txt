project(fp16Test)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_53 -Xcompiler -mf16c")

file(GLOB sources *.cc *cu)
cuda_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUCuda)
