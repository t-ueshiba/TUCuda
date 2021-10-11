/*
 *  \file cudaJob.cu
 */
#include "TU/cuda/BoxFilter.h"
#include "TU/cuda/vec.h"
#include "TU/cuda/chrono.h"
#include "TU/Profiler.h"

namespace TU
{
template <class U, class S, class T> void
cudaJob(const Array2<S>& in, Array2<T>& out, size_t winSize, bool shift)
{
    cuda::BoxFilter2<U>	box(winSize, winSize);
    cuda::Array2<U>	in_d(in);
    cuda::Array2<U>	out_d(in_d.nrow(), in_d.ncol());

    box.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(), shift);
    cudaDeviceSynchronize();
    
    Profiler<cuda::clock>	profiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	profiler.start(0);
	box.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(), shift);
	profiler.nextFrame();
    }

    profiler.print(std::cerr);

    out = out_d;
    out *= float(1)/float(winSize*winSize);
}
    
template void
cudaJob<float>(const Array2<u_char>& in, Array2<float>& out,
	       size_t winSize, bool shift);
// template void
// cudaJob<float4>(const Array2<RGBA>& in, Array2<RGBA>& out,
// 		size_t winSize, bool shift);
    
}	// namespace TU
