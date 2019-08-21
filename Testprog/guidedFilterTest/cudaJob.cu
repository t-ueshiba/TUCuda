/*
 *  $Id$
 */
#include "TU/Profiler.h"
#include "TU/cuda/chrono.h"
#include "TU/cuda/GuidedFilter.h"

namespace TU
{
template <class S, class T, class U> void
cudaJob(const Array2<S>& in, const Array2<S>& guide,
	Array2<T>& out, size_t winSize, U epsilon)
{
    cuda::GuidedFilter2<U, cuda::clock>
				cudaFilter(winSize, winSize, epsilon);
    cuda::Array2<U>		in_d(in);
    cuda::Array2<U>		guide_d(guide);
    cuda::Array2<U>		out_d(in_d.nrow(), in_d.ncol());

    cudaFilter.convolve(in_d.cbegin(), in_d.cend(),
			guide_d.cbegin(), guide_d.cend(), out_d.begin());
    cudaDeviceSynchronize();

    Profiler<cuda::clock>	cudaProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	cudaFilter.convolve(in_d.cbegin(), in_d.cend(),
			    guide_d.cbegin(), guide_d.cend(), out_d.begin());
	cudaProfiler.nextFrame();
    }
    cudaProfiler.print(std::cerr);
    cudaFilter.print(std::cerr);
    
    out = out_d;
}

template void
cudaJob(const Array2<u_char>& in, const Array2<u_char>& guide,
	Array2<float>& out, size_t winSize, float epsilon)		;
  /*
template void
cudaJob<float4>(const Array2<RGBA>& in, Array2<RGBA>& out, size_t winSize);
  */
}	// namespace TU
