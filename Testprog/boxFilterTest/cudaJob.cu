/*
 *  $Id$
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/BoxFilter.h"
#include "TU/cuda/chrono.h"

namespace TU
{
template <class S, class T> void
cudaJob(const Image<S>& in, Image<T>& out, size_t winSize)
{
    cuda::BoxFilter2<T, 15>	cudaFilter(winSize, winSize);
    cuda::Array2<S>		in_d(in);
    cuda::Array2<T>		out_d(in_d.nrow(), in_d.ncol());
    cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
    cudaThreadSynchronize();

    Profiler<cuda::clock>	cuProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);

    out_d.write(out);
}

template void
cudaJob(const Image<float>& in, Image<float>& out, size_t winSize)	;
    
}
