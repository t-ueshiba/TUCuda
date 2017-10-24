/*
 *  $Id$
 */
#include "TU/Profiler.h"
#if 1
#  include "TU/cuda/BoxFilter.h"
#elif 0
#  include "TU/cuda/NewBoxFilter.h"
#else
#  include "TU/cuda/NeoBoxFilter.h"
#endif
#include "TU/cuda/chrono.h"

namespace TU
{
template <class S, class T> void
cudaJob(const Array2<S>& in, Array2<T>& out, size_t winSize)
{
    cuda::BoxFilter2<T, 15>	cudaFilter(winSize, winSize);
    cuda::Array2<S>		in_d(in.nrow(), in.ncol());
    in_d = in;
    cuda::Array2<T>		out_d(in_d.nrow(), in_d.ncol());
    cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
    cudaThreadSynchronize();

    Profiler<cuda::clock>	cudaProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	cudaFilter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin());
	cudaProfiler.nextFrame();
    }
    cudaProfiler.print(std::cerr);

    out = out_d;
}

template void
cudaJob(const Array2<float>& in, Array2<float>& out, size_t winSize)	;
template void
cudaJob(const Array2<u_char>& in, Array2<short>& out, size_t winSize)	;
    
}
