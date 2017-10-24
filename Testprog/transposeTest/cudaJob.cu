/*
 *  $Id$
 */
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/chrono.h"

namespace TU
{
template <class S, class T> void
cudaJob(const Array2<S>& in, Array2<T>& out)
{
  // GPUによって計算する．
    cuda::Array2<S>	in_d(in);
    cuda::Array2<T>	out_d(in_d.ncol(), in_d.nrow());

    cuda::transpose(in_d.cbegin(), in_d.cend(), out_d.begin());
    cudaThreadSynchronize();

    Profiler<cuda::clock>	cuProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	cuda::transpose(in_d.cbegin(), in_d.cend(), out_d.begin());
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
	
    out = out_d;
}

template void	cudaJob(const Array2<float>& in, Array2<float>& out)	;
}
