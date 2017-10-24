/*
 *  $Id$
 */
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/chrono.h"

namespace TU
{
template <template <class> class OP, class S, class T> void
cudaJob(const Array2<S>& in, Array2<T>& out)
{
  // GPUによって計算する．
    cuda::Array2<S>	in_d(in);
    cuda::Array2<T>	out_d(in.nrow(), in.ncol());
    cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<S>());
    cudaThreadSynchronize();

    Profiler<cuda::clock>	cuProfiler(1);
    constexpr size_t		NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)		// フィルタリング
    {
	cuProfiler.start(0);
	cuda::op3x3(in_d.cbegin(), in_d.cend(), out_d.begin(), OP<S>());
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
	
    out = out_d;
}

template void
cudaJob<cuda::maximal3x3>(const Array2<float>& in, Array2<float>& out)	;
}
