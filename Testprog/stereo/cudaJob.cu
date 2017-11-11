/*
 *  $Id$
 */
#include "TU/Profiler.h"
#if 1
#  include "TU/cuda/BoxFilter.h"
#else
#  include "TU/cuda/NewBoxFilter.h"
#endif
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"

namespace TU
{
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array3<S>& scores,
	size_t winSize, size_t disparitySearchWidth)
{
  // スコアを計算する．
    cuda::BoxFilter2<S, 20>	cudaFilter(winSize, winSize);
    cuda::Array2<S>		imageL_d(imageL), imageR_d(imageR);
    cuda::Array3<S>		scores_d(32, imageL_d.nrow(), imageL_d.ncol(),
					 disparitySearchWidth);
    cudaFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			imageR_d.cbegin(), scores_d.begin(),
			cuda::diff<T>(50), disparitySearchWidth);
    cudaThreadSynchronize();
#if 1
    Profiler<cuda::clock>	cudaProfiler(1);
    constexpr size_t		NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	cudaFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			    imageR_d.cbegin(), scores_d.begin(),
			    cuda::diff<T>(50), disparitySearchWidth);
	cudaProfiler.nextFrame();
    }
    cudaProfiler.print(std::cerr);
#endif
    
    scores = scores_d;;
}

template void
cudaJob(const Array2<u_char>& imageL, const Array2<u_char>& imageR,
	Array3<short>& out, size_t winSize, size_t disparitySearchWidth);
    
template void
cudaJob(const Array2<u_char>& imageL, const Array2<u_char>& imageR,
	Array3<float>& out, size_t winSize, size_t disparitySearchWidth);
    
}
