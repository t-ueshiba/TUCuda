/*
 *  $Id$
 */
#include "TU/Profiler.h"
#include "TU/cuda/BoxFilter.h"
#include "TU/cuda/StereoUtility.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"

namespace TU
{
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array3<S>& costs,
	size_t winSize, size_t disparitySearchWidth, size_t intensityDiffMax)
{
  // スコアを計算する．
    cuda::Array2<T>	imageL_d(imageL), imageR_d(imageR);
    cuda::BoxFilter2<S, cuda::clock>
			boxFilter(winSize, winSize);
#ifdef DISPARITY_MAJOR
    cuda::Array3<S>	costs_d(disparitySearchWidth,
    				imageL_d.nrow(), imageL_d.ncol());
#else
    cuda::Array3<S>	costs_d(imageL_d.nrow(), imageL_d.ncol(),
    				disparitySearchWidth);
#endif
    boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
		       imageR_d.cbegin(), costs_d.begin(),
		       cuda::diff<T>(50), disparitySearchWidth);
    cudaDeviceSynchronize();
#if 1
    Profiler<cuda::clock>	cudaProfiler(1);
    constexpr size_t		NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			   imageR_d.cbegin(), costs_d.begin(),
			   cuda::diff<T>(intensityDiffMax),
			   disparitySearchWidth);
	cudaProfiler.nextFrame();
    }
    cudaProfiler.print(std::cerr);
    boxFilter.print(std::cerr);
#endif
    costs = costs_d;
}
    
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array2<S>& imageD,
	size_t winSize, size_t disparitySearchWidth, size_t disparityMax,
	size_t intensityDiffMax, size_t disparityInconsistency)
{
  // スコアを計算する．
    cuda::Array2<T>	imageL_d(imageL), imageR_d(imageR);
    cuda::BoxFilter2<S, std::chrono::system_clock>
			boxFilter(winSize, winSize);
    cuda::Array3<S>	costs_d(disparitySearchWidth,
    				imageL_d.nrow(), imageL_d.ncol());
    // cuda::Array3<S>	costs_d(disparitySearchWidth,
    // 				boxFilter.outSizeV(imageL_d.nrow()),
    // 				boxFilter.outSizeH(imageL_d.ncol()));

    boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
		       imageR_d.cbegin(), costs_d.begin(),
		       cuda::diff<T>(50), disparitySearchWidth);

    cuda::DisparitySelector<S>
			disparitySelector(disparityMax, disparityInconsistency);
    cuda::Array2<S>	imageD_d(imageL_d.nrow(), imageL_d.ncol());
    auto		rowD = make_range_iterator(
				   imageD_d[boxFilter.offsetV()].begin()
					  + boxFilter.offsetH(),
				   imageD_d.stride(),
				   imageD_d.ncol() - boxFilter.offsetH());
  //disparitySelector.select(costs_d, rowD);
    cudaDeviceSynchronize();
#if 0
    Profiler<cuda::clock>	cudaProfiler(2);
    constexpr size_t		NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			   imageR_d.cbegin(), costs_d.begin(),
			   cuda::diff<T>(50), disparitySearchWidth);
	cudaDeviceSynchronize();

	cudaProfiler.start(1);
	disparitySelector.select(costs_d, rowD);
	cudaDeviceSynchronize();

	cudaProfiler.nextFrame();
    }
    cudaProfiler.print(std::cerr);
    boxFilter.print(std::cerr);
#endif
    imageD = imageD_d;
}
    
template void
cudaJob(const Array2<u_char>& imageL,
	const Array2<u_char>& imageR, Array3<float>& costs,
	size_t winSize, size_t disparitySearchWidth, size_t intensityDiffMax);
    
template void
cudaJob(const Array2<u_char>& imageL,
	const Array2<u_char>& imageR, Array2<float>& imageD,
	size_t winSize, size_t disparitySearchWidth, size_t disparityMax,
	size_t intensityDiffMax, size_t disparityInconsistency);
    
}
