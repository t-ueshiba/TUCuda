/*
 *  $Id$
 */
#include "TU/Profiler.h"
#include "TU/cu/BoxFilter.h"
#include "TU/cu/StereoUtility.h"
#include "TU/cu/functional.h"
#include "TU/cu/chrono.h"

namespace TU
{
template <class T, class S> void
cudaJob(const Array2<T>& imageL, const Array2<T>& imageR, Array3<S>& costs,
	size_t winSize, size_t disparitySearchWidth, size_t intensityDiffMax)
{
  // スコアを計算する．
    cu::Array2<T>	imageL_d(imageL), imageR_d(imageR);
    cu::BoxFilter2<cu::device::box_convolver<S>,
		   cu::BlockTraits<>, 23, cu::clock>
			boxFilter(winSize, winSize);
#ifdef DISPARITY_MAJOR
    cu::Array3<S>	costs_d(disparitySearchWidth,
    				imageL_d.nrow(), imageL_d.ncol());
#else
    cu::Array3<S>	costs_d(imageL_d.nrow(), imageL_d.ncol(),
    				disparitySearchWidth);
#endif
    boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
		       imageR_d.cbegin(), costs_d.begin(),
		       cu::diff<T>(50), disparitySearchWidth);
    cudaDeviceSynchronize();
#if 1
    Profiler<cu::clock>	cudaProfiler(1);
    constexpr size_t	NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			   imageR_d.cbegin(), costs_d.begin(),
			   cu::diff<T>(intensityDiffMax),
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
    cu::Array2<T>	imageL_d(imageL), imageR_d(imageR);
    cu::BoxFilter2<cu::device::box_convolver<S>,
		   cu::BlockTraits<>, 23, std::chrono::system_clock>
			boxFilter(winSize, winSize);
    cu::Array3<S>	costs_d(disparitySearchWidth,
    				imageL_d.nrow(), imageL_d.ncol());
    // cu::Array3<S>	costs_d(disparitySearchWidth,
    // 				boxFilter.outSizeV(imageL_d.nrow()),
    // 				boxFilter.outSizeH(imageL_d.ncol()));

    boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
		       imageR_d.cbegin(), costs_d.begin(),
		       cu::diff<T>(50), disparitySearchWidth);

    cu::DisparitySelector<S>
			disparitySelector(disparityMax, disparityInconsistency);
    cu::Array2<S>	imageD_d(imageL_d.nrow(), imageL_d.ncol());
    auto		rowD = make_range_iterator(
				   imageD_d[boxFilter.offsetV()].begin()
					  + boxFilter.offsetH(),
				   imageD_d.stride(),
				   imageD_d.ncol() - boxFilter.offsetH());
  //disparitySelector.select(costs_d, rowD);
    cudaDeviceSynchronize();
#if 0
    Profiler<cu::clock>	cudaProfiler(2);
    constexpr size_t	NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	cudaProfiler.start(0);
	boxFilter.convolve(imageL_d.cbegin(), imageL_d.cend(),
			   imageR_d.cbegin(), costs_d.begin(),
			   cu::diff<T>(50), disparitySearchWidth);
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
