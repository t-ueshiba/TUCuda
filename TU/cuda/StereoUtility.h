/*!
  \file		StereoUtility.h
  \author	Toshio UESHIBA
  \brief	ステレオビジョンをサポートする各種クラスの定義と実装
*/
#pragma once

#include <limits>
#include "TU/cuda/Array++.h"

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <class COL, class COL_C, class COL_D> __global__ void
select_disparity(COL colC, int width, COL_D colD,
	    int disparitySearchWidth, int disparityMax,
	    int disparityInconsistency,
	    int strideX, int strideYX, int strideD,
	    COL_C cminR, int strideCminR, int* dminR, int strideDminR)
{
    using value_type	= typename std::iterator_traits<COL>::value_type;

    const auto	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const auto	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    colC  += (__mul24(y, strideX) + x);		// 左画像のx座標
    colD  += (__mul24(y, strideD) + x);		// 視差画像のx座標
    cminR +=  __mul24(y, strideCminR);
    dminR +=  __mul24(y, strideDminR);

    value_type	cminL = std::numeric_limits<value_type>::max();
    int		dminL = 0;

    cminR[x] = std::numeric_limits<value_type>::max();
    dminR[x] = 0;
    __syncthreads();

    for (int d = 0; d < disparitySearchWidth; ++d)
    {
	const auto	cost = *colC;	// cost[d][y][x]

	colC += strideYX;		// points to cost[d+1][y][x]

	if (cost < cminL)
	{
	    cminL = cost;
	    dminL = d;
	}

	const auto	xR = x + d;

	if (xR < width)
	{
	    if (cost < cminR[xR])
	    {
		cminR[xR] = cost;
		dminR[xR] = d;
	    }
	}
	__syncthreads();
    }

    const auto	dR = dminR[x + dminL];

    *colD = ((dminL > dR ? dminL - dR : dR - dminL) < disparityInconsistency ?
	     disparityMax - dminL : 0);
}

}	// namespace device
#endif

/************************************************************************
*  class DisparitySelector<T>						*
************************************************************************/
template <class T>
class DisparitySelector
{
  public:
    using value_type	= T;

    constexpr static size_t	BlockDimX = 32;
    constexpr static size_t	BlockDimY = 16;

  public:
    DisparitySelector(int disparityMax, int disparityInconsistency)
	:_disparityMax(disparityMax),
	 _disparityInconsistency(disparityInconsistency)		{}

    template <class ROW_D>
    void	select(const Array3<T>& costs, ROW_D rowD)		;

  private:
    const int	_disparityMax;
    const int	_disparityInconsistency;
    Array2<T>	_cminR;			//!< 右画像から見た最小コスト
    Array2<int>	_dminR;			//!< 右画像から見た最小コストを与える視差
};

template <class T> template <class ROW_D> void
DisparitySelector<T>::select(const Array3<T>& costs, ROW_D rowD)
{
    const auto	disparitySearchWidth = size<0>(costs);
    const auto	height		     = size<1>(costs);
    const auto	width		     = size<2>(costs);
    const auto	strideX		     = costs.stride();
    const auto	strideYX	     = height * strideX;
    const auto	strideD		     = stride(rowD);

    _cminR.resize(height, width);
    _dminR.resize(height, width);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(width/threads.x, height/threads.y);
    device::select_disparity<<<blocks, threads>>>(get(costs[0][0].cbegin()),
						  width,
						  get(std::begin(*rowD)),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  get(std::begin(_cminR[0])),
						  _cminR.stride(),
						  get(_dminR[0].begin()),
						  _dminR.stride());
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = width - x;
    blocks.x  = 1;
    device::select_disparity<<<blocks, threads>>>(get(costs[0][0].cbegin() + x),
						  width,
						  get(std::begin(*rowD) + x),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  get(std::begin(_cminR[0]) + x),
						  _cminR.stride(),
						  get(_dminR[0].begin() + x),
						  _dminR.stride());
  // 左下
    const auto	y = blocks.y*threads.y;
    std::advance(rowD, y);
    threads.x = BlockDim;
    blocks.x  = width/threads.x;
    threads.y = height - y;
    blocks.y  = 1;
    device::select_disparity<<<blocks, threads>>>(get(costs[0][y].cbegin()),
						  width,
						  get(std::begin(*rowD)),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  get(std::begin(_cminR[y])),
						  _cminR.stride(),
						  get(_dminR[y].begin()),
						  _dminR.stride());
  // 右下
    device::select_disparity<<<blocks, threads>>>(get(costs[0][y].cbegin() + x),
						  width,
						  get(std::begin(*rowD) + x),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  get(std::begin(_cminR[y]) + x),
						  _cminR.stride(),
						  get(_dminR[y].begin() + x),
						  _dminR.stride());
}

}	// namespace cuda
}	// namepsace TU
