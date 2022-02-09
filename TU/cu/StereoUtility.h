// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		StereoUtility.h
  \author	Toshio UESHIBA
  \brief	ステレオビジョンをサポートする各種クラスの定義と実装
*/
#pragma once

#include <limits>
#include "TU/cu/Array++.h"

namespace TU
{
namespace cu
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
*  class DisparitySelector<T, BLOCK_TRAITS>				*
************************************************************************/
template <class T, class BLOCK_TRAITS=BlockTraits<> >
class DisparitySelector : public BLOCK_TRAITS
{
  public:
    using value_type	= T;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
    constexpr static size_t	BlockDim = BlockDimY;

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

template <class T, class BLOCK_TRAITS> template <class ROW_D> void
DisparitySelector<T, BLOCK_TRAITS>::select(const Array3<T>& costs, ROW_D rowD)
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
    device::select_disparity<<<blocks, threads>>>(costs[0][0].cbegin(),
						  width,
						  std::begin(*rowD),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  std::begin(_cminR[0]),
						  _cminR.stride(),
						  _dminR[0].begin(),
						  _dminR.stride());
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = width - x;
    blocks.x  = 1;
    device::select_disparity<<<blocks, threads>>>(costs[0][0].cbegin() + x,
						  width,
						  std::begin(*rowD) + x,
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  std::begin(_cminR[0]) + x,
						  _cminR.stride(),
						  _dminR[0].begin() + x,
						  _dminR.stride());
  // 左下
    const auto	y = blocks.y*threads.y;
    std::advance(rowD, y);
    threads.x = BlockDim;
    blocks.x  = width/threads.x;
    threads.y = height - y;
    blocks.y  = 1;
    device::select_disparity<<<blocks, threads>>>(costs[0][y].cbegin(),
						  width,
						  std::begin(*rowD),
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  std::begin(_cminR[y]),
						  _cminR.stride(),
						  _dminR[y].begin(),
						  _dminR.stride());
  // 右下
    device::select_disparity<<<blocks, threads>>>(costs[0][y].cbegin() + x,
						  width,
						  std::begin(*rowD) + x,
						  disparitySearchWidth,
						  _disparityMax,
						  _disparityInconsistency,
						  strideX, strideYX, strideD,
						  std::begin(_cminR[y]) + x,
						  _cminR.stride(),
						  _dminR[y].begin() + x,
						  _dminR.stride());
}

}	// namespace cu
}	// namepsace TU
