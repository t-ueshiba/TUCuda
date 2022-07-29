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
  \file		Morphology.h
  \brief	boxフィルタの定義と実装
*/
#pragma once

#include <cub/block/block_scan.cuh>
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/vec.h"

namespace TU
{
namespace cu
{
#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __device__ functions							*
************************************************************************/

/************************************************************************
*  __global__ functions							*
************************************************************************/
//! スレッドブロックの横方向にモルフォロジーフィルタを適用する
/*!
  \param in		入力2次元配列
  \param out		出力2次元配列
  \param op		結合葎を満たす2項演算子
  \param null_val	任意の値valに対して op(null_val, val) = val となる値
  \param winRadius	ウィンドウの半径
*/
template <class FILTER, class IN, class OUT, class OP, class T>
__global__ void
morphology(range<range_iterator<IN> > in,
	   range<range_iterator<OUT> > out, OP op, T null_val, int winRadius)
{

    using value_type	= T;
    using BlockScan	= cub::BlockScan<value_type, FILTER::BlockDimX,
					 cub::BLOCK_SCAN_RAKING,
					 FILTER::BlockDimY>;
    using TempStorage	= typename BlockScan::TempStorage;

    const int	ncol = in.cbegin().size();		 // 画像幅
    const int	x0   = __mul24(blockIdx.x, blockDim.x);  // ブロック左端
    const int	y0   = __mul24(blockIdx.y, blockDim.y);  // ブロック上端
    const int	xorg = ::max(x0 - winRadius, 0);
    const int	xsiz = ::min(int(blockDim.x + 2*winRadius), ncol - xorg);
    const int	ysiz = ::min(int(blockDim.y), in.size() - y0);

    __shared__ value_type in_s[FILTER::BlockDimY]
			      [FILTER::BlockDimX + 2*FILTER::WinRadiusMax];
    loadTile(slice(in.cbegin(), y0, ysiz, xorg, xsiz), in_s);
    __syncthreads();

    const int	x = x0 + threadIdx.x;
    const int	y = y0 + threadIdx.y;
    if (y >= in.size() || x >= ncol)
	return;

    __shared__ value_type	r[FILTER::BlockDimY]
				 [FILTER::BlockDimX + 2*FILTER::WinRadiusMax];
    __shared__ TempStorage	tmp;

    const int	x1 = ::min(x0 + blockDim.x, ncol);	// ブロック右端
    for (int w0 = x0; w0 < x1; )
    {
	const int	w1   = w0 + 2*winRadius + 1;
	const int	xr   = 2*w0 + winRadius - x;
	value_type	rval = null_val;
	value_type	sval = null_val;

	if (w0 <= x && x < w1)	// xがwindow内にあるか？
	{
	    if (xorg <= xr && xr < ncol)
		rval = in_s[y - y0][xr - xorg];

	    if (x + winRadius < ncol)
		sval = in_s[y - y0][x + winRadius - xorg];
	}

	BlockScan(tmp).InclusiveScan(rval, r[y - y0][xr + winRadius - xorg],
				     op);
	value_type	s;
	BlockScan(tmp).InclusiveScan(sval, s, op);
	__syncthreads();
	printf("[(%d,%d) x=%d in [%d, %d) x'=%d]: sval=%f, s=%f, rval=%f, r=%f\n",
	       threadIdx.x, threadIdx.y, x, w0, w1, xr,
	       sval, s, rval, r[y - y0][x - xorg]);

	if (w0 <= x && x < w1)	// xがwindow内にあるか？
	    out[y][x] = op(r[y - y0][x - xorg], s);

	w0 = w1;
    }
}

}	// namespace device
#endif	// __NVCC__

/************************************************************************
*  class Morphology<T, BLOCK_TRAITS, WMAX, CLOCK>			*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class T, class BLOCK_TRAITS=BlockTraits<128, 4>, class CLOCK=void>
class Morphology : public BLOCK_TRAITS, public Profiler<CLOCK>
{
  private:
    using profiler_t	= Profiler<CLOCK>;

  public:
    using value_type	= T;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
    constexpr static size_t	WinRadiusMax = (BlockDimX - 1)/2;

  public:
		Morphology(size_t winRadiusV, size_t winRadiusH)
		    :profiler_t(3)
		{
		    setWinSizeV(winRadiusV);
		    setWinSizeH(winRadiusH);
		}

    size_t	winRadiusV()		const	{ return _winRadiusV; }
    size_t	winRadiusH()		const	{ return _winRadiusH; }
    Morphology&	setWinSizeV(size_t winRadius)
		{
		    if (winRadius > WinRadiusMax)
			throw std::runtime_error("Too large window radius!");
		    _winRadiusV = winRadius;
		    return *this;
		}
    Morphology&	setWinSizeH(size_t winRadius)
		{
		    if (winRadius > WinRadiusMax)
			throw std::runtime_error("Too large window radius!");
		    _winRadiusH = winRadius;
		    return *this;
		}

    template <class ROW, class ROW_O, class OP>
    void	apply(ROW row, ROW rowe,
		      ROW_O rowO, OP op, T null_val)		const	;

    template <class ROW, class ROW_O, class OP>
    void	apply_debug(ROW row, ROW rowe,
			    ROW_O rowO, OP op, T null_val)	const	;

  private:
    size_t			_winRadiusV;
    size_t			_winRadiusH;
    mutable Array2<value_type>	_buf;
};

#if defined(__NVCC__)
template <class T, class BOX_TRAITS, class CLOCK>
template <class ROW, class ROW_O, class OP> void
Morphology<T, BOX_TRAITS, CLOCK>::apply(ROW row, ROW rowe, ROW_O rowO,
					OP op, T null_val) const
{
    std::cerr << "OK0" << std::endl;
    profiler_t::start(0);

    const size_t nrows = std::distance(row, rowe);
    const size_t ncols = std::size(*row);

    _buf.resize(ncols, nrows);

  // Apply filter in horizontal direction.
    std::cerr << "OK1" << std::endl;
    profiler_t::start(1);
    const dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(divUp(ncols, threads.x), divUp(nrows, threads.y));
    device::morphology<Morphology><<<blocks, threads>>>(
	cu::make_range(row,  nrows),
	cu::make_range(_buf.begin(), _buf.nrow()), op, null_val, _winRadiusH);
    gpuCheckLastError();

  // Apply filter in horizontal direction.
    std::cerr << "OK2" << std::endl;
    profiler_t::start(2);
    blocks.x = divUp(nrows, threads.x);
    blocks.y = divUp(ncols, threads.y);
    device::morphology<Morphology><<<blocks, threads>>>(
	cu::make_range(_buf.cbegin(), _buf.nrow()),
	cu::make_range(rowO, nrows), op, null_val, _winRadiusV);
    gpuCheckLastError();

    std::cerr << "OK3" << std::endl;
    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}

template <class T, class BOX_TRAITS, class CLOCK>
template <class ROW, class ROW_O, class OP> void
Morphology<T, BOX_TRAITS, CLOCK>::apply_debug(ROW row, ROW rowe, ROW_O rowO,
					      OP op, T null_val) const
{
    profiler_t::start(0);

    const size_t nrows = std::distance(row, rowe);
    const size_t ncols = std::size(*row);

  // Apply filter in horizontal direction.
    profiler_t::start(1);
    const dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(divUp(ncols, threads.x), divUp(nrows, threads.y));
    device::morphology<Morphology><<<blocks, threads>>>(
	cu::make_range(row,  nrows),
	cu::make_range(rowO, nrows), op, null_val, _winRadiusH);
    gpuCheckLastError();

    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}

#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
