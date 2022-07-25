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

    const int	ncol = in.cbegin().size();
    const int	x0   = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0   = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xorg = ::max(x0 - winRadius, 0);
    const int	xsiz = ::min(int(blockDim.x + winRadius), ncol - xorg);
    const int	ysiz = ::min(int(blockDim.y), in.size() - y0);

    __shared__ value_type in_s[FILTER::BlockDimY]
			      [FILTER::BlockDimX + 2*FILTER::WinRadiusMax];
    loadTile(slice(in.cbegin(), y0, ysiz, xorg, xsiz), in_s);
    __syncthreads();

    const int	x = x0 + threadIdx.x;
    const int	y = y0 + threadIdx.y;
    if (y >= in.size() || x >= ncol)
	return;

    __shared__ value_type	out_s[FILTER::BlockDim][FILTER::BlockDim + 1];
    __shared__ TempStorage	tmp;

    const int	x1 = ::min(x0 + blockDim.x, ncol);
    for (int wx0 = x0; wx0 < x1; )
    {
	const int	wx1  = ::min(wx0 + 2*winRadius + 1, ncol);
	value_type	rval = null_val;
	value_type	sval = null_val;

	if (wx0 <= x && x < wx1)	// xがwindow内にあるか？
	{
	    if (wx0 + winRadius < ncol && x + winRadius < ncol)
		sval = in_s[x + winRadius - xorg];

	    if (winRadius <= wx0 && winRadius <= x)
		rval =;
	}

	value_type	r;
	BlockScan(tmp).InclusiveScan(rval, r, op);
	value_type	s;
	BlockScan(tmp).InclusiveScan(sval, s, op);

	if (wx0 <= x && x < wx1)	// xがwindow内にあるか？
	    out_s[y][x] = op(r, s);

	wx0 = wx1;
    }

    __syncthreads();

    const int	ye = ysiz - winSize + 1;

    if (threadIdx.x < xsiz && ye > 0)
    {
	if (threadIdx.y == 0)		// 各列を並列に縦方向積算
	{
	    const convolver_type	convolve;
	    convolve(in_s, out_s, winSize, ye);
	}
	__syncthreads();

      // 結果を格納
	if (xsiz == blockDim.x)
	{
	    if (threadIdx.x < ye)
		out[x0 + threadIdx.y][y0 + threadIdx.x]
		    = out_s[threadIdx.x][threadIdx.y];
	}
	else
	{
	    if (threadIdx.y < ye)
		out[x][y] = out_s[threadIdx.y][threadIdx.x];
	}
    }
}

}	// namespace device
#endif	// __NVCC__

/************************************************************************
*  class Morphology<CONVOLVER, BLOCK_TRAITS, WMAX, CLOCK>		*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class CONVOLVER=device::box_convolver<float>,
	  class BLOCK_TRAITS=BlockTraits<>, size_t WMAX=23, class CLOCK=void>
class Morphology : public BLOCK_TRAITS, public Profiler<CLOCK>
{
  private:
    using profiler_t	= Profiler<CLOCK>;

  public:
    using convolver_type= CONVOLVER;
    using value_type	= typename convolver_type::value_type;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
    constexpr static size_t	BlockDim   = BlockDimY;
    constexpr static size_t	WinSizeMax = WMAX;

  public:
  //! CUDAによる2次元boxフィルタを生成する．
  /*!
    \param winRadiusV	boxフィルタのウィンドウの行幅(高さ)
    \param winRadiusH	boxフィルタのウィンドウの列幅(幅)
   */
		Morphology(size_t winRadiusV, size_t winRadiusH)
		    :profiler_t(3)
		{
		    setWinSizeV(winRadiusV);
		    setWinSizeH(winRadiusH);
		}

  //! boxフィルタのウィンドウの高さを返す．
  /*!
    \return	boxフィルタのウィンドウの高さ
   */
    size_t	winRadiusV()		const	{ return _winRadiusV; }

  //! boxフィルタのウィンドウの幅を返す．
  /*!
    \return	boxフィルタのウィンドウの幅
   */
    size_t	winRadiusH()		const	{ return _winRadiusH; }

  //! boxフィルタのウィンドウの高さを設定する．
  /*!
    \param winRadiusV	boxフィルタのウィンドウの高さ
    \return		このboxフィルタ
   */
    Morphology&	setWinSizeV(size_t winRadius)
		{
		    if (winRadius > WinSizeMax)
			throw std::runtime_error("Too large window size!");
		    _winRadiusV = winRadius;
		    return *this;
		}

  //! boxフィルタのウィンドウの幅を設定する．
  /*!
    \param winRadiusH	boxフィルタのウィンドウの幅
    \return		このboxフィルタ
   */
    Morphology&	setWinSizeH(size_t winRadius)
		{
		    if (winRadius > WinSizeMax)
			throw std::runtime_error("Too large window size!");
		    _winRadiusH = winRadius;
		    return *this;
		}

  //! 与えられた高さを持つ入力データ列に対する出力データ列の高さを返す．
  /*!
    \param inSizeV	入力データ列の高さ
    \return		出力データ列の高さ
   */
    size_t	outSizeV(size_t inSize)	const	{return inSize + 1 - _winRadiusV;}

  //! 与えられた幅を持つ入力データ列に対する出力データ列の幅を返す．
  /*!
    \param inSizeH	入力データ列の幅
    \return		出力データ列の幅
   */
    size_t	outSizeH(size_t inSize)	const	{return inSize + 1 - _winRadiusH;}

  //! 与えられた高さを持つ入力データ列に対する出力データ列の高さを返す．
  /*!
    \param inSizeV	入力データ列の高さ
    \return		出力データ列の高さ
   */
    size_t	offsetV()		const	{return _winRadiusV/2;}

  //! 与えられた幅を持つ入力データ列に対する出力データ列の幅を返す．
  /*!
    \param inSizeH	入力データ列の幅
    \return		出力データ列の幅
   */
    size_t	offsetH()		const	{return _winRadiusH/2;}

  //! 与えられた2次元配列とこのフィルタの畳み込みを行う
  /*!
    \param row	入力2次元データ配列の先頭行を指す反復子
    \param rowe	入力2次元データ配列の末尾の次の行を指す反復子
    \param rowO	出力2次元データ配列の先頭行を指す反復子
  */
    template <class ROW, class ROW_O>
    void	apply(ROW row, ROW rowe,
		      ROW_O rowO, bool shift=false)		const	;

  private:
    size_t			_winRadiusV;
    size_t			_winRadiusH;
    mutable Array2<value_type>	_buf;
};

#if defined(__NVCC__)
template <class CONVOLVER, class BOX_TRAITS, size_t WMAX, class CLOCK>
template <class ROW, class ROW_O> void
Morphology<CONVOLVER, BOX_TRAITS, WMAX, CLOCK>::apply(ROW row, ROW rowe,
							 ROW_O rowO,
							 bool shift) const
{
    using	std::size;

    profiler_t::start(0);

    const size_t nrows = std::distance(row, rowe);
    if (nrows < _winRadiusV)
	return;

    const size_t ncols = size(*row);
    if (ncols < _winRadiusH)
	return;

    _buf.resize(ncols, outSizeV(nrows));

  // Accumulate vertically.
    profiler_t::start(1);
    const dim3	threads(BlockDim, BlockDim);
    dim3	blocks(divUp(ncols, threads.x), divUp(nrows, threads.y));
    device::morphology<Morphology><<<blocks, threads>>>(
	cu::make_range(row, nrows),
	cu::make_range(_buf.begin(), _buf.nrow()), _winRadiusV);
    gpuCheckLastError();

  // Accumulate horizontally.
    cudaDeviceSynchronize();
    profiler_t::start(2);
    blocks.x = divUp(_buf.ncol(), threads.x);
    blocks.y = divUp(_buf.nrow(), threads.y);
    device::morphology<Morphology><<<blocks, threads>>>(
	cu::make_range(_buf.cbegin(), _buf.nrow()),
	cu::slice(rowO, (shift ? offsetV() : 0), outSizeV(nrows),
			(shift ? offsetH() : 0), outSizeH(ncols)),
	_winRadiusH);
    gpuCheckLastError();

    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}
#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
