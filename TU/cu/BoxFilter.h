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
  \file		BoxFilter.h
  \brief	boxフィルタの定義と実装
*/
#pragma once

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
*  class deque<T, MAX_SIZE>						*
************************************************************************/
template <class T, int MAX_SIZE>
class deque
{
  public:
    __device__		deque()	:_front(0), _back(0)	{}

    __device__ bool	empty()	const	{ return _front == _back; }
    __device__ T	front() const	{ return _buf[_front]; }
    __device__ T	back()	const
			{
			    return _buf[_back == 0 ? MAX_SIZE -1 : _back - 1];
			}
    __device__ void	push_front(T val)
			{
			    if (_front == 0)
				_front = MAX_SIZE;
			    _buf[--_front] = val;
			}
    __device__ void	push_back(T val)
			{
			    _buf[_back] = val;
			    if (++_back == MAX_SIZE)
				_back = 0;
			}
    __device__ void	pop_front()
			{
			    if (++_front == MAX_SIZE)
				_front = 0;
			}
    __device__ void	pop_back()
			{
			    if (_back == 0)
				_back = MAX_SIZE;
			    --_back;
			}

  private:
    T	_buf[MAX_SIZE];
    int	_front;
    int	_back;
};

/************************************************************************
*  __device__ functionals used as operator of extrema_finder<OP, WMAX>	*
************************************************************************/
template <class COMP>
struct extrema_value : public COMP
{
    using argument_type	= typename COMP::first_argument_type;
    using value_type	= typename COMP::first_argument_type;
    using result_type	= value_type;

    using COMP::operator ();

    __device__
    value_type	operator ()(const argument_type& v, int pos) const
		{
		    return v;
		}
};

namespace detail
{
template <class COMP>
struct extrema_position_base : public COMP
{
    using argument_type	= typename COMP::first_argument_type;
    using value_type	= thrust::tuple<argument_type, int>;
    using COMP::operator ();

    __device__
    value_type	operator ()(const argument_type& a, int pos) const
		{
		    return {a, pos};
		}
    __device__
    bool	operator ()(const value_type& v, const value_type& w) const
		{
		    return operator ()(thrust::get<0>(v), thrust::get<0>(w));
		}
};
}	// namespace detail

template <class COMP>
struct extrema_position : public detail::extrema_position_base<COMP>
{
    using super		= detail::extrema_position_base<COMP>;
    using typename super::value_type;
    using result_type	= vec<int, 2>;
    using super::operator ();

    __device__
    result_type	operator ()(const value_type& v, int pos) const
		{
		    return {pos, thrust::get<1>(v)};
		}
};

template <class COMP>
struct extrema_value_position : public detail::extrema_position_base<COMP>
{
    using super		= detail::extrema_position_base<COMP>;
    using typename super::argument_type;
    using typename super::value_type;
    using result_type	= thrust::tuple<argument_type, vec<int, 2> >;
    using super::operator ();

    __device__
    result_type	operator ()(const value_type& v, int pos) const
		{
		    return {thrust::get<0>(v), {pos, thrust::get<1>(v)}};
		}
};

/************************************************************************
*  __device__ convolution algorithms					*
************************************************************************/
template <class T>
struct box_convolver
{
    using value_type	= T;
    using result_type	= T;

    template <class S_, class T_, size_t W_> __device__
    void	operator ()(const S_ in_s[][W_],
			    T_ out_s[][W_], int winSize, int ye) const
		{
		    out_s[0][threadIdx.x] = in_s[0][threadIdx.x];
		    for (int y = 1; y != winSize; ++y)
			out_s[0][threadIdx.x] += in_s[y][threadIdx.x];

		    for (int y = 1; y != ye; ++y)
			out_s[y][threadIdx.x] = out_s[y-1][threadIdx.x]
					      + in_s[y-1+winSize][threadIdx.x]
					      - in_s[y-1][threadIdx.x];
		}
};

template <class OP, size_t WMAX=23>
struct extrema_finder
{
    using value_type	= typename OP::value_type;
    using result_type	= typename OP::result_type;

    template <class S_, class T_, size_t W_> __device__
    void	operator ()(const S_ in_s[][W_],
			    T_ out_s[][W_], int winSize, int ye) const
		{
		    const OP	op;
		    const int	x0	 = __mul24(blockIdx.x, blockDim.x);
		    const int	y0	 = __mul24(blockIdx.y, blockDim.y);
		    const int	winSize1 = winSize - 1;
		    deque<int, WMAX> q;
		    q.push_back(0);

		    for (int y = 1; y != ye + winSize1; ++y)
		    {
			while (op(in_s[y][threadIdx.x],
				  in_s[q.back()][threadIdx.x]))
			{
			    q.pop_back();
			    if (q.empty())
				break;
			}
			q.push_back(y);

			if (y == q.front() + winSize)
			    q.pop_front();

			if (y >= winSize1)
			    out_s[y - winSize1][threadIdx.x]
				= op(in_s[q.front()][threadIdx.x],
				     y0 + q.front());
		    }
		}
};

/************************************************************************
*  __global__ functions							*
************************************************************************/
//! スレッドブロックの縦方向にフィルタを適用する
/*!
  sliding windowを使用するが threadIdx.y が0のスレッドしか仕事をしないので
  ウィンドウ幅が大きいときのみ高効率．また，結果は転置して格納される．
  \param in		入力2次元配列
  \param out		出力2次元配列
  \param winSize	boxフィルタのウィンドウの行幅(高さ)
*/
template <class FILTER, class IN, class OUT>
__global__ void
box_filter(range<range_iterator<IN> > in,
	   range<range_iterator<OUT> > out, int winSize)
{
    using convolver_type = typename FILTER::convolver_type;
    using value_type	 = typename convolver_type::value_type;
    using result_type	 = typename convolver_type::result_type;
    using in_type	 = typename std::iterator_traits<IN>::value_type;
    using out_type	 = std::conditional_t<
				std::is_same<in_type, value_type>::value,
				result_type, value_type>;

    const int	x0   = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0   = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xsiz = ::min(blockDim.x, in.begin().size() - x0);
    const int	ysiz = ::min(blockDim.y + winSize - 1, in.size() - y0);

    __shared__ in_type	in_s[FILTER::BlockDim + FILTER::WinSizeMax - 1]
			    [FILTER::BlockDim + 1];
    loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
    __syncthreads();

    __shared__ out_type	out_s[FILTER::BlockDim][FILTER::BlockDim + 1];

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
		out[x0 + threadIdx.x][y0 + threadIdx.y]
		    = out_s[threadIdx.y][threadIdx.x];
	}
    }
}

template <class FILTER, class IN, class OUT, class OP> __global__ void
box_filterH(IN inL, IN inR, int ncols, OUT out, OP op, int winSizeH,
	    int strideL, int strideR, int strideXD, int strideD)
{
    using value_type  =	typename FILTER::value_type;

    __shared__ value_type val_s[FILTER::WinSizeMax]
			       [FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const auto	d = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// 視差
    const auto	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;	// 行

    inL += __mul24(y, strideL);
    inR += (__mul24(y, strideR)  + d);
    out += (__mul24(y, strideXD) + d);

  // 最初のwinSize画素分の相違度を計算してvalに積算
    auto	val = (val_s[0][threadIdx.y][threadIdx.x] = op(*inL, *inR));
    for (int i = 0; ++i != winSizeH; )
	val += (val_s[i][threadIdx.y][threadIdx.x] = op(*++inL, *++inR));
    *out = val;

  // 逐次的にvalを更新して出力
    for (int i = 0; --ncols; )
    {
	val -= val_s[i][threadIdx.y][threadIdx.x];
	*(out += strideD) = (val += (val_s[i][threadIdx.y][threadIdx.x]
				     = op(*++inL, *++inR)));
	if (++i == winSizeH)
	    i = 0;
    }
}

template <class FILTER, class IN> __global__ void
box_filterV(IN in, int nrows, int winSizeV, int strideXD, int strideD)
{
    using value_type  =	typename FILTER::value_type;

    __shared__ value_type	in_s[FILTER::WinSizeMax]
				    [FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const auto	d = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// 視差
    const auto	x = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;	// 列

    in += (__mul24(x, strideD) + d);

    auto	out = in;
    auto	val = (in_s[0][threadIdx.y][threadIdx.x] = *in);
    for (int i = 0; ++i < winSizeV; )
  	val += (in_s[i][threadIdx.y][threadIdx.x] = *(in += strideXD));
    *out = val;

    for (int i = 0; --nrows; )
    {
	val -= in_s[i][threadIdx.y][threadIdx.x];
	*(out += strideXD)
	    = (val += (in_s[i][threadIdx.y][threadIdx.x] = *(in += strideXD)));

	if (++i == winSizeV)
	    i = 0;
    }
}

template <class FILTER, class IN, class OUT> __global__ void
box_filterV(IN in, int nrows, OUT out, int winSizeV,
	    int strideXD, int strideD, int strideYX_O, int strideX_O)
{
    using value_type  =	typename FILTER::value_type;

    __shared__ value_type	in_s[FILTER::WinSizeMax]
				    [FILTER::BlockDim][FILTER::BlockDim + 1];
    __shared__ value_type	out_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const auto	d0 = __mul24(blockIdx.x, blockDim.x);	// 視差
    const auto	x0 = __mul24(blockIdx.y, blockDim.y);	// 列

    in  += (__mul24(x0 + threadIdx.y, strideD) + d0 + threadIdx.x);
    out += (__mul24(d0,		   strideYX_O) + x0);

    out_s[threadIdx.y][threadIdx.x] = (in_s[0][threadIdx.y][threadIdx.x] = *in);
    for (int i = 0; ++i < winSizeV; )
	out_s[threadIdx.y][threadIdx.x]
	    += (in_s[i][threadIdx.y][threadIdx.x] = *(in += strideXD));
    __syncthreads();
    if (blockDim.x == blockDim.y)
	out[__mul24(threadIdx.y, strideYX_O) + threadIdx.x]
	    = out_s[threadIdx.x][threadIdx.y];
    else
	out[__mul24(threadIdx.x, strideYX_O) + threadIdx.y]
	    = out_s[threadIdx.y][threadIdx.x];

    for (int i = 0; --nrows; )
    {
	out_s[threadIdx.y][threadIdx.x] -=  in_s[i][threadIdx.y][threadIdx.x];
	out_s[threadIdx.y][threadIdx.x] += (in_s[i][threadIdx.y][threadIdx.x]
					    = *(in += strideXD));
	__syncthreads();

	out += strideX_O;

	if (blockDim.x == blockDim.y)
	    out[__mul24(threadIdx.y, strideYX_O) + threadIdx.x]
		= out_s[threadIdx.x][threadIdx.y];
	else
	    out[__mul24(threadIdx.x, strideYX_O) + threadIdx.y]
		= out_s[threadIdx.y][threadIdx.x];

	if (++i == winSizeV)
	    i = 0;
    }
}
}	// namespace device
#endif	// __NVCC__

/************************************************************************
*  class BoxFilter2<CONVOLVER, BLOCK_TRAITS, WMAX, CLOCK>		*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class CONVOLVER=device::box_convolver<float>,
	  class BLOCK_TRAITS=BlockTraits<>, size_t WMAX=23, class CLOCK=void>
class BoxFilter2 : public BLOCK_TRAITS, public Profiler<CLOCK>
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
    \param winSizeV	boxフィルタのウィンドウの行幅(高さ)
    \param winSizeH	boxフィルタのウィンドウの列幅(幅)
   */
		BoxFilter2(size_t winSizeV, size_t winSizeH)
		    :profiler_t(3)
		{
		    setWinSizeV(winSizeV);
		    setWinSizeH(winSizeH);
		}

  //! boxフィルタのウィンドウの高さを返す．
  /*!
    \return	boxフィルタのウィンドウの高さ
   */
    size_t	winSizeV()		const	{ return _winSizeV; }

  //! boxフィルタのウィンドウの幅を返す．
  /*!
    \return	boxフィルタのウィンドウの幅
   */
    size_t	winSizeH()		const	{ return _winSizeH; }

  //! boxフィルタのウィンドウの高さを設定する．
  /*!
    \param winSizeV	boxフィルタのウィンドウの高さ
    \return		このboxフィルタ
   */
    BoxFilter2&	setWinSizeV(size_t winSize)
		{
		    if (winSize > WinSizeMax)
			throw std::runtime_error("Too large window size!");
		    _winSizeV = winSize;
		    return *this;
		}

  //! boxフィルタのウィンドウの幅を設定する．
  /*!
    \param winSizeH	boxフィルタのウィンドウの幅
    \return		このboxフィルタ
   */
    BoxFilter2&	setWinSizeH(size_t winSize)
		{
		    if (winSize > WinSizeMax)
			throw std::runtime_error("Too large window size!");
		    _winSizeH = winSize;
		    return *this;
		}

  //! 与えられた高さを持つ入力データ列に対する出力データ列の高さを返す．
  /*!
    \param inSizeV	入力データ列の高さ
    \return		出力データ列の高さ
   */
    size_t	outSizeV(size_t inSize)	const	{return inSize + 1 - _winSizeV;}

  //! 与えられた幅を持つ入力データ列に対する出力データ列の幅を返す．
  /*!
    \param inSizeH	入力データ列の幅
    \return		出力データ列の幅
   */
    size_t	outSizeH(size_t inSize)	const	{return inSize + 1 - _winSizeH;}

  //! 与えられた高さを持つ入力データ列に対する出力データ列の高さを返す．
  /*!
    \param inSizeV	入力データ列の高さ
    \return		出力データ列の高さ
   */
    size_t	offsetV()		const	{return _winSizeV/2;}

  //! 与えられた幅を持つ入力データ列に対する出力データ列の幅を返す．
  /*!
    \param inSizeH	入力データ列の幅
    \return		出力データ列の幅
   */
    size_t	offsetH()		const	{return _winSizeH/2;}

  //! 与えられた2次元配列とこのフィルタの畳み込みを行う
  /*!
    \param row	入力2次元データ配列の先頭行を指す反復子
    \param rowe	入力2次元データ配列の末尾の次の行を指す反復子
    \param rowO	出力2次元データ配列の先頭行を指す反復子
  */
    template <class ROW, class ROW_O>
    void	convolve(ROW row, ROW rowe,
			 ROW_O rowO, bool shift=true)		const	;

  //! 2組の2次元配列間の相違度とこのフィルタの畳み込みを行う
  /*!
    \param rowL			左入力2次元データ配列の先頭行を指す反復子
    \param rowLe		左入力2次元データ配列の末尾の次の行を指す反復子
    \param rowR			右入力2次元データ配列の先頭行を指す反復子
    \param rowO			出力2次元相違度配列の先頭行を指す反復子
    \param op			左右の画素間の相違度
    \param disparitySearchWidth	視差の探索幅
  */
    template <class ROW, class ROW_O, class OP>
    void	convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			 OP op, size_t disparitySearchWidth)	const	;

  private:
    size_t			_winSizeV;
    size_t			_winSizeH;
    mutable Array2<value_type>	_buf;
    mutable Array3<value_type>	_buf3;
};

#if defined(__NVCC__)
template <class CONVOLVER, class BOX_TRAITS, size_t WMAX, class CLOCK>
template <class ROW, class ROW_O> void
BoxFilter2<CONVOLVER, BOX_TRAITS, WMAX, CLOCK>::convolve(ROW row, ROW rowe,
							 ROW_O rowO,
							 bool shift) const
{
    using	std::size;

    profiler_t::start(0);

    const size_t nrows = std::distance(row, rowe);
    if (nrows < _winSizeV)
	return;

    const size_t ncols = size(*row);
    if (ncols < _winSizeH)
	return;

    _buf.resize(ncols, outSizeV(nrows));

  // Accumulate vertically.
    profiler_t::start(1);
    const dim3	threads(BlockDim, BlockDim);
    dim3	blocks(divUp(ncols, threads.x), divUp(nrows, threads.y));
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
	cu::make_range(row, nrows),
	cu::make_range(_buf.begin(), _buf.nrow()), _winSizeV);
    gpuCheckLastError();

  // Accumulate horizontally.
    cudaDeviceSynchronize();
    profiler_t::start(2);
    blocks.x = divUp(_buf.ncol(), threads.x);
    blocks.y = divUp(_buf.nrow(), threads.y);
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
	cu::make_range(_buf.cbegin(), _buf.nrow()),
	cu::slice(rowO, (shift ? offsetV() : 0), outSizeV(nrows),
			(shift ? offsetH() : 0), outSizeH(ncols)),
	_winSizeH);
    gpuCheckLastError();

    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}

template <class CONVOLVER, class BOX_TRAITS, size_t WMAX, class CLOCK>
template <class ROW, class ROW_O, class OP> void
BoxFilter2<CONVOLVER, BOX_TRAITS, WMAX, CLOCK>::convolve(
    ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
    OP op, size_t disparitySearchWidth) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;
    using	std::size;

    profiler_t::start(0);

    size_t	nrows = std::distance(rowL, rowLe);
    if (nrows < _winSizeV)
	return;

    size_t	ncols = size(*rowL);
    if (ncols < _winSizeH)
	return;

    _buf3.resize(nrows, ncols, disparitySearchWidth);

    const auto	strideL    = stride(rowL);
    const auto	strideR    = stride(rowR);
    const auto	strideD	   = _buf3.stride();
    const auto	strideXD   = size<1>(_buf3)*strideD;
    const auto	strideX_O  = stride(cbegin(*rowO));

  // ---- 横方向積算 ----
    profiler_t::start(1);
    ncols = outSizeH(ncols);
  // 視差左半かつ画像上半
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(disparitySearchWidth/threads.x, nrows/threads.y);
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
	cbegin(*rowL), cbegin(*rowR), ncols,
#ifdef DISPARITY_MAJOR
	begin(_buf3[0][0]),
#else
	begin(*begin(*rowO)),
#endif
	op, _winSizeH, strideL, strideR, strideXD, strideD);
    gpuCheckLastError();
  // 視差右半かつ画像上半
    const auto	d = blocks.x*threads.x;
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
	cbegin(*rowL), cbegin(*rowR) + d, ncols,
#ifdef DISPARITY_MAJOR
	begin(_buf3[0][0]) + d,
#else
	begin(*begin(*rowO)) + d,
#endif
	op, _winSizeH, strideL, strideR, strideXD, strideD);
    gpuCheckLastError();
  // 視差左半かつ画像下半
    threads.x = BlockDimX;
    blocks.x  = disparitySearchWidth/threads.x;
    const auto	y = blocks.y*threads.y;
    threads.y = disparitySearchWidth - d;
    blocks.y  = 1;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
	cbegin(*(rowL + y)), cbegin(*(rowR + y)), ncols,
#ifdef DISPARITY_MAJOR
	begin(_buf3[y][0]),
#else
	begin(*begin(*(rowO + y))),
#endif
	op, _winSizeH, strideL, strideR, strideXD, strideD);
    gpuCheckLastError();
  // 視差右半かつ画像下半
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
	cbegin(*(rowL + y)) + d, cbegin(*(rowR + y)) + d,
	ncols,
#ifdef DISPARITY_MAJOR
	begin(_buf3[y][0]) + d,
#else
	begin(*begin(*(rowO + y))) + d,
#endif
	op, _winSizeH, strideL, strideR, strideXD, strideD);
    gpuCheckLastError();
  // ---- 縦方向積算 ----
    cudaDeviceSynchronize();
    profiler_t::start(2);
#ifdef DISPARITY_MAJOR
    const auto	strideYX_O = nrows*strideX_O;
  // 視差左半かつ画像左半
    nrows     = outSizeV(nrows);
    threads.x = BlockDim;
    blocks.x  = disparitySearchWidth/threads.x;
    threads.y = BlockDim;
    blocks.y  = ncols/threads.y;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(_buf3[0][0]), nrows, begin(*begin(*rowO)),
	_winSizeV, strideXD, strideD, strideYX_O, strideX_O);
    gpuCheckLastError();
  // 視差右半かつ画像左半
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(_buf3[0][0]) + d, nrows,
	begin(*begin(*(rowO + d))),
	_winSizeV, strideXD, strideD, strideYX_O, strideX_O);
    gpuCheckLastError();
  // 視差左半かつ画像右半
    threads.x = BlockDim;
    blocks.x  = disparitySearchWidth/threads.x;
    const auto	x = blocks.y*threads.y;
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(_buf3[0][x]), nrows, begin(*begin(*rowO)) + x,
	_winSizeV, strideXD, strideD, strideYX_O, strideX_O);
    gpuCheckLastError();
  // 視差右半かつ画像右半
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(_buf3[0][x]) + d, nrows,
	begin(*begin(*(rowO + d))) + x,
	_winSizeV, strideXD, strideD, strideYX_O, strideX_O);
    gpuCheckLastError();
#else
  // 視差左半かつ画像左半
    nrows     = outSizeV(nrows);
    threads.x = BlockDimX;
    blocks.x  = disparitySearchWidth/threads.x;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(*begin(*rowO)), nrows, _winSizeV, strideXD, strideD);
    gpuCheckLastError();
  // 視差右半かつ画像左半
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(*begin(*(rowO + d))), nrows, _winSizeV, strideXD, strideD);
    gpuCheckLastError();
  // 視差左半かつ画像右半
    threads.x = BlockDimX;
    blocks.x  = disparitySearchWidth/threads.x;
    const auto	x = blocks.y*threads.y;
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(*(begin(*rowO) + x)), nrows, _winSizeV, strideXD, strideD);
    gpuCheckLastError();
  // 視差右半かつ画像右半
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
	begin(*(begin(*rowO) + x)) + d,
	nrows, _winSizeV, strideXD, strideD);
    gpuCheckLastError();
#endif
    cudaDeviceSynchronize();
    profiler_t::nextFrame();
}
#endif	// __NVCC__
}	// namespace cu
}	// namespace TU
