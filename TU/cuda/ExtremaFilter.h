/*!
  \file		ExtreamFilter.h
  \brief	finite impulse responseフィルタの定義と実装
*/
#pragma once

#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"

namespace TU
{
namespace cuda
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
*  __global__ functions							*
************************************************************************/
template <class FILTER, class IN, class OUT, class COMPARE>
__global__ void
extrema_filter(range<range_iterator<IN> > in,
	       range<range_iterator<OUT> > out, COMPARE compare, int winSize)
{
    using value_type	= typename FILTER::value_type;

    const int	winSize1 = winSize - 1;
    const int	x0	 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0	 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xsiz	 = ::min(blockDim.x, in.begin().size() - x0);
    const int	ysiz	 = ::min(blockDim.y + winSize1, in.size() - y0);
      
    __shared__ value_type in_s[FILTER::BlockDim + FILTER::WinSizeMax - 1]
			      [FILTER::BlockDim + 1];
    loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
    __syncthreads();

    __shared__ value_type out_s[FILTER::BlockDim][FILTER::BlockDim + 1];

    const int	ye = ysiz - winSize1;

    if (threadIdx.x < xsiz && ye > 0)
    {
	if (threadIdx.y == 0)
	{
	    deque<int, FILTER::WinSizeMax> q;
	    q.push_back(0);

	    for (int y = 0; ++y != blockDim.y + winSize1; )
	    {
		while (compare(in_s[y][threadIdx.x],
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
			= in_s[q.front()][threadIdx.x];
	    }
	}
	__syncthreads();

      // 結果を転置して格納
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

template <class FILTER, class IN, class OUT, class POS,
	  class COMPARE, class STRIDE_I, class STRIDE_O>
__global__ void
extrema_filterV(IN in, OUT out, POS pos, COMPARE compare, int winSize,
		STRIDE_I strideI, STRIDE_O strideO, ptrdiff_t strideP,
		int xx, int yy)
{
    using value_type  =	typename FILTER::value_type;

    __shared__ value_type	in_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
				    [FILTER::BlockDimX + 1];
    __shared__ value_type	out_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];
    __shared__ short2		pos_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const auto	x0 = __mul24(blockIdx.x, blockDim.x); // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y); // ブロック左上隅
    const auto	winSize1 = winSize - 1;

    advance_stride(in, y0*strideI);
    in += x0;
    loadTileV(in, strideI, in_s, winSize1);

    __syncthreads();

    if (threadIdx.y == 0)
    {
	deque<short, FILTER::WinSizeMax> q;
	q.push_back(0);

	for (short y = 0; ++y != blockDim.y + winSize1; )
	{
	    while (compare(in_s[y][threadIdx.x],
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
	    {
		out_s[y - winSize1][threadIdx.x]
		    = in_s[q.front()][threadIdx.x];
		pos_s[y - winSize1][threadIdx.x]
		    = make_short2(x0 + xx + threadIdx.x, y0 + yy + q.front());
	    }
	}
    }
    __syncthreads();

  // 結果を転置して格納
    if (blockDim.x == blockDim.y)
    {
	advance_stride(out, (x0 + threadIdx.y)*strideO);
	out[y0 + threadIdx.x] = out_s[threadIdx.x][threadIdx.y];
	advance_stride(pos, (x0 + threadIdx.y)*strideP);
	pos[y0 + threadIdx.x] = pos_s[threadIdx.x][threadIdx.y];
    }
    else
    {
	advance_stride(out, (x0 + threadIdx.x)*strideO);
	out[y0 + threadIdx.y] = out_s[threadIdx.y][threadIdx.x];
	advance_stride(pos, (x0 + threadIdx.x)*strideP);
	pos[y0 + threadIdx.y] = pos_s[threadIdx.y][threadIdx.x];
    }
}

template <class FILTER, class IN, class OUT, class POS_IN, class POS_OUT,
	  class COMPARE, class STRIDE_I, class STRIDE_O>
__global__ void
extrema_filterH(IN in, OUT out, POS_IN pos_in, POS_OUT pos_out,
		COMPARE compare, int winSize,
		STRIDE_I strideI, STRIDE_O strideO,
		ptrdiff_t stridePI, ptrdiff_t stridePO)
{
    using value_type  =	typename FILTER::value_type;

    __shared__ value_type	in_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
				    [FILTER::BlockDimX + 1];
    __shared__ value_type	out_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];
    __shared__ short2		pos_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
				     [FILTER::BlockDimX + 1];

    const auto	x0 = __mul24(blockIdx.x, blockDim.x); // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y); // ブロック左上隅
    const short	winSize1 = winSize - 1;
      
    advance_stride(in, y0*strideI);
    in += x0;
    loadTileV(in, strideI, in_s, winSize1);
    
    advance_stride(pos_in, y0*stridePI);
    pos_in += x0;
    loadTileV(pos_in, stridePI, pos_s, winSize1);

    __syncthreads();

    if (threadIdx.y == 0)
    {
	deque<short, FILTER::WinSizeMax> q;
	q.push_back(0);

	for (short y = 0; ++y != blockDim.y + winSize1; )
	{
	    while (compare(in_s[y][threadIdx.x],
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
	    {
		out_s[y - winSize1][threadIdx.x]
		    = in_s[q.front()][threadIdx.x];
		pos_s[y - winSize1][threadIdx.x]
		    = pos_s[q.front()][threadIdx.x];
	    }
	}
    }
    __syncthreads();

  // 結果を転置して格納
    if (blockDim.x == blockDim.y)
    {
	advance_stride(out, (x0 + threadIdx.y)*strideO);
	out[y0 + threadIdx.x] = out_s[threadIdx.x][threadIdx.y];
	advance_stride(pos_out, (x0 + threadIdx.y)*stridePO);
	pos_out[y0 + threadIdx.x] = pos_s[threadIdx.x][threadIdx.y];
    }
    else
    {
	advance_stride(out, (x0 + threadIdx.x)*strideO);
	out[y0 + threadIdx.y] = out_s[threadIdx.y][threadIdx.x];
	advance_stride(pos_out, (x0 + threadIdx.x)*stridePO);
	pos_out[y0 + threadIdx.y] = pos_s[threadIdx.y][threadIdx.x];
    }
}
    
}	// namespace device
#endif	// __NVCC__
    
/************************************************************************
*  class ExtremaFilter2<T, BLOCK_TRAITS, WMAX>				*
************************************************************************/
//! CUDAによる2次元extremaフィルタを表すクラス
template <class T, class BLOCK_TRAITS=BlockTraits<16, 16>, size_t WMAX=23>
class ExtremaFilter2 : public BLOCK_TRAITS
{
  public:
    using value_type	= T;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
  //constexpr static size_t	BlockDim   = BlockDimY;
    constexpr static size_t	WinSizeMax = WMAX;

  public:
			ExtremaFilter2(size_t winSizeV, size_t winSizeH)
			    :_winSizeV(winSizeV), _winSizeH(winSizeH)
			{
			    if (_winSizeV > WinSizeMax ||
				_winSizeH > WinSizeMax)
				throw std::runtime_error("Too large window size!");
			}

    size_t		winSizeV()		const	{ return _winSizeV; }
    size_t		winSizeH()		const	{ return _winSizeH; }

    ExtremaFilter2&	setWinSizeV(size_t winSize)
			{
			    _winSizeV = winSize;
			    return *this;
			}

    ExtremaFilter2&	setWinSizeH(size_t winSize)
			{
			    _winSizeH = winSize;
			    return *this;
			}

    size_t		outSizeV(size_t inSize)	const
			{
			    return inSize + 1 - _winSizeV;
			}

    size_t		outSizeH(size_t inSize)	const
			{
			    return inSize + 1 - _winSizeH;
			}

    size_t		offsetV()		const	{ return _winSizeV/2; }
    size_t		offsetH()		const	{ return _winSizeH/2; }

    template <class ROW, class ROW_O, class COMPARE>
    void		convolve(ROW row, ROW rowe, ROW_O rowO,
				 COMPARE op, bool shift=false)	const	;
    template <class ROW, class ROW_O, class ROW_P, class COMPARE>
    void		extrema(ROW row, ROW rowe, ROW_O rowO, ROW_P rowP,
				COMPARE compare, bool shift=false) const;

  private:
    size_t			_winSizeV;
    size_t			_winSizeH;
    mutable Array2<T>		_buf;
    mutable Array2<short2>	_buf_pos;
};

#if defined(__NVCC__)
template <class T, class BLOCK_TRAITS, size_t WMAX>
template <class ROW, class ROW_O, class COMPARE> void
ExtremaFilter2<T, BLOCK_TRAITS, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO,
						COMPARE compare,
						bool shift) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;
    
    size_t	nrows = std::distance(row, rowe);
    if (nrows < _winSizeV)
	return;

    size_t	ncols = std::distance(cbegin(*row), cend(*row));
    if (ncols < _winSizeH)
	return;

    nrows = outSizeV(nrows);
    _buf.resize(ncols, nrows);

    const auto	strideI = stride(row);
    const auto	strideO = stride(rowO);
    const auto	strideB = _buf.stride();

  // ---- 縦方向積算 ----
  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncols/threads.x, nrows/threads.y);
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(*row), begin(_buf[0]), compare, _winSizeV, strideI, strideB);

  // 右上
    auto	x = blocks.x*threads.x;
    threads.x = ncols - x;
    blocks.x  = 1;
    if (x < ncols)
	device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	    cbegin(*row) + x, begin(_buf[x]), compare, _winSizeV,
	    strideI, strideB);

  // 左下
    auto	y = blocks.y*threads.y;
    std::advance(row, y);
    threads.x = BlockDimX;
    blocks.x  = ncols/threads.x;
    threads.y = nrows - y;
    blocks.y  = 1;
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(*row), begin(_buf[0]) + y, compare, _winSizeV,
	strideI, strideB);

  // 右下
    threads.x = ncols - x;
    blocks.x  = 1;
    if (x < ncols)
	device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	    cbegin(*row) + x, begin(_buf[x]) + y, compare, _winSizeV,
	    strideI, strideB);

  // ---- 横方向積算 ----
    size_t	dx = 0;
    if (shift)
    {
	rowO += offsetV();
	dx    = offsetH();
    }
    ncols = outSizeH(ncols);

  // 左上
    threads.x = BlockDimX;
    blocks.x  = nrows/threads.x;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[0]), begin(*rowO) + dx, compare, _winSizeH,
	strideB, strideO);

  // 左下
    x	      = blocks.y*threads.y;
    threads.y = ncols - x;
    blocks.y  = 1;
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[x]), begin(*rowO) + x + dx, compare, _winSizeH,
	strideB, strideO);

  // 右上
    y	      = blocks.x*threads.x;
    std::advance(rowO, y);
    threads.x = nrows - y;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[0]) + y, begin(*rowO) + dx, compare, _winSizeH,
	strideB, strideO);

  // 右下
    threads.y = ncols - x;
    blocks.y  = 1;
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[x]) + y, begin(*rowO) + x + dx, compare, _winSizeH,
	strideB, strideO);
}

template <class T, class BLOCK_TRAITS, size_t WMAX>
template <class ROW, class ROW_O, class ROW_P, class COMPARE> void
ExtremaFilter2<T, BLOCK_TRAITS, WMAX>::extrema(ROW row, ROW rowe,
					       ROW_O rowO, ROW_P rowP,
					       COMPARE compare,
					       bool shift) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;
    
    size_t	nrows = std::distance(row, rowe);
    if (nrows < _winSizeV)
	return;

    size_t	ncols = std::distance(cbegin(*row), cend(*row));
    if (ncols < _winSizeH)
	return;

    nrows = outSizeV(nrows);
    _buf.resize(ncols, nrows);
    _buf_pos.resize(ncols, nrows);

    const auto	strideI  = stride(row);
    const auto	strideO  = stride(rowO);
    const auto	strideP  = stride(rowP);
    const auto	strideB  = _buf.stride();
    const auto	strideBP = _buf_pos.stride();

  // ---- 縦方向積算 ----
  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncols/threads.x, nrows/threads.y);
    device::extrema_filterV<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(*row), begin(_buf[0]), begin(_buf_pos[0]),
	compare, _winSizeV, strideI, strideB, strideBP, 0, 0);

  // 右上
    auto	x = blocks.x*threads.x;
    threads.x = ncols - x;
    blocks.x  = 1;
    if (x < ncols)
	device::extrema_filterV<ExtremaFilter2><<<blocks, threads>>>(
	    cbegin(*row) + x, begin(_buf[x]), begin(_buf_pos[x]),
	    compare, _winSizeV, strideI, strideB, strideBP, x, 0);

  // 左下
    auto	y = blocks.y*threads.y;
    std::advance(row, y);
    threads.x = BlockDimX;
    blocks.x  = ncols/threads.x;
    threads.y = nrows - y;
    blocks.y  = 1;
    device::extrema_filterV<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(*row), begin(_buf[0]) + y, begin(_buf_pos[0]) + y,
	compare, _winSizeV, strideI, strideB, strideBP, 0, y);

  // 右下
    threads.x = ncols - x;
    blocks.x  = 1;
    if (x < ncols)
	device::extrema_filterV<ExtremaFilter2><<<blocks, threads>>>(
	    cbegin(*row) + x, begin(_buf[x]) + y, begin(_buf_pos[x]) + y,
	    compare, _winSizeV, strideI, strideB, strideBP, x, y);

  // ---- 横方向積算 ----
    size_t	dx = 0;
    if (shift)
    {
	rowO += offsetV();
	rowP += offsetV();
	dx    = offsetH();
    }
    ncols = outSizeH(ncols);

  // 左上
    threads.x = BlockDimX;
    blocks.x  = nrows/threads.x;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::extrema_filterH<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[0]), begin(*rowO) + dx,
	cbegin(_buf_pos[0]), begin(*rowP) + dx,
	compare, _winSizeH, strideB, strideO, strideBP, strideP);

  // 左下
    x	      = blocks.y*threads.y;
    threads.y = ncols - x;
    blocks.y  = 1;
    device::extrema_filterH<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[x]), begin(*rowO) + x + dx,
	cbegin(_buf_pos[x]), begin(*rowP) + x + dx,
	compare, _winSizeH, strideB, strideO, strideBP, strideP);

  // 右上
    y	      = blocks.x*threads.x;
    std::advance(rowO, y);
    std::advance(rowP, y);
    threads.x = nrows - y;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::extrema_filterH<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[0]) + y, begin(*rowO) + dx,
	cbegin(_buf_pos[0]) + y, begin(*rowP) + dx,
	compare, _winSizeH, strideB, strideO, strideBP, strideP);

  // 右下
    threads.y = ncols - x;
    blocks.y  = 1;
    device::extrema_filterH<ExtremaFilter2><<<blocks, threads>>>(
	cbegin(_buf[x]) + y, begin(*rowO) + x + dx,
	cbegin(_buf_pos[x]) + y, begin(*rowP) + x + dx,
	compare, _winSizeH, strideB, strideO, strideBP, strideP);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
