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
extrema_filter(range<range_iterator<IN> >  in,
	       range<range_iterator<OUT> > out, int winSize, COMPARE compare)
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

template <class FILTER, class IN, class OUT, class POS, class COMPARE>
__global__ void
extrema_filterV(range<range_iterator<IN> > in, range<range_iterator<OUT> > out,
		range<range_iterator<POS> > pos, int winSize, COMPARE compare)
{
    using value_type  =	typename FILTER::value_type;

    const int	winSize1 = winSize - 1;
    const int	x0	 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0	 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xsiz	 = ::min(blockDim.x, in.begin().size() - x0);
    const int	ysiz	 = ::min(blockDim.y + winSize1, in.size() - y0);
      
    __shared__ value_type in_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
			      [FILTER::BlockDimX + 1];
    loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
    __syncthreads();
    
    __shared__ value_type out_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];
    __shared__ short2	  pos_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const int	ye = ysiz - winSize1;

    if (threadIdx.x < xsiz && ye > 0)
    {
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
			= make_short2(x0 + threadIdx.x,
				      y0 + q.front());
		}
	    }
	}
	__syncthreads();

      // 結果を転置して格納
	if (xsiz == blockDim.x)
	{
	    if (threadIdx.x < ye)
	    {
		out[x0 + threadIdx.y][y0 + threadIdx.x]
		    = out_s[threadIdx.x][threadIdx.y];
		pos[x0 + threadIdx.y][y0 + threadIdx.x]
		    = pos_s[threadIdx.x][threadIdx.y];
	    }
	}
	else
	{
	    if (threadIdx.y < ye)
	    {
		out[x0 + threadIdx.x][y0 + threadIdx.y]
		    = out_s[threadIdx.y][threadIdx.x];
		pos[x0 + threadIdx.x][y0 + threadIdx.y]
		    = pos_s[threadIdx.y][threadIdx.x];
	    }
	}
    }
}

template <class FILTER,
	  class IN, class POS_IN, class OUT, class POS_OUT, class COMPARE>
__global__ void
extrema_filterH(range<range_iterator<IN>      > in, 
		range<range_iterator<POS_IN>  > pos_in,
		range<range_iterator<OUT>     > out,
		range<range_iterator<POS_OUT> > pos_out,
		int winSize, COMPARE compare)
{
    using value_type  =	typename FILTER::value_type;

    const int	winSize1 = winSize - 1;
    const int	x0	 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0	 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xsiz	 = ::min(blockDim.x, in.begin().size() - x0);
    const int	ysiz	 = ::min(blockDim.y + winSize1, in.size() - y0);
      
    __shared__ value_type in_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
			      [FILTER::BlockDimX + 1];
    loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
    __syncthreads();

    __shared__ short2	  pos_s[FILTER::BlockDimY + FILTER::WinSizeMax - 1]
			       [FILTER::BlockDimX + 1];
    loadTile(slice(pos_in.cbegin(), y0, ysiz, x0, xsiz), pos_s);
    __syncthreads();

    __shared__ value_type out_s[FILTER::BlockDimY][FILTER::BlockDimX + 1];

    const int	ye = ysiz - winSize1;

    if (threadIdx.x < xsiz && ye > 0)
    {
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
	if (xsiz == blockDim.x)
	{
	    if (threadIdx.x < ye)
	    {
		out[x0 + threadIdx.y][y0 + threadIdx.x]
		    = out_s[threadIdx.x][threadIdx.y];
		pos_out[x0 + threadIdx.y][y0 + threadIdx.x]
		    = pos_s[threadIdx.x][threadIdx.y];
	    }
	}
	else
	{
	    if (threadIdx.y < ye)
	    {
		out[x0 + threadIdx.x][y0 + threadIdx.y]
		    = out_s[threadIdx.y][threadIdx.x];
		pos_out[x0 + threadIdx.x][y0 + threadIdx.y]
		    = pos_s[threadIdx.y][threadIdx.x];
	    }
	}
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
    constexpr static size_t	BlockDim   = BlockDimY;
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
    const int	nrows = std::distance(row, rowe);
    if (nrows < _winSizeV)
	return;

    const int	ncols = TU::size(*row);
    if (ncols < _winSizeH)
	return;

    _buf.resize(ncols, outSizeV(nrows));

  // Accumulate vertically.
    const dim3	threads(BlockDim, BlockDim);
    dim3	blocks(gridDim(ncols, threads.x), gridDim(nrows, threads.y));
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cuda::make_range(row, nrows),
	cuda::make_range(_buf.begin(), _buf.nrow()), _winSizeV, compare);

  // Accumulate horizontally,
    blocks.x = gridDim(_buf.ncol(), threads.x);
    blocks.y = gridDim(_buf.nrow(), threads.y);
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cuda::make_range(_buf.cbegin(), _buf.nrow()),
	cuda::slice(rowO, (shift ? offsetV() : 0), outSizeV(nrows),
			  (shift ? offsetH() : 0), outSizeH(ncols)),
	_winSizeH, compare);
}

template <class T, class BLOCK_TRAITS, size_t WMAX>
template <class ROW, class ROW_O, class ROW_P, class COMPARE> void
ExtremaFilter2<T, BLOCK_TRAITS, WMAX>::extrema(ROW row, ROW rowe,
					       ROW_O rowO, ROW_P rowP,
					       COMPARE compare,
					       bool shift) const
{
    const int	nrows = std::distance(row, rowe);
    if (nrows < _winSizeV)
	return;

    const int	ncols = TU::size(*row);
    if (ncols < _winSizeH)
	return;

    _buf.resize(ncols, outSizeV(nrows));
    _buf_pos.resize(_buf.nrow(), _buf.ncol());

  // Accumultate vetically.
    const dim3	threads(BlockDim, BlockDim);
    dim3	blocks(gridDim(ncols, threads.x), gridDim(nrows, threads.y));
    device::extrema_filterV<ExtremaFilter2><<<blocks, threads>>>(
	cuda::make_range(row, nrows),
	cuda::make_range(_buf.begin(), _buf.nrow()),
	cuda::make_range(_buf_pos.begin(), _buf_pos.nrow()),
	_winSizeV, compare);

  // Accumulate horizontally.
    blocks.x = gridDim(_buf.ncol(), threads.x);
    blocks.y = gridDim(_buf.nrow(), threads.y);
    device::extrema_filterH<ExtremaFilter2><<<blocks, threads>>>(
	cuda::make_range(_buf.cbegin(),	    _buf.nrow()),
	cuda::make_range(_buf_pos.cbegin(), _buf_pos.nrow()),
	cuda::slice(rowO, (shift ? offsetV() : 0), outSizeV(nrows),
			  (shift ? offsetH() : 0), outSizeH(ncols)),
	cuda::slice(rowP, (shift ? offsetV() : 0), outSizeV(nrows),
			  (shift ? offsetH() : 0), outSizeH(ncols)),
	_winSizeH, compare);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
