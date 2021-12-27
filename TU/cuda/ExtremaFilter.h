/*!
  \file		ExtreamaFilter.h
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
*  class extreme_value<COMP>						*
************************************************************************/
template <class COMP>
class extreme_value
{
  public:
    using argument_type	= typename COMP::first_argument_type;
    using result_type	= argument_type;
    
  public:
    __host__	extreme_value() :_comp()				{}
    
    __device__
    bool	operator ()(const argument_type& a,
			    const argument_type& b) const
		{
		    return _comp(a, b);
		}
    __device__
    result_type	operator ()(int x, int y, const argument_type& v) const
		{
		    return v;
		}

  private:
    const COMP	_comp;
};
    
/************************************************************************
*  class extreme_value_position<COMP, POS>				*
************************************************************************/
template <class COMP, class POS=int2>
class extreme_value_position
{
  public:
    using argument_type	= typename COMP::first_argument_type;
    using position_type	= POS;
    using result_type	= thrust::tuple<argument_type, position_type>;
    
  public:
    __host__	extreme_value_position() :_comp()			{}
    
    __device__
    bool	operator ()(const argument_type& a,
			    const argument_type& b) const
		{
		    return _comp(a, b);
		}
    __device__
    bool	operator ()(const result_type& a, const result_type& b) const
		{
		    return _comp(thrust::get<0>(a), thrust::get<0>(b));
		}
    __device__
    result_type	operator ()(int x, int y, const argument_type& v) const
		{
		    return {v, position_type{x, y}};
		}
    __device__
    result_type	operator ()(int x, int y, const result_type& v) const
		{
		    return v;
		}

  private:
    const COMP	_comp;
};
    
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
template <class FILTER, class IN, class OUT, class OP>
__global__ void
extrema_filter(range<range_iterator<IN> >  in,
	       range<range_iterator<OUT> > out, int winSize, OP op)
{
    using in_type	= typename std::iterator_traits<IN>::value_type;
    using out_type	= typename FILTER::value_type;

    const int	winSize1 = winSize - 1;
    const int	x0	 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const int	y0	 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅
    const int	xsiz	 = ::min(blockDim.x, in.begin().size() - x0);
    const int	ysiz	 = ::min(blockDim.y + winSize1, in.size() - y0);
      
    __shared__ in_type	in_s[FILTER::BlockDim + FILTER::WinSizeMax - 1]
			    [FILTER::BlockDim + 1];
    loadTile(slice(in.cbegin(), y0, ysiz, x0, xsiz), in_s);
    __syncthreads();

    __shared__ out_type	out_s[FILTER::BlockDim][FILTER::BlockDim + 1];

    const int	ye = ysiz - winSize1;

    if (threadIdx.x < xsiz && ye > 0)
    {
	if (threadIdx.y == 0)
	{
	    deque<int, FILTER::WinSizeMax> q;
	    q.push_back(0);

	    for (int y = 1; y != ysiz; ++y)
	    {
		while (op(in_s[y][threadIdx.x], in_s[q.back()][threadIdx.x]))
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
		    	= op(x0 + threadIdx.x, y0 + q.front(),
		    	     in_s[q.front()][threadIdx.x]);
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

    template <class ROW, class ROW_O, class OP>
    void		convolve(ROW row, ROW rowe, ROW_O rowO,
				 OP op, bool shift=false)	const	;
    template <class ROW, class ROW_O, class ROW_P, class OP>
    void		extrema(ROW row, ROW rowe, ROW_O rowO, ROW_P rowP,
				OP op, bool shift=false) const;

  private:
    size_t		_winSizeV;
    size_t		_winSizeH;
    mutable Array2<T>	_buf;
};

#if defined(__NVCC__)
template <class T, class BLOCK_TRAITS, size_t WMAX>
template <class ROW, class ROW_O, class OP> void
ExtremaFilter2<T, BLOCK_TRAITS, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO,
						OP op, bool shift) const
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
	cuda::make_range(_buf.begin(), _buf.nrow()),
	_winSizeV, op);

  // Accumulate horizontally,
    blocks.x = gridDim(_buf.ncol(), threads.x);
    blocks.y = gridDim(_buf.nrow(), threads.y);
    device::extrema_filter<ExtremaFilter2><<<blocks, threads>>>(
	cuda::make_range(_buf.cbegin(), _buf.nrow()),
	cuda::slice(rowO, (shift ? offsetV() : 0), outSizeV(nrows),
			  (shift ? offsetH() : 0), outSizeH(ncols)),
	_winSizeH, op);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
