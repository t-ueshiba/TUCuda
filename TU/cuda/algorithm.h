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
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#pragma once

#include "TU/cuda/allocator.h"
#include "TU/cuda/iterator.h"
#include "TU/cuda/vec.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  struct BlockTraits							*
************************************************************************/
template <size_t BLOCK_DIM_X=32, size_t BLOCK_DIM_Y=16>
struct BlockTraits
{
  //! 1ブロックあたりのスレッド数(x方向)    
    constexpr static size_t	BlockDimX = BLOCK_DIM_X;
  //! 1ブロックあたりのスレッド数(y方向)    
    constexpr static size_t	BlockDimY = BLOCK_DIM_Y;
};
    
/************************************************************************
*  global functions							*
************************************************************************/
static inline size_t
divUp(size_t dim, size_t blockDim)
{
    return (dim + blockDim - 1)/blockDim;
}
    
#if defined(__NVCC__)
inline std::ostream&
operator <<(std::ostream& out, const dim3& d)
{
    return out << '[' << d.x << ' ' << d.y << ' ' << d.z << ']';
}
    
//! デバイス関数を納める名前空間
namespace device
{
  /**********************************************************************
  *  __device__ functions						*
  **********************************************************************/
  //! スレッドブロック中の1次元領域をコピーする
  /*!
    \param src		コピー元の1次元領域
    \param dst		コピー先の1次元配列
  */
  template <class IN, class T> __device__ static inline void
  loadLine(const range<IN>& src, T dst[])
  {
      for (int tx = threadIdx.x; tx < src.size(); tx += blockDim.x)
	  dst[tx] = src[tx];
  }

  //! スレッドブロック中の2次元領域をコピーする
  /*!
    \param src		コピー元の2次元領域
    \param dst		コピー先の2次元配列
  */
  template <class IN, class T, size_t W> __device__ static inline void
  loadTile(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.x)
	      dst[ty][tx] = row[tx];
      }
  }

  //! スレッドブロック中の2次元領域を転置コピーする
  /*!
    \param src		コピー元の2次元領域
    \param dst		コピー先の2次元配列
  */
  template <class IN, class T, size_t W> __device__ static inline void
  loadTileT(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.x)
	      dst[tx][ty] = row[tx];
      }
  }
}	// namespace device
#endif

/************************************************************************
*  copyToConstantMemory(ITER begin, ITER end, T* dst)			*
************************************************************************/
//! CUDAの定数メモリ領域にデータをコピーする．
/*!
  \param begin	コピー元データの先頭を指す反復子
  \param end	コピー元データの末尾の次を指す反復子
  \param dst	コピー先の定数メモリ領域を指すポインタ
*/
template <class ITER, class T> inline void
copyToConstantMemory(ITER begin, ITER end, T* dst)
{
    if (begin < end)
	cudaMemcpyToSymbol(reinterpret_cast<const char*>(dst), &(*begin),
			   std::distance(begin, end)*sizeof(T), 0,
			   cudaMemcpyHostToDevice);
}

/************************************************************************
*  subsample<BLOCK_TRAITS>(IN in, IN ie, OUT out)			*
************************************************************************/
//! CUDAによって2次元配列を水平／垂直方向それぞれ1/2に間引く．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT> void
subsample(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
namespace device
{
  template <class IN, class OUT>
  __global__ static void
  subsample(range<range_iterator<IN> > in, range<range_iterator<OUT> > out)
  {
      const int	x = blockIdx.x*blockDim.x + threadIdx.x;
      const int	y = blockIdx.y*blockDim.y + threadIdx.y;

      if (2*y < in.size() && 2*x < in.begin().size())
	  out[y][x] = in[2*y][2*x];
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT> void
subsample(IN in, IN ie, OUT out)
{
    const int	nrow = std::distance(in, ie);
    if (nrow < 2)
	return;

    const int	ncol = TU::size(*in);
    if (ncol < 2)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol/2, threads.x), divUp(nrow/2, threads.y));
    device::subsample<<<blocks, threads>>>(cuda::make_range(in,  nrow),
					   cuda::make_range(out, nrow/2));
}
#endif

/************************************************************************
*  op3x3<BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)			*
************************************************************************/
//! CUDAによって2次元配列に対して3x3近傍演算を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
  \param op	3x3近傍演算子
*/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP>
void	op3x3(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  template <class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  op3x3(range<range_iterator<IN> > in, range<range_iterator<OUT> > out, OP op)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      __shared__ value_type	in_s[BLOCK_TRAITS::BlockDimY + 2]
				    [BLOCK_TRAITS::BlockDimX + 2 + 1];
      loadTile(slice(in.cbegin(),
		     y0, ::min(blockDim.y + 2, in.size() - y0),
		     x0, ::min(blockDim.x + 2, in.begin().size() - x0)),
	       in_s);
      __syncthreads();

    // 共有メモリに保存した原画像から現在画素に対するフィルタ出力を計算
      const int	tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int	x  = x0 + tx;
      const int	y  = y0 + ty;
      if (y + 2 < in.size() && x + 2 < in.begin().size())
	  out[y + 1][x + 1] = op(in_s[ty]     + tx,
				 in_s[ty + 1] + tx,
				 in_s[ty + 2] + tx);
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
op3x3(IN in, IN ie, OUT out, OP op)
{
    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = TU::size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::op3x3<BLOCK_TRAITS><<<blocks, threads>>>(
	cuda::make_range(in, nrow), cuda::make_range(out, nrow), op);
}
#endif

/************************************************************************
*  opNxN<N, BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)			*
************************************************************************/
template <size_t N, class BLOCK_TRAITS=BlockTraits<>,
	  class IN, class OUT, class OP>
void	opNxN(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  template <int N, class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  opNxN(range<range_iterator<IN> > in, range<range_iterator<OUT> > out, OP op)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      constexpr int		Stride = BLOCK_TRAITS::BlockDimX + N - 1;
      __shared__ value_type	in_s[BLOCK_TRAITS::BlockDimY + N - 1][Stride];
      loadTile(slice(in.cbegin(),
		     y0, ::min(blockDim.y + N - 1, in.size() - y0),
		     x0, ::min(blockDim.x + N - 1, in.begin().size() - x0)),
	       in_s);
      __syncthreads();

    // 共有メモリに保存した原画像から現在画素に対するフィルタ出力を計算
      const int	tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int	x  = x0 + tx;
      const int	y  = y0 + ty;
      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(y, x, in.size(), in.begin().size(),
			 in_s[ty] + tx, Stride);
  }
}	// namespace device

template <size_t N, class BLOCK_TRAITS, class IN, class OUT, class OP> void
opNxN(IN in, IN ie, OUT out, OP op)
{
    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = TU::size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::opNxN<N, BLOCK_TRAITS><<<blocks, threads>>>(
	cuda::make_range(in, nrow), cuda::make_range(out, nrow), op);
}
#endif

/************************************************************************
*  transpose<BLOCK_DIM>(IN in, IN ie, OUT out)				*
************************************************************************/
//! CUDAによって2次元配列の転置処理を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <size_t BLOCK_DIM=32, class IN, class OUT> void
transpose(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
namespace device
{
  template <size_t BLOCK_DIM, class IN, class OUT>
  __global__ static void
  transpose(range<range_iterator<IN> > in, range<range_iterator<OUT> > out)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      __shared__ value_type	tile[BLOCK_DIM][BLOCK_DIM + 1];

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);

      tile[threadIdx.y][threadIdx.x] = in[y0 + threadIdx.y][x0 + threadIdx.x];
      __syncthreads();

      out[x0 + threadIdx.y][y0 + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
  }
}	// namespace device

namespace detail
{
  template <size_t BLOCK_DIM, class IN, class OUT> static void
  transpose(IN in, IN ie, OUT out, size_t i, size_t j)
  {
      size_t	r = std::distance(in, ie);
      if (r < 1)
	  return;

      size_t	c = TU::size(*in) - j;
      if (c < 1)
	  return;

      const auto	blockDim = std::min({BLOCK_DIM, r, c});
      const dim3	threads(blockDim, blockDim);
      const dim3	blocks(c/threads.x, r/threads.y);
      cuda::device::transpose<BLOCK_DIM><<<blocks, threads>>>(
	  cuda::slice(in, 0, r, j, c), cuda::slice(out, 0, c, i, r));	// 左上

      r = blocks.y*threads.y;
      c = blocks.x*threads.x;

      auto	in_n = in;
      std::advance(in_n, r);
      auto	out_n = out;
      std::advance(out_n, c);
      transpose<BLOCK_DIM>(in,   in_n, out_n, i,     j + c);	// 右上
      transpose<BLOCK_DIM>(in_n, ie,   out,   i + r, j);	// 下
  }
}	// namesapce detail

template <size_t BLOCK_DIM, class IN, class OUT> inline void
transpose(IN in, IN ie, OUT out)
{
    detail::transpose<BLOCK_DIM>(in, ie, out, 0, 0);
}
#endif

/************************************************************************
*  transform2<BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)		*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP> void
transform2(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  template <class IN, class OUT, class OP> __global__ static void
  transform2(range<range_iterator<IN> > in,
	     range<range_iterator<OUT> > out, OP op, std::true_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(in[y][x]);
  }
    
  template <class IN, class OUT, class OP> __global__ static void
  transform2(range<range_iterator<IN> > in,
	     range<range_iterator<OUT> > out, OP op, std::false_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(y, x, in[y][x]);
  }
}	// namespace device

namespace detail
{
  template <class OP, class T>
  static auto	check_unarity(OP op, T arg)
		    -> decltype(op(arg), std::true_type());
  template <class OP, class T>
  static auto	check_unarity(OP op, T arg)
		    -> decltype(op(0, 0, arg), std::false_type());
  template <class OP, class T>
  using is_unary = decltype(check_unarity(std::declval<OP>(),
					  std::declval<T>()));
}
    
template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
transform2(IN in, IN ie, OUT out, OP op)
{
    using value_type	= typename std::iterator_traits<IN>::value_type
							   ::value_type;

    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = TU::size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::transform2<<<blocks, threads>>>(
	cuda::make_range(in,  nrow), cuda::make_range(out, nrow), op,
	std::integral_constant<bool,
			       detail::is_unary<OP, value_type>::value>());
}
#endif

/************************************************************************
*  fill<BLOCK_TRAITS>(OUT out, OUT oe, T val)				*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class OUT, class T> void
fill(OUT out, OUT oe, T val)						;

#if defined(__NVCC__)
namespace device
{
  template <class OUT, class T> __global__ void
  fill(range<range_iterator<OUT> > out, T val)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < out.size() && x < out.begin().size())
	  out[y][x] = val;
  }
}	// namespace cuda

template <class BLOCK_TRAITS, class OUT, class T> void
fill(OUT out, OUT oe, T val)
{
    const int	nrow = std::distance(out, oe);
    if (nrow < 1)
	return;

    const int	ncol = TU::size(*out);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::fill<<<blocks, threads>>>(cuda::make_range(out, nrow), val);
}
#endif
}	// namespace cuda
}	// namespace TU
